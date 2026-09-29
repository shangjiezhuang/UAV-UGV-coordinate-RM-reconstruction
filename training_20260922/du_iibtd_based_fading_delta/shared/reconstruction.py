"""Shared reconstruction lifecycle for quantized and unquantized observations."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None
from du_iibtd_based_fading_delta.uncertainty_refresh import uncertainty_norm
from du_iibtd_based_fading_delta.shared.noquant.active_sampling import (
    ensemble_reconstruct_maps, fuse_observations_by_grid, incremental_refresh_ensemble_models,
    release_reconstruction_model, select_reconstruction_outer_iters,
)
Config = Any
SpectrumSample = Any

@dataclass
class RadioMapState:
    spectrum_map: np.ndarray
    nmse: float
    last_update_step: int

class IIBTD_opt:
    """
    DU-IIBTD wrapper using member-only ensemble mean maps for the main radio map.
    """

    def __init__(
        self,
        config: Config,
        grid_coords: np.ndarray,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        i_mask: np.ndarray,
        n_sources: int = 1,
    ):
        self.config = config
        self.Nx, self.Ny = config.scene.grid_size
        self.K = config.scene.total_freq_bands_nums
        self.grid_coords = np.asarray(grid_coords, dtype=float)
        self.bounds = bounds
        self.I_mask = np.asarray(i_mask, dtype=bool)
        self.n_sources = int(n_sources)

        self.btd = None
        self._ground_truth: Optional[np.ndarray] = None
        self._pending_samples: List[SpectrumSample] = []
        self._all_samples: List[SpectrumSample] = []
        self._has_fit = False
        self._current_map: Optional[np.ndarray] = None
        self._latest_ensemble_mean_map: Optional[np.ndarray] = None
        self._latest_ensemble_var_map: Optional[np.ndarray] = None
        self._latest_ensemble_info: Dict[str, object] = {}
        self._latest_ensemble_sample_count: int = 0
        self._ensemble_member_models: List[object] = []
        self._ensemble_member_observation_counts = np.empty((0,), dtype=int)
        self._reconstruct_round: int = 0
        self._effective_grid_positions: set[Tuple[int, int]] = set()
        self._last_fusion_meta: Dict[str, object] = {
            "raw_count": 0,
            "fused_count": 0,
            "compression_ratio": 1.0,
            "raw_per_fused": np.empty((0,), dtype=int),
        }
        self.reset()

    def add_samples(self, samples: List[SpectrumSample]) -> None:
        self._pending_samples.extend(samples)
        self._all_samples.extend(samples)
        for sample in samples:
            pos = np.asarray(sample.position, dtype=float).reshape(2)
            gx = int(np.clip(np.round(pos[0]), 0, self.Nx - 1))
            gy = int(np.clip(np.round(pos[1]), 0, self.Ny - 1))
            self._effective_grid_positions.add((gx, gy))

    @staticmethod
    def _unique_models(models: List[object]) -> List[object]:
        unique_models: List[object] = []
        seen_ids: set[int] = set()
        for model in models:
            if model is None:
                continue
            model_id = id(model)
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            unique_models.append(model)
        return unique_models

    def _tracked_models(self) -> List[object]:
        return self._unique_models([self.btd, *list(self._ensemble_member_models or [])])

    def _release_models(self, models: List[object], clear_cuda_cache: bool = False) -> None:
        released_cuda = False
        for model in self._unique_models(models):
            released_cuda = release_reconstruction_model(model) or released_cuda
        if clear_cuda_cache and released_cuda and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_stale_models(
        self,
        previous_models: List[object],
        retained_models: List[object],
    ) -> None:
        retained_ids = {id(model) for model in self._unique_models(retained_models)}
        stale_models = [model for model in self._unique_models(previous_models) if id(model) not in retained_ids]
        self._release_models(stale_models)

    def reconstruct(self) -> RadioMapState:
        if self._pending_samples:
            obs_locs, gamma, omega = self._samples_to_arrays(self._all_samples)
            fused = fuse_observations_by_grid(obs_locs, gamma, omega, self.Nx, self.Ny)
            self._reconstruct_round += 1
            if not self._ensemble_member_models:
                outputs = self._fit_full_ensemble(
                    obs_locs, gamma, omega, fused[3],
                    self.config.training.seed + 31 * self._reconstruct_round + len(self._all_samples),
                )
                mode = "ensemble_refresh_missing_members"
            else:
                pending = self._samples_to_arrays(self._pending_samples)
                new_locs, new_gamma, new_omega, _ = fuse_observations_by_grid(
                    *pending, self.Nx, self.Ny,
                )
                planner = self.config.planner
                outputs = incremental_refresh_ensemble_models(
                    member_models=self._ensemble_member_models,
                    member_observation_counts=self._ensemble_member_observation_counts,
                    new_obs_locs=new_locs, new_gamma=new_gamma, new_omega=new_omega,
                    fused_obs_locs=fused[0], fused_gamma=fused[1], fused_omega=fused[2],
                    n_sources=self.n_sources, grid_size=(self.Nx, self.Ny),
                    grid_points=self.grid_coords, bounds=self.bounds, i_mask=self.I_mask,
                    n_outer_iter=planner.incremental_outer_iters,
                    max_svt_iter=planner.incremental_max_svt_iters,
                    quality_weighted=bool(planner.ensemble_quality_weighted), mu=planner.iibtd_mu,
                    member_kernel_bandwidths=self._latest_ensemble_info.get("member_kernel_bandwidths"),
                    solver_backend=planner.iibtd_backend, solver_device=self._solver_device(),
                    du_iibtd_checkpoints=planner.du_iibtd_checkpoints,
                    du_iibtd_min_sensors_for_update=planner.du_iibtd_min_sensors_for_update,
                    du_iibtd_update_batch_size=planner.du_iibtd_update_batch_size,
                    member_nus=self._latest_ensemble_info.get("member_nus"), fusion_meta=fused[3],
                )
                mode = "ensemble_incremental"
            self._commit_ensemble(outputs, fused[3], mode)
            self._pending_samples.clear()
        return self._radio_map_state()

    def _solver_device(self):
        value = self.config.planner.iibtd_device
        return self.config.training.device if str(value).strip().lower() == "auto" else value

    def _fit_full_ensemble(self, obs_locs, gamma, omega, fusion_meta, seed):
        planner = self.config.planner
        return ensemble_reconstruct_maps(
            obs_locs=obs_locs, gamma=gamma, omega=omega, n_sources=self.n_sources,
            grid_size=(self.Nx, self.Ny), grid_points=self.grid_coords, bounds=self.bounds,
            i_mask=self.I_mask, m_ens=planner.ensemble_size, seed=seed,
            member_max_iter=select_reconstruction_outer_iters(max(1, int(fusion_meta.get("fused_count", 0))), warmstart=False),
            quality_weighted=bool(planner.ensemble_quality_weighted), mu=planner.iibtd_mu,
            solver_backend=planner.iibtd_backend, solver_device=self._solver_device(),
            du_iibtd_checkpoints=planner.du_iibtd_checkpoints,
            du_iibtd_min_sensors_for_update=planner.du_iibtd_min_sensors_for_update,
            du_iibtd_update_batch_size=planner.du_iibtd_update_batch_size,
            member_init_jitter_scale=planner.ensemble_init_jitter_scale, return_info=True,
        )

    def _commit_ensemble(self, outputs, fusion_meta, mode):
        mean_map, var_map, _, info = outputs
        before = (uncertainty_norm(self._latest_ensemble_var_map)
                  if self._latest_ensemble_var_map is not None else float("nan"))
        after = uncertainty_norm(var_map)
        switched = mode == "ensemble_mode_switch_refresh"
        info = dict(info)
        info.update(recon_mode=mode, mode_switch_refresh_triggered=switched,
                    reconstruction_refresh_mode="local_to_global",
                    pre_refresh_uncertainty=before if switched else float("nan"),
                    post_refresh_uncertainty=after,
                    # Passive fields retained for existing result readers; never used as triggers.
                    full_refresh_due=False, uncertainty_refresh_triggered=False,
                    uncertainty_refresh_ratio=0.0, uncertainty_refresh_streak=0,
                    uncertainty_refresh_reference_before=before,
                    uncertainty_refresh_reference_after=after,
                    uncertainty_relative_increase=float("nan"))
        next_models = list(info.get("member_models", self._ensemble_member_models))
        self._release_stale_models(self._tracked_models(), next_models)
        self.btd = next_models[0] if next_models else None
        self._has_fit = bool(next_models)
        self._ensemble_member_models = next_models
        self._ensemble_member_observation_counts = np.asarray(
            info.get("member_observation_counts", self._ensemble_member_observation_counts), dtype=int,
        ).copy()
        self._latest_ensemble_info = info
        self._last_fusion_meta = info.get("fusion_meta", fusion_meta)
        self._current_map = np.asarray(mean_map, dtype=float).copy()
        self._latest_ensemble_mean_map = self._current_map
        self._latest_ensemble_var_map = np.maximum(np.asarray(var_map, dtype=float), 0.0)
        self._latest_ensemble_sample_count = len(self._all_samples)

    def _radio_map_state(self):
        current_map = self.get_current_map()
        return RadioMapState(current_map, self._compute_nmse(current_map), len(self._all_samples))

    def full_refit_for_mode_switch(self) -> RadioMapState:
        """Refit received observations once, without advancing the data-update round."""
        if not self._all_samples or self._pending_samples:
            raise RuntimeError("Mode-switch refit requires a completed incremental data update")
        if self._last_switch_refit_round == self._reconstruct_round:
            raise RuntimeError("Duplicate mode-switch refit for the same data update")
        obs_locs, gamma, omega = self._samples_to_arrays(self._all_samples)
        _, _, _, fusion_meta = fuse_observations_by_grid(obs_locs, gamma, omega, self.Nx, self.Ny)
        outputs = self._fit_full_ensemble(
            obs_locs, gamma, omega, fusion_meta,
            self.config.training.seed + 61000 + 53 * self._reconstruct_round + len(self._all_samples),
        )
        self._commit_ensemble(outputs, fusion_meta, "ensemble_mode_switch_refresh")
        self._last_switch_refit_round = self._reconstruct_round
        return self._radio_map_state()

    def get_current_map(self) -> np.ndarray:
        if self._current_map is not None:
            return np.asarray(self._current_map, dtype=float).copy()
        if self.btd is not None and getattr(self.btd, "H_hat", None) is not None:
            return np.asarray(self.btd.H_hat, dtype=float).copy()
        return np.zeros((self.Nx, self.Ny, self.K), dtype=float)

    def get_btd_model(self):
        return self.btd if self._has_fit else None

    def get_latest_ensemble_outputs(
        self,
        expected_sample_count: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._latest_ensemble_mean_map is None or self._latest_ensemble_var_map is None:
            return None
        if (
            expected_sample_count is not None
            and int(expected_sample_count) != int(self._latest_ensemble_sample_count)
        ):
            return None
        return (
            np.asarray(self._latest_ensemble_mean_map, dtype=float).copy(),
            np.asarray(self._latest_ensemble_var_map, dtype=float).copy(),
        )

    def get_latest_ensemble_diagnostics(self) -> Dict[str, object]:
        if not self._latest_ensemble_info:
            return {}
        keys = (
            "recon_mode",
            "mode_switch_refresh_triggered",
            "reconstruction_refresh_mode",
            "ensemble_observation_mode",
            "member_observation_counts",
            "member_kernel_bandwidths",
            "full_refresh_due",
            "uncertainty_refresh_triggered",
            "uncertainty_refresh_ratio",
            "uncertainty_refresh_reference_before",
            "uncertainty_refresh_reference_after",
            "uncertainty_relative_increase",
            "uncertainty_refresh_streak",
            "pre_refresh_uncertainty",
            "post_refresh_uncertainty",
        )
        diagnostics: Dict[str, object] = {}
        for key in keys:
            if key in self._latest_ensemble_info:
                diagnostics[key] = self._latest_ensemble_info[key]
        return diagnostics

    def get_effective_sample_count(self) -> int:
        return int(len(self._effective_grid_positions))

    def get_compression_ratio(self) -> float:
        return float(self._last_fusion_meta.get("compression_ratio", 1.0))

    def reset(self) -> None:
        self._release_models(self._tracked_models())
        self._pending_samples.clear()
        self._all_samples.clear()
        self._has_fit = False
        self._current_map = None
        self._latest_ensemble_mean_map = None
        self._latest_ensemble_var_map = None
        self._latest_ensemble_info = {}
        self._latest_ensemble_sample_count = 0
        self._ensemble_member_models = []
        self._ensemble_member_observation_counts = np.empty((0,), dtype=int)
        self._reconstruct_round = 0
        self._last_switch_refit_round = -1
        self._effective_grid_positions.clear()
        self._last_fusion_meta = {
            "raw_count": 0,
            "fused_count": 0,
            "compression_ratio": 1.0,
            "raw_per_fused": np.empty((0,), dtype=int),
        }
        self.btd = None

    def close(self) -> None:
        self._release_models(self._tracked_models(), clear_cuda_cache=True)
        self.btd = None
        self._ensemble_member_models = []
        self._ensemble_member_observation_counts = np.empty((0,), dtype=int)
        self._latest_ensemble_info = {}
        self._current_map = None
        self._latest_ensemble_mean_map = None
        self._latest_ensemble_var_map = None

    def get_num_samples(self) -> int:
        return len(self._all_samples)

    def set_ground_truth(self, gt: np.ndarray) -> None:
        self._ground_truth = np.asarray(gt, dtype=float)

    def _compute_nmse(self, est_map: Optional[np.ndarray] = None) -> float:
        if self._ground_truth is None:
            return 1.0
        est = self.get_current_map() if est_map is None else np.asarray(est_map, dtype=float)
        gt = self._ground_truth
        eval_mask = np.asarray(self.I_mask, dtype=bool)
        if eval_mask.shape == gt.shape[:2] and np.any(eval_mask):
            est_eval = est[eval_mask]
            gt_eval = gt[eval_mask]
        else:
            est_eval = est
            gt_eval = gt
        return float(np.sum((est_eval - gt_eval) ** 2) / (np.sum(gt_eval ** 2) + 1e-10))

    @staticmethod
    def _samples_to_arrays(
        samples: List[SpectrumSample],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_locs = np.asarray(
            [np.asarray(sample.position, dtype=float).reshape(2) for sample in samples],
            dtype=float,
        )
        gamma = np.asarray(
            [np.asarray(sample.gamma, dtype=float) for sample in samples],
            dtype=float,
        )
        omega = np.asarray(
            [np.asarray(sample.omega, dtype=np.int32) for sample in samples],
            dtype=np.int32,
        )
        return obs_locs, gamma, omega
