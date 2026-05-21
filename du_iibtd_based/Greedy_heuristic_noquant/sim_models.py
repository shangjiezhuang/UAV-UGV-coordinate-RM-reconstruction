"""
Simulation data models for Greedy active sensing.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch exists in training env
    torch = None

from active_sampling import (
    ensemble_reconstruct_maps,
    fuse_observations_by_grid,
    incremental_refresh_ensemble_models,
    quantize_to_8bit,
    release_reconstruction_model,
    select_reconstruction_outer_iters,
)
from config import Config

_SHARED_MEMBER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_DIR = os.path.dirname(_SHARED_MEMBER_DIR)
for _path in (_CODE_DIR, _SHARED_MEMBER_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@dataclass
class SpectrumSample:
    position: np.ndarray
    freq_group_idx: int
    freq_band_indices: np.ndarray
    measurements: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    timestamp: int


@dataclass
class RadioMapState:
    spectrum_map: np.ndarray
    nmse: float
    last_update_step: int


@dataclass
class UncertaintyMap:
    spatial_uncertainty: np.ndarray
    frequency_uncertainty: np.ndarray
    joint_uncertainty: np.ndarray


@dataclass
class ChannelInfo:
    path_loss_db: float
    channel_gain: float
    los: bool
    capacity_bps: float
    snr_db: float


class SimDataGen:
    """
    Load and serve sensing/communication data from RadioSeerSelect.
    """

    def __init__(
        self,
        config: Config,
        seed: int = 42,
        precomputed_data: Optional[Dict] = None,
    ):
        self.config = config
        self.seed = int(seed)
        self.K = int(config.scene.total_freq_bands_nums)
        self.rng = np.random.RandomState(seed)

        if precomputed_data is None:
            self._data = self._generate_sim_data(seed)
        else:
            self._data = precomputed_data

        self.ground_truth = np.asarray(self._data["H"], dtype=float)
        if self.ground_truth.ndim != 3:
            raise ValueError(
                f"Ground truth tensor must be 3D, got shape {self.ground_truth.shape}"
            )
        self.Nx, self.Ny = tuple(int(v) for v in self.ground_truth.shape[:2])
        self.config.scene.grid_size = (self.Nx, self.Ny)
        self.grid_coords = np.asarray(self._data["grid_coords"], dtype=float)
        self.I_mask = np.asarray(self._data["I_mask"], dtype=bool)
        quant_meta = self._data.get("Gamma_quant_meta")
        if quant_meta is None:
            _, quant_meta = quantize_to_8bit(
                np.asarray(self.ground_truth, dtype=float).reshape(-1, self.K)
            )
        self.quant_meta = dict(quant_meta)
        self.bounds = self._data["bounds"]
        self.building_mask = np.asarray(
            self._data.get("building_mask", np.zeros((self.Nx, self.Ny), dtype=bool)),
            dtype=bool,
        )
        self.non_building_mask = np.asarray(
            self._data.get("non_building_mask", ~self.building_mask),
            dtype=bool,
        )
        self.building_heights = np.asarray(
            self._data.get(
                "building_heights",
                self.building_mask.astype(float) * float(self.config.scene.building_height_m),
            ),
            dtype=float,
        )
        self.radioseer_metadata = dict(self._data.get("radioseer_metadata", {}))
        self.radioseer_row = dict(self._data.get("radioseer_row", {}))

        if self.ground_truth.shape != (self.Nx, self.Ny, self.K):
            raise ValueError(
                f"Ground truth shape {self.ground_truth.shape} != ({self.Nx}, {self.Ny}, {self.K})"
            )
        if self.building_heights.shape != (self.Nx, self.Ny):
            raise ValueError(
                f"building_heights shape {self.building_heights.shape} != ({self.Nx}, {self.Ny})"
            )
    def _generate_sim_data(self, seed: int) -> Dict:
        return self._generate_radioseer_data(seed)

    def _resolve_radioseer_root(self) -> Path:
        configured = Path(str(self.config.scene.radioseer_root).strip())
        if configured.is_absolute():
            root = configured
        else:
            root = Path(_CODE_DIR) / configured
        if not (root / "manifest.csv").exists():
            raise FileNotFoundError(
                f"RadioSeerSelect manifest not found under {root}."
            )
        return root

    @staticmethod
    def _load_radioseer_manifest(dataset_root: Path) -> List[Dict[str, str]]:
        manifest_path = dataset_root / "manifest.csv"
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No RadioSeer samples found in {manifest_path}")
        return rows

    @staticmethod
    def _make_grid_coords(width: int, height: int) -> np.ndarray:
        gx, gy = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel()]).astype(float)

    def _generate_power_spectrum(self, rng: np.random.RandomState, n_sinc: int = 2) -> np.ndarray:
        k_vals = np.arange(1, self.K + 1, dtype=float)
        phi = np.zeros(self.K, dtype=np.float64)
        for _ in range(max(1, int(n_sinc))):
            amplitude = rng.uniform(0.5, 2.0)
            center = rng.randint(1, self.K + 1)
            width = rng.uniform(2.0, 4.0)
            phi += amplitude * np.sinc((k_vals - center) / width) ** 2
        phi_sum = float(np.sum(phi))
        if phi_sum <= 1e-12:
            return np.ones(self.K, dtype=np.float64)
        return phi * (float(self.K) / phi_sum)

    @staticmethod
    def _prepare_radioseer_field(
        gain_img: np.ndarray,
    ) -> np.ndarray:
        """Decode RadioSeer gain_DPM directly into normalized pixel space."""
        gain_norm_xy = np.clip(np.asarray(gain_img, dtype=np.float64).T / 255.0, 0.0, 1.0)
        field_xy = np.maximum(gain_norm_xy, 1e-6)
        return field_xy

    def _build_building_height_map(self, building_mask_xy: np.ndarray) -> np.ndarray:
        mask = np.asarray(building_mask_xy, dtype=bool)
        heights = np.zeros(mask.shape, dtype=float)
        default_height = max(0.0, float(self.config.scene.building_height_m))
        if default_height <= 0.0 or not np.any(mask):
            return heights
        heights[mask] = default_height
        return heights

    def _generate_radioseer_data(self, seed: int) -> Dict:
        dataset_root = self._resolve_radioseer_root()
        manifest_rows = self._load_radioseer_manifest(dataset_root)
        sample_index = int(self.config.scene.radioseer_sample_index)
        if sample_index < 0:
            sample_index += len(manifest_rows)
        if sample_index < 0 or sample_index >= len(manifest_rows):
            raise IndexError(
                f"scene.radioseer_sample_index={sample_index} is out of range 0..{len(manifest_rows) - 1}"
            )

        row = dict(manifest_rows[sample_index])
        metadata_path = dataset_root / row["metadata_json"]
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        arrays_rel_path = metadata.get("data_files", {}).get("arrays_npz")
        if not arrays_rel_path:
            raise KeyError(
                f"RadioSeer metadata {metadata_path} does not contain data_files.arrays_npz"
            )

        npz_path = dataset_root / str(arrays_rel_path)
        with np.load(npz_path) as arrays:
            gain_img = np.asarray(arrays["gain_DPM"], dtype=np.float64)
            # RadioSeerSelect stores these masks as boolean arrays in the NPZ.
            # Casting to bool also preserves compatibility with nonzero uint8 masks.
            building_mask_img = np.asarray(arrays["building_mask"], dtype=bool)
            non_building_mask_img = np.asarray(arrays["non_building_mask"], dtype=bool)

        field_xy = self._prepare_radioseer_field(gain_img=gain_img)
        building_mask_xy = building_mask_img.T.astype(bool)
        non_building_mask_xy = non_building_mask_img.T.astype(bool)
        if building_mask_xy.shape != non_building_mask_xy.shape:
            raise ValueError(
                "RadioSeerSelect building/non-building masks must share the same shape, got "
                f"{building_mask_xy.shape} vs {non_building_mask_xy.shape}"
            )
        if np.any(building_mask_xy & non_building_mask_xy):
            raise ValueError("RadioSeerSelect building_mask and non_building_mask overlap.")
        if not np.all(building_mask_xy | non_building_mask_xy):
            raise ValueError(
                "RadioSeerSelect building_mask and non_building_mask do not cover the full grid."
            )
        building_heights_xy = self._build_building_height_map(building_mask_xy)

        width, height = field_xy.shape
        phi = self._generate_power_spectrum(np.random.RandomState(int(seed)))[np.newaxis, :]
        spatial_field = field_xy[np.newaxis, :, :]
        ground_truth = np.einsum("rxy,rk->xyk", spatial_field, phi)
        grid_coords = self._make_grid_coords(width, height)
        _, quant_meta = quantize_to_8bit(np.asarray(ground_truth, dtype=float).reshape(-1, self.K))
        bounds = ((0.0, float(width - 1)), (0.0, float(height - 1)))

        return {
            "config": {
                "dataset_root": str(dataset_root),
                "sample_index": int(sample_index),
                "sample_tag": str(row.get("sample_tag", "")),
                "scene_source": "radioseerselect",
            },
            "H": ground_truth,
            "S": spatial_field,
            "Phi": phi,
            "gain_norm_xy": field_xy.copy(),
            "grid_coords": grid_coords,
            "I_mask": non_building_mask_xy.copy(),
            "Gamma_quant_meta": quant_meta,
            "bounds": bounds,
            "building_mask": building_mask_xy,
            "non_building_mask": non_building_mask_xy,
            "building_heights": building_heights_xy,
            "radioseer_metadata": metadata,
            "radioseer_row": row,
        }

    def export_data(self) -> Dict:
        return self._data

    def reset_rng(self, seed: int) -> None:
        """
        Reset internal RNG used by channel/noise simulation.

        This is useful for deterministic evaluation episodes where `env.reset(seed=...)`
        is expected to make the full rollout reproducible.
        """
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

    def get_channel_info(
        self,
        uav_position: np.ndarray,
        ugv_position: np.ndarray,
        bandwidth_comm: float,
        tx_power_dbm: float,
        los: bool,
    ) -> ChannelInfo:
        height_gap = max(
            float(self.config.scene.uav_height) - float(self.config.scene.ugv_height),
            0.0,
        )
        d_horiz = np.sqrt(
            ((uav_position[0] - ugv_position[0]) * self.config.scene.grid_spacing) ** 2
            + ((uav_position[1] - ugv_position[1]) * self.config.scene.grid_spacing) ** 2
        )
        d_3d = max(np.sqrt(d_horiz ** 2 + height_gap ** 2), 1.0)

        fc = self.config.comm.carrier_freq
        fspl = 20 * np.log10(d_3d) + 20 * np.log10(fc) - 147.55
        is_los = bool(los)

        shadow_std = (
            self.config.comm.shadow_std_los_db
            if is_los
            else self.config.comm.shadow_std_nlos_db
        )
        shadow = self.rng.normal(0, shadow_std)
        nlos_extra = 0.0 if is_los else self.config.comm.nlos_excess_db

        path_loss = fspl + shadow + nlos_extra
        channel_gain = 10 ** (-path_loss / 10)
        noise_power_dbm = -174 + 10 * np.log10(bandwidth_comm + 1e-3) + self.config.comm.noise_figure_db
        snr_linear = 10 ** ((tx_power_dbm - path_loss - noise_power_dbm) / 10)
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))
        capacity = bandwidth_comm * np.log2(1 + max(snr_linear, 0))

        return ChannelInfo(
            path_loss_db=float(path_loss),
            channel_gain=float(channel_gain),
            los=bool(is_los),
            capacity_bps=float(capacity),
            snr_db=float(snr_db),
        )

    def _quantize_with_dataset_meta(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        v_min = float(self.quant_meta.get("min", np.min(arr)))
        v_max = float(self.quant_meta.get("max", np.max(arr)))
        if v_max <= v_min + 1e-12:
            return np.full_like(arr, v_min, dtype=float)
        norm = np.clip((arr - v_min) / (v_max - v_min), 0.0, 1.0)
        q_uint8 = np.round(norm * 255.0).astype(np.uint8)
        dq = (q_uint8.astype(float) / 255.0) * (v_max - v_min) + v_min
        return dq

    def _grid_index_from_position(self, position: np.ndarray) -> Tuple[int, int]:
        pos = np.asarray(position, dtype=float).reshape(2)
        gx = int(np.clip(np.round(pos[0]), 0, self.Nx - 1))
        gy = int(np.clip(np.round(pos[1]), 0, self.Ny - 1))
        return gx, gy

    def get_data_at_newpos(
        self,
        position: np.ndarray,
        add_noise: bool = False,
        quantized: bool = True,
    ) -> np.ndarray:
        """
        Get full-band spectrum data at a grid position.

        Sampling is done directly from the ground-truth tensor H at the
        corresponding grid cell instead of using pre-generated Gamma_obs.
        """
        gx, gy = self._grid_index_from_position(position)
        values = self.ground_truth[gx, gy, :].astype(float).copy()
        if add_noise:
            sigma = 0.01 * (np.mean(np.abs(values)) + 1e-9)
            values = values + self.rng.normal(0.0, sigma, size=values.shape)
        if quantized:
            values = self._quantize_with_dataset_meta(values)
        return np.asarray(values, dtype=float)

    def get_spectrum_ground_truth(
        self,
        uav_position: np.ndarray,
        freq_band_indices: np.ndarray,
    ) -> np.ndarray:
        gamma_full = self.get_data_at_newpos(
            position=uav_position,
            add_noise=True,
            quantized=False,
        )
        bands = np.asarray(freq_band_indices, dtype=int)
        return gamma_full[bands]

    def get_full_ground_truth_map(self) -> np.ndarray:
        return self.ground_truth.copy()

    def get_building_mask(self) -> np.ndarray:
        return self.building_mask.copy()

    def get_non_building_mask(self) -> np.ndarray:
        return self.non_building_mask.copy()

    def get_building_heights(self) -> np.ndarray:
        return self.building_heights.copy()


class IIBTD_opt:
    """
    II-BTD wrapper using member-only ensemble mean maps for the main radio map.
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
        self._nmse_refresh_reference: Optional[float] = None
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
            previous_models = self._tracked_models()
            obs_locs, gamma, omega = self._samples_to_arrays(self._all_samples)
            fused_obs_locs, fused_gamma, fused_omega, fusion_meta = fuse_observations_by_grid(
                obs_locs,
                gamma,
                omega,
                self.Nx,
                self.Ny,
            )
            pending_obs_locs, pending_gamma, pending_omega = self._samples_to_arrays(self._pending_samples)
            (
                fused_pending_obs_locs,
                fused_pending_gamma,
                fused_pending_omega,
                _,
            ) = fuse_observations_by_grid(
                pending_obs_locs,
                pending_gamma,
                pending_omega,
                self.Nx,
                self.Ny,
            )

            fused_effective_count = max(1, int(fusion_meta.get("fused_count", 0)))
            solver_device = (
                self.config.mappo.device
                if str(self.config.planner.iibtd_device).strip().lower() == "auto"
                else self.config.planner.iibtd_device
            )
            quality_weighted = bool(self.config.planner.ensemble_quality_weighted)
            self._reconstruct_round += 1
            refresh_interval = int(self.config.planner.ensemble_full_refresh_interval)
            full_refresh_due = bool(
                refresh_interval > 0 and (self._reconstruct_round % refresh_interval == 0)
            )
            nmse_refresh_delta = float(
                max(0.0, getattr(self.config.planner, "nmse_refresh_delta", 0.0))
            )
            nmse_reference_before = (
                float(self._nmse_refresh_reference)
                if self._nmse_refresh_reference is not None
                else float("nan")
            )
            nmse_degradation = float("nan")
            nmse_refresh_triggered = False
            pre_refresh_nmse = float("nan")

            if not self._ensemble_member_models:
                mean_map, var_map, _, ensemble_info = ensemble_reconstruct_maps(
                    obs_locs=obs_locs,
                    gamma=gamma,
                    omega=omega,
                    n_sources=self.n_sources,
                    grid_size=(self.Nx, self.Ny),
                    grid_points=self.grid_coords,
                    bounds=self.bounds,
                    i_mask=self.I_mask,
                    m_ens=self.config.planner.ensemble_size,
                    seed=self.config.mappo.seed + 31 * self._reconstruct_round + len(self._all_samples),
                    member_max_iter=select_reconstruction_outer_iters(
                        fused_effective_count,
                        warmstart=False,
                    ),
                    quality_weighted=quality_weighted,
                    mu=self.config.planner.iibtd_mu,
                    nu=self.config.planner.iibtd_nu,
                    kernel_bandwidth=self.config.planner.iibtd_kernel_bandwidth,
                    ensemble_kernel_bandwidth_mode=self.config.planner.ensemble_kernel_bandwidth_mode,
                    ensemble_kernel_bandwidth_delta=self.config.planner.ensemble_kernel_bandwidth_delta,
                    ensemble_init_jitter_scale=self.config.planner.ensemble_init_jitter_scale,
                    solver_backend=self.config.planner.iibtd_backend,
                    solver_device=solver_device,
                    gpu_phi_solver=self.config.planner.iibtd_gpu_phi_solver,
                    du_iibtd_checkpoints=self.config.planner.du_iibtd_checkpoints,
                    return_info=True,
                )
                recon_mode = "ensemble_refresh_missing_members"
            elif full_refresh_due:
                mean_map, var_map, _, ensemble_info = ensemble_reconstruct_maps(
                    obs_locs=obs_locs,
                    gamma=gamma,
                    omega=omega,
                    n_sources=self.n_sources,
                    grid_size=(self.Nx, self.Ny),
                    grid_points=self.grid_coords,
                    bounds=self.bounds,
                    i_mask=self.I_mask,
                    m_ens=self.config.planner.ensemble_size,
                    seed=self.config.mappo.seed + 31 * self._reconstruct_round + len(self._all_samples),
                    member_max_iter=select_reconstruction_outer_iters(
                        fused_effective_count,
                        warmstart=False,
                    ),
                    quality_weighted=quality_weighted,
                    mu=self.config.planner.iibtd_mu,
                    nu=self.config.planner.iibtd_nu,
                    kernel_bandwidth=self.config.planner.iibtd_kernel_bandwidth,
                    ensemble_kernel_bandwidth_mode=self.config.planner.ensemble_kernel_bandwidth_mode,
                    ensemble_kernel_bandwidth_delta=self.config.planner.ensemble_kernel_bandwidth_delta,
                    ensemble_init_jitter_scale=self.config.planner.ensemble_init_jitter_scale,
                    solver_backend=self.config.planner.iibtd_backend,
                    solver_device=solver_device,
                    gpu_phi_solver=self.config.planner.iibtd_gpu_phi_solver,
                    du_iibtd_checkpoints=self.config.planner.du_iibtd_checkpoints,
                    return_info=True,
                )
                recon_mode = "ensemble_full_refresh"
            else:
                mean_map, var_map, _, ensemble_info = incremental_refresh_ensemble_models(
                    member_models=self._ensemble_member_models,
                    member_observation_counts=self._ensemble_member_observation_counts,
                    new_obs_locs=fused_pending_obs_locs,
                    new_gamma=fused_pending_gamma,
                    new_omega=fused_pending_omega,
                    fused_obs_locs=fused_obs_locs,
                    fused_gamma=fused_gamma,
                    fused_omega=fused_omega,
                    n_sources=self.n_sources,
                    grid_size=(self.Nx, self.Ny),
                    grid_points=self.grid_coords,
                    bounds=self.bounds,
                    i_mask=self.I_mask,
                    n_outer_iter=self.config.planner.incremental_outer_iters,
                    max_svt_iter=self.config.planner.incremental_max_svt_iters,
                    quality_weighted=quality_weighted,
                    mu=self.config.planner.iibtd_mu,
                    nu=self.config.planner.iibtd_nu,
                    kernel_bandwidth=self.config.planner.iibtd_kernel_bandwidth,
                    member_kernel_bandwidths=self._latest_ensemble_info.get(
                        "member_kernel_bandwidths"
                    ),
                    solver_backend=self.config.planner.iibtd_backend,
                    solver_device=solver_device,
                    gpu_phi_solver=self.config.planner.iibtd_gpu_phi_solver,
                    fusion_meta=fusion_meta,
                )
                recon_mode = "ensemble_incremental"

            current_nmse = float(self._compute_nmse(np.asarray(mean_map, dtype=float)))
            post_refresh_nmse = float(current_nmse)
            if (
                recon_mode == "ensemble_incremental"
                and nmse_refresh_delta > 0.0
                and np.isfinite(current_nmse)
                and np.isfinite(nmse_reference_before)
            ):
                nmse_degradation = float(current_nmse - nmse_reference_before)
                if nmse_degradation >= nmse_refresh_delta:
                    pre_refresh_nmse = float(current_nmse)
                    mean_map, var_map, _, ensemble_info = ensemble_reconstruct_maps(
                        obs_locs=obs_locs,
                        gamma=gamma,
                        omega=omega,
                        n_sources=self.n_sources,
                        grid_size=(self.Nx, self.Ny),
                        grid_points=self.grid_coords,
                        bounds=self.bounds,
                        i_mask=self.I_mask,
                        m_ens=self.config.planner.ensemble_size,
                        seed=self.config.mappo.seed + 61_000 + 53 * self._reconstruct_round + len(self._all_samples),
                        member_max_iter=select_reconstruction_outer_iters(
                            fused_effective_count,
                            warmstart=False,
                        ),
                        quality_weighted=quality_weighted,
                        mu=self.config.planner.iibtd_mu,
                        nu=self.config.planner.iibtd_nu,
                        kernel_bandwidth=self.config.planner.iibtd_kernel_bandwidth,
                        solver_backend=self.config.planner.iibtd_backend,
                        solver_device=solver_device,
                        gpu_phi_solver=self.config.planner.iibtd_gpu_phi_solver,
                        du_iibtd_checkpoints=self.config.planner.du_iibtd_checkpoints,
                        return_info=True,
                    )
                    recon_mode = "ensemble_nmse_refresh"
                    nmse_refresh_triggered = True
                    current_nmse = float(self._compute_nmse(np.asarray(mean_map, dtype=float)))
                    post_refresh_nmse = float(current_nmse)

            if np.isfinite(current_nmse):
                if nmse_refresh_triggered or recon_mode in {
                    "ensemble_refresh_missing_members",
                    "ensemble_full_refresh",
                }:
                    self._nmse_refresh_reference = float(current_nmse)
                elif np.isfinite(nmse_reference_before):
                    self._nmse_refresh_reference = float(
                        min(nmse_reference_before, current_nmse)
                    )
                else:
                    self._nmse_refresh_reference = float(current_nmse)
            nmse_reference_after = (
                float(self._nmse_refresh_reference)
                if self._nmse_refresh_reference is not None
                else float("nan")
            )

            ensemble_info = dict(ensemble_info)
            ensemble_info.update(
                {
                    "recon_mode": recon_mode,
                    "full_refresh_due": bool(full_refresh_due),
                    "nmse_refresh_triggered": bool(nmse_refresh_triggered),
                    "nmse_refresh_delta": float(nmse_refresh_delta),
                    "nmse_refresh_reference_before": float(nmse_reference_before),
                    "nmse_refresh_reference_after": float(nmse_reference_after),
                    "nmse_degradation": float(nmse_degradation),
                    "pre_refresh_nmse": float(pre_refresh_nmse),
                    "post_refresh_nmse": float(post_refresh_nmse),
                }
            )
            next_member_models = list(
                ensemble_info.get("member_models", self._ensemble_member_models)
            )
            next_member_observation_counts = np.asarray(
                ensemble_info.get(
                    "member_observation_counts",
                    self._ensemble_member_observation_counts,
                ),
                dtype=int,
            ).copy()
            self._release_stale_models(
                previous_models,
                next_member_models,
            )
            if next_member_models:
                self.btd = next_member_models[0]
                self._has_fit = True
            else:
                self.btd = None
                self._has_fit = False
            self._ensemble_member_models = next_member_models
            self._ensemble_member_observation_counts = next_member_observation_counts
            self._latest_ensemble_info = dict(ensemble_info)
            self._last_fusion_meta = ensemble_info.get("fusion_meta", fusion_meta)
            self._current_map = np.asarray(mean_map, dtype=float).copy()
            self._latest_ensemble_mean_map = np.asarray(mean_map, dtype=float).copy()
            self._latest_ensemble_var_map = np.maximum(np.asarray(var_map, dtype=float), 0.0)
            self._latest_ensemble_sample_count = int(len(self._all_samples))
            self._pending_samples.clear()

        current_map = self.get_current_map()
        return RadioMapState(
            spectrum_map=current_map.copy(),
            nmse=self._compute_nmse(current_map),
            last_update_step=len(self._all_samples),
        )

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
            "ensemble_observation_mode",
            "member_observation_counts",
            "member_kernel_bandwidths",
            "full_refresh_due",
            "nmse_refresh_triggered",
            "nmse_refresh_delta",
            "nmse_refresh_reference_before",
            "nmse_refresh_reference_after",
            "nmse_degradation",
            "pre_refresh_nmse",
            "post_refresh_nmse",
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
        self._nmse_refresh_reference = None
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
        self._nmse_refresh_reference = None

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


class GridScene:
    """Scene map with building-aware motion, sampling, and LOS/NLOS geometry."""

    def __init__(
        self,
        config: Config,
        occupancy_grid: Optional[np.ndarray] = None,
        building_heights: Optional[np.ndarray] = None,
    ):
        self.Nx, self.Ny = tuple(int(v) for v in config.scene.grid_size)
        self.grid_spacing = float(config.scene.grid_spacing)
        self.uav_height = float(config.scene.uav_height)
        self.ugv_height = float(config.scene.ugv_height)
        if occupancy_grid is None:
            self.occupancy = np.zeros((self.Nx, self.Ny), dtype=bool)
        else:
            occ = np.asarray(occupancy_grid, dtype=bool)
            if occ.shape != (self.Nx, self.Ny):
                raise ValueError(
                    f"occupancy_grid shape {occ.shape} does not match scene {(self.Nx, self.Ny)}"
                )
            self.occupancy = occ.copy()
        if building_heights is None:
            self.building_heights = (
                self.occupancy.astype(float) * float(config.scene.building_height_m)
            )
        else:
            heights = np.asarray(building_heights, dtype=float)
            if heights.shape != (self.Nx, self.Ny):
                raise ValueError(
                    f"building_heights shape {heights.shape} does not match scene {(self.Nx, self.Ny)}"
                )
            self.building_heights = np.maximum(heights, 0.0)
        self._supercover_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], ...]] = {}
        self._los_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], bool] = {}
        self._cache_max_entries = 16_384

    def _grid_index(self, grid_position: np.ndarray) -> Tuple[int, int]:
        pos = np.asarray(grid_position, dtype=float).reshape(2)
        ix = int(np.round(pos[0]))
        iy = int(np.round(pos[1]))
        return ix, iy

    def _is_within_bounds(self, grid_position: np.ndarray) -> bool:
        ix, iy = self._grid_index(grid_position)
        return 0 <= ix < self.Nx and 0 <= iy < self.Ny

    def is_uav_position_valid(self, grid_position: np.ndarray) -> bool:
        return self._is_within_bounds(grid_position)

    def is_ugv_position_valid(self, grid_position: np.ndarray) -> bool:
        if not self._is_within_bounds(grid_position):
            return False
        ix, iy = self._grid_index(grid_position)
        return not bool(self.occupancy[ix, iy])

    def is_non_building_position_valid(self, grid_position: np.ndarray) -> bool:
        return self.is_ugv_position_valid(grid_position)

    def is_building_position(self, grid_position: np.ndarray) -> bool:
        if not self._is_within_bounds(grid_position):
            return False
        ix, iy = self._grid_index(grid_position)
        return bool(self.occupancy[ix, iy])

    @staticmethod
    def _supercover_line_cells(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        dx = x1 - x0
        dy = y1 - y0
        nx = abs(dx)
        ny = abs(dy)
        sign_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        sign_y = 0 if dy == 0 else (1 if dy > 0 else -1)

        x = x0
        y = y0
        ix = 0
        iy = 0
        cells = [(x, y)]
        while ix < nx or iy < ny:
            lhs = (1 + 2 * ix) * ny
            rhs = (1 + 2 * iy) * nx
            if lhs == rhs:
                x += sign_x
                y += sign_y
                ix += 1
                iy += 1
            elif lhs < rhs:
                x += sign_x
                ix += 1
            else:
                y += sign_y
                iy += 1
            cells.append((x, y))
        return cells

    def _cache_store(self, cache: Dict, key, value) -> None:
        if len(cache) >= self._cache_max_entries:
            cache.clear()
        cache[key] = value

    def _get_supercover_line_cells(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], ...]:
        key = ((int(start[0]), int(start[1])), (int(end[0]), int(end[1])))
        cached = self._supercover_cache.get(key)
        if cached is not None:
            return cached
        cells = tuple(self._supercover_line_cells(key[0], key[1]))
        self._cache_store(self._supercover_cache, key, cells)
        return cells

    def has_line_of_sight(
        self,
        uav_position: np.ndarray,
        ugv_position: np.ndarray,
    ) -> bool:
        if not self._is_within_bounds(uav_position) or not self._is_within_bounds(ugv_position):
            return False

        uav_xy = np.asarray(uav_position, dtype=float).reshape(2)
        ugv_xy = np.asarray(ugv_position, dtype=float).reshape(2)
        uav_cell = self._grid_index(uav_xy)
        ugv_cell = self._grid_index(ugv_xy)
        cache_key = ((int(uav_cell[0]), int(uav_cell[1])), (int(ugv_cell[0]), int(ugv_cell[1])))
        cached = self._los_cache.get(cache_key)
        if cached is not None:
            return bool(cached)

        horizontal_vec = ugv_xy - uav_xy
        horizontal_len_sq = float(np.dot(horizontal_vec, horizontal_vec))

        for ix, iy in self._get_supercover_line_cells(uav_cell, ugv_cell):
            if not (0 <= ix < self.Nx and 0 <= iy < self.Ny):
                self._cache_store(self._los_cache, cache_key, False)
                return False
            building_height = float(self.building_heights[ix, iy])
            if building_height <= 0.0:
                continue
            sample_xy = np.array([float(ix), float(iy)], dtype=float)
            if horizontal_len_sq <= 1e-12:
                t = 0.0
            else:
                t = float(np.dot(sample_xy - uav_xy, horizontal_vec) / horizontal_len_sq)
                t = float(np.clip(t, 0.0, 1.0))
            link_height = self.uav_height + t * (self.ugv_height - self.uav_height)
            if link_height <= building_height + 1e-6:
                self._cache_store(self._los_cache, cache_key, False)
                return False
        self._cache_store(self._los_cache, cache_key, True)
        return True

    def get_occupancy_grid(self) -> np.ndarray:
        return self.occupancy.copy()

    def get_building_heights(self) -> np.ndarray:
        return self.building_heights.copy()

    def get_scene_bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, 0.0, self.Nx * self.grid_spacing, self.Ny * self.grid_spacing)
