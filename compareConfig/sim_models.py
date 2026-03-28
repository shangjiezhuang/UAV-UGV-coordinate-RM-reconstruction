"""
Simulation data models for MAPPO active sensing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from active_sampling import (
    adaptive_keep_ratio,
    ensemble_reconstruct_maps,
    fit_reconstruction_model,
    fuse_observations_by_grid,
    init_reconstruction_solver,
    quantize_to_8bit,
    select_reconstruction_outer_iters,
)
from config import Config

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from Test.spectrumMapTensorGen import SimConfig, generate_data


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


@dataclass
class SensorDataset:
    sensor_locs: np.ndarray
    gamma_obs: np.ndarray
    omega: np.ndarray
    grid_coords: np.ndarray
    i_mask: np.ndarray
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    quant_meta: Dict


class SimDataGen:
    """
    Generate and serve synthetic sensing/communication data.
    """

    def __init__(
        self,
        config: Config,
        seed: int = 42,
        precomputed_data: Optional[Dict] = None,
    ):
        self.config = config
        self.seed = int(seed)
        self.Nx, self.Ny = config.scene.grid_size
        self.K = config.scene.total_freq_bands_nums
        self.rng = np.random.RandomState(seed)

        if precomputed_data is None:
            self._data = self._generate_sim_data(seed)
        else:
            self._data = precomputed_data

        self.sensor_locs = np.asarray(self._data["sensor_locs"], dtype=float)
        self.ground_truth = np.asarray(self._data["H"], dtype=float)
        self.grid_coords = np.asarray(self._data["grid_coords"], dtype=float)
        self.I_mask = np.asarray(self._data["I_mask"], dtype=bool)
        self.quantized_gamma = np.asarray(self._data["Gamma_obs_quantized"], dtype=float)
        self.quant_meta = dict(self._data["Gamma_quant_meta"])
        self.omega_full = np.asarray(self._data["Omega"], dtype=np.int32)
        self.bounds = self._data["bounds"]

        if self.ground_truth.shape != (self.Nx, self.Ny, self.K):
            raise ValueError(
                f"Ground truth shape {self.ground_truth.shape} != ({self.Nx}, {self.Ny}, {self.K})"
            )
        if self.sensor_locs.shape[0] != self.quantized_gamma.shape[0]:
            raise ValueError("sensor_locs and quantized_gamma row count mismatch")

    def _generate_sim_data(self, seed: int) -> Dict:
        planner = self.config.planner
        sim_cfg = SimConfig(
            full_obs=True,
            R=1,
            K=self.K,
            M=int(planner.sensor_budget),
            N1=self.Nx,
            N2=self.Ny,
            L=max(self.Nx, self.Ny),
        )
        data = generate_data(sim_cfg, seed=seed, addShadow=False)
        gamma_q, meta = quantize_to_8bit(data["Gamma_obs"])
        return {
            "config": sim_cfg,
            "sensor_locs": data["sensor_locs"],
            "H": data["H"],
            "S": data["S"],
            "Phi": data["Phi"],
            "grid_coords": data["grid_coords"],
            "I_mask": data["I_mask"],
            "Omega": data["Omega"],
            "Gamma_obs_quantized": gamma_q,
            "Gamma_quant_meta": meta,
            "bounds": ((0.0, float(sim_cfg.L)), (0.0, float(sim_cfg.L))),
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

    def get_sensor_dataset(self) -> SensorDataset:
        return SensorDataset(
            sensor_locs=self.sensor_locs.copy(),
            gamma_obs=self.quantized_gamma.copy(),
            omega=self.omega_full.copy(),
            grid_coords=self.grid_coords.copy(),
            i_mask=self.I_mask.copy(),
            bounds=self.bounds,
            quant_meta=dict(self.quant_meta),
        )

    def get_channel_info(
        self,
        uav_position: np.ndarray,
        ugv_position: np.ndarray,
        bandwidth_comm: float,
        tx_power_dbm: float,
    ) -> ChannelInfo:
        d_horiz = np.sqrt(
            ((uav_position[0] - ugv_position[0]) * self.config.scene.grid_spacing) ** 2
            + ((uav_position[1] - ugv_position[1]) * self.config.scene.grid_spacing) ** 2
        )
        d_3d = max(np.sqrt(d_horiz ** 2 + self.config.scene.uav_height ** 2), 1.0)

        fc = self.config.comm.carrier_freq
        fspl = 20 * np.log10(d_3d) + 20 * np.log10(fc) - 147.55
        theta_deg = np.degrees(
            np.arctan2(self.config.scene.uav_height, max(d_horiz, 1e-9))
        )
        a = float(self.config.comm.los_model_a)
        b = float(self.config.comm.los_model_b)
        p_los = 1.0 / (1.0 + a * np.exp(-b * (theta_deg - a)))
        p_los = float(np.clip(p_los, 0.0, 1.0))
        is_los = self.rng.random() < p_los

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


class IIBTD_opt:
    """
    II-BTD wrapper using history-refit ensemble mean maps for the main radio map.
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
        self._latest_ensemble_sample_count: int = 0
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

    def reconstruct(self) -> RadioMapState:
        if self._pending_samples:
            obs_locs, gamma, omega = self._samples_to_arrays(self._all_samples)
            keep_ratio = adaptive_keep_ratio(
                int(obs_locs.shape[0]),
                early_ratio=self.config.planner.ensemble_keep_ratio,
                late_ratio=max(0.05, float(self.config.planner.ensemble_keep_ratio) - 0.2),
                switch_M=30,
            )
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
                keep_ratio=keep_ratio,
                keep_recent=self.config.planner.ensemble_keep_recent,
                seed=self.config.mappo.seed + len(self._all_samples),
                base_model=self.btd if self._has_fit else None,
                quality_weighted=True,
                solver_backend=self.config.planner.iibtd_backend,
                solver_device=(
                    self.config.mappo.device
                    if str(self.config.planner.iibtd_device).strip().lower() == "auto"
                    else self.config.planner.iibtd_device
                ),
                gpu_phi_solver=self.config.planner.iibtd_gpu_phi_solver,
                return_info=True,
            )
            representative_model = ensemble_info.get("representative_model")
            if representative_model is not None:
                self.btd = representative_model
                self._has_fit = True
            elif ensemble_info.get("base_model") is not None:
                self.btd = ensemble_info["base_model"]
                self._has_fit = True
            fusion_meta = ensemble_info.get("fusion_meta", self._last_fusion_meta)
            self._last_fusion_meta = fusion_meta
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
        return np.asarray(self.btd.H_hat, dtype=float).copy()

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

    def get_effective_sample_count(self) -> int:
        return int(self._last_fusion_meta.get("fused_count", 0))

    def get_compression_ratio(self) -> float:
        return float(self._last_fusion_meta.get("compression_ratio", 1.0))

    def reset(self) -> None:
        self._pending_samples.clear()
        self._all_samples.clear()
        self._has_fit = False
        self._current_map = None
        self._latest_ensemble_mean_map = None
        self._latest_ensemble_var_map = None
        self._latest_ensemble_sample_count = 0
        self._last_fusion_meta = {
            "raw_count": 0,
            "fused_count": 0,
            "compression_ratio": 1.0,
            "raw_per_fused": np.empty((0,), dtype=int),
        }
        self.btd = self._make_empty_solver()

    def get_num_samples(self) -> int:
        return len(self._all_samples)

    def set_ground_truth(self, gt: np.ndarray) -> None:
        self._ground_truth = np.asarray(gt, dtype=float)

    def _compute_nmse(self, est_map: Optional[np.ndarray] = None) -> float:
        if self._ground_truth is None:
            return 1.0
        est = self.btd.H_hat if est_map is None else np.asarray(est_map, dtype=float)
        gt = self._ground_truth
        return float(np.sum((est - gt) ** 2) / (np.sum(gt ** 2) + 1e-10))

    def _make_empty_solver(self):
        solver_device = str(self.config.planner.iibtd_device).strip()
        if not solver_device or solver_device.lower() == "auto":
            solver_device = str(self.config.mappo.device).strip() or "cpu"
        return init_reconstruction_solver(
            grid_points=self.grid_coords,
            bounds=self.bounds,
            K=self.K,
            i_mask=self.I_mask,
            n_sources=self.n_sources,
            grid_size=(self.Nx, self.Ny),
            max_iter=6,
            kernel_bandwidth=0.46,
            warmstart=False,
            solver_backend=self.config.planner.iibtd_backend,
            solver_device=solver_device,
            gpu_phi_solver=self.config.planner.iibtd_gpu_phi_solver,
        )

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
    """Simple scene map: both agents can move inside grid bounds."""

    def __init__(self, config: Config):
        self.Nx, self.Ny = config.scene.grid_size
        self.grid_spacing = config.scene.grid_spacing

    def is_uav_position_valid(self, grid_position: np.ndarray) -> bool:
        ix, iy = int(np.round(grid_position[0])), int(np.round(grid_position[1]))
        return 0 <= ix < self.Nx and 0 <= iy < self.Ny

    def is_ugv_position_valid(self, grid_position: np.ndarray) -> bool:
        ix, iy = int(np.round(grid_position[0])), int(np.round(grid_position[1]))
        return 0 <= ix < self.Nx and 0 <= iy < self.Ny

    def get_occupancy_grid(self) -> np.ndarray:
        return np.zeros((self.Nx, self.Ny), dtype=bool)

    def get_building_heights(self) -> np.ndarray:
        return np.zeros((self.Nx, self.Ny), dtype=float)

    def get_scene_bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, 0.0, self.Nx * self.grid_spacing, self.Ny * self.grid_spacing)
