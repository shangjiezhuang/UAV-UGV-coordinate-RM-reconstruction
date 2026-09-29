"""
Shared quant simulation data models for active sensing.
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
from du_iibtd_based_fading_delta.shared.reconstruction import IIBTD_opt, RadioMapState
from du_iibtd_based_fading_delta.channel_loss import (
    cached_blocked_length_m, excess_loss_db, length_correction_db,
)
from du_iibtd_based_fading_delta.building_heights import build_building_height_map
from du_iibtd_based_fading_delta.uav_navigation import HeightAwareUAVNavigation

from du_iibtd_based_fading_delta.shared.quant.config import Config

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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
    large_scale_snr_db: float
    outage: bool
    shannon_capacity_bps: float
    blocked_length_m: float = 0.0
    length_correction_db: float = 0.0
    excess_loss_db: float = 0.0


class SimDataGen:
    """
    Load and serve sensing/communication data from a manifest-compatible RadioSeer dataset.
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
        self.source_spectra = np.asarray(self._data.get("Phi"), dtype=float)
        if self.source_spectra.ndim == 1:
            self.source_spectra = self.source_spectra[np.newaxis, :]
        if self.source_spectra.ndim != 2 or self.source_spectra.shape[1] != self.K:
            raise ValueError(
                "Source spectra Phi must have shape (R, K) to generate "
                f"source-dependent small-scale fading, got {self.source_spectra.shape}."
            )
        self.Nx, self.Ny = tuple(int(v) for v in self.ground_truth.shape[:2])
        self.config.scene.grid_size = (self.Nx, self.Ny)
        self.grid_coords = np.asarray(self._data["grid_coords"], dtype=float)
        self.I_mask = np.asarray(self._data["I_mask"], dtype=bool)
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
                build_building_height_map(self.building_mask, self.config.scene),
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
            root = Path(_PROJECT_ROOT) / configured
        if not (root / "manifest.csv").exists():
            raise FileNotFoundError(f"RadioSeer manifest not found under {root}.")
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
        """Decode RadioSeer gain_DPM into normalized shared-environment pixels."""
        gain_norm_xy = np.clip(np.asarray(gain_img, dtype=np.float64).T / 255.0, 0.0, 1.0)
        field_xy = np.maximum(gain_norm_xy, 1e-6)
        return field_xy

    def _build_building_height_map(self, building_mask_xy: np.ndarray) -> np.ndarray:
        return build_building_height_map(building_mask_xy, self.config.scene)

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
        bounds = ((0.0, float(width - 1)), (0.0, float(height - 1)))

        return {
            "config": {
                "dataset_root": str(dataset_root),
                "sample_index": int(sample_index),
                "sample_tag": str(row.get("sample_tag", "")),
                "scene_source": "radioseerselect",
                "pixel_resolution_m": metadata.get("pixel_resolution_m"),
                "building_height_mode": self.config.scene.building_height_mode,
                "building_height_min_m": self.config.scene.building_height_min_m,
                "building_height_max_m": self.config.scene.building_height_max_m,
                "building_height_seed": self.config.scene.building_height_seed,
            },
            "H": ground_truth,
            "S": spatial_field,
            "Phi": phi,
            "gain_norm_xy": field_xy.copy(),
            "grid_coords": grid_coords,
            "I_mask": non_building_mask_xy.copy(),
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
        blocked_length_m: float = 0.0,
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
        excess_loss = excess_loss_db(self.config.comm, is_los, blocked_length_m)

        large_scale_path_loss = fspl + excess_loss
        path_loss = large_scale_path_loss + shadow
        channel_gain = 10 ** (-path_loss / 10)
        noise_power_dbm = -174 + 10 * np.log10(bandwidth_comm + 1e-3) + self.config.comm.noise_figure_db
        large_scale_snr_db = tx_power_dbm - large_scale_path_loss - noise_power_dbm
        snr_linear = 10 ** ((tx_power_dbm - path_loss - noise_power_dbm) / 10)
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))
        shannon_capacity = bandwidth_comm * np.log2(1 + max(snr_linear, 0))
        # Transmission uses the same shadow realization for service and rate.
        # Candidate planning remains deterministic in its separate nominal model.
        outage_snr_db = (
            snr_db if self.config.comm.outage_snr_mode == "received"
            else large_scale_snr_db
        )
        outage = bool(outage_snr_db < self.config.comm.snr_outage_threshold_db)
        capacity = 0.0 if outage else shannon_capacity

        return ChannelInfo(
            path_loss_db=float(path_loss),
            channel_gain=float(channel_gain),
            los=bool(is_los),
            capacity_bps=float(capacity),
            snr_db=float(snr_db),
            large_scale_snr_db=float(large_scale_snr_db),
            outage=outage,
            shannon_capacity_bps=float(shannon_capacity),
            blocked_length_m=0.0 if is_los else float(blocked_length_m),
            length_correction_db=length_correction_db(self.config.comm, is_los, blocked_length_m),
            excess_loss_db=float(excess_loss),
        )

    def _grid_index_from_position(self, position: np.ndarray) -> Tuple[int, int]:
        pos = np.asarray(position, dtype=float).reshape(2)
        gx = int(np.clip(np.round(pos[0]), 0, self.Nx - 1))
        gy = int(np.clip(np.round(pos[1]), 0, self.Ny - 1))
        return gx, gy

    def get_data_at_newpos(
        self,
        position: np.ndarray,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        Get full-band spectrum data at a grid position.

        Sampling is done directly from the ground-truth tensor H at the
        corresponding grid cell instead of using pre-generated Gamma_obs.
        """
        gx, gy = self._grid_index_from_position(position)
        values = self.ground_truth[gx, gy, :].astype(float).copy()
        if add_noise:
            small_scale_std = float(self.config.uav.small_scale_fading_std)
            eta = self.rng.normal(
                0.0,
                small_scale_std,
                size=self.source_spectra.shape,
            )
            small_scale_fading = np.sum(self.source_spectra * eta, axis=0)

            receiver_sigma = float(self.config.uav.receiver_noise_std)
            receiver_noise = self.rng.normal(0.0, receiver_sigma, size=values.shape)
            values = values + small_scale_fading + receiver_noise
        return np.asarray(values, dtype=float)

    def get_spectrum_ground_truth(
        self,
        uav_position: np.ndarray,
        freq_band_indices: np.ndarray,
    ) -> np.ndarray:
        gamma_full = self.get_data_at_newpos(
            position=uav_position,
            add_noise=True,
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




class GridScene(HeightAwareUAVNavigation):
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
            self.building_heights = build_building_height_map(self.occupancy, config.scene)
        else:
            heights = np.asarray(building_heights, dtype=float)
            if heights.shape != (self.Nx, self.Ny):
                raise ValueError(
                    f"building_heights shape {heights.shape} does not match scene {(self.Nx, self.Ny)}"
                )
            self.building_heights = np.maximum(heights, 0.0)
        self._init_uav_navigation()
        self._supercover_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], ...]] = {}
        self._los_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], bool] = {}
        self._blocked_length_cache: Dict[tuple, float] = {}
        self._cache_max_entries = 16_384

    def _grid_index(self, grid_position: np.ndarray) -> Tuple[int, int]:
        pos = np.asarray(grid_position, dtype=float).reshape(2)
        ix = int(np.round(pos[0]))
        iy = int(np.round(pos[1]))
        return ix, iy

    def _is_within_bounds(self, grid_position: np.ndarray) -> bool:
        ix, iy = self._grid_index(grid_position)
        return 0 <= ix < self.Nx and 0 <= iy < self.Ny

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

    def get_blocked_length_m(self, uav_position: np.ndarray, ugv_position: np.ndarray) -> float:
        return cached_blocked_length_m(self, uav_position, ugv_position)

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
