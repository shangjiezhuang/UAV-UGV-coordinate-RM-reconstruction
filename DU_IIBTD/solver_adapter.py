from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .DU_IIBTD import DU_IIBTD, append_observation


class DU_IIBTDSolverAdapter:
    """Expose a trained DU-IIBTD checkpoint through the legacy IIBTD solver API.

    The UAV ensemble code was written around the iterative IIBTD classes, whose
    public contract is `init_sequential`, `fit_2`, `fit_incremental`,
    `get_current_map`, and numpy attributes `Theta/Phi/Sr/H_hat`.  DU-IIBTD is a
    torch module instead, so this adapter keeps the high-level UAV loop unchanged
    while replacing each ensemble member with one trained checkpoint.
    """

    def __init__(
        self,
        n_sources=1,
        grid_size=(128, 128),
        mu=1.0,
        nu=1.0,
        max_iter=1,
        tol=1e-5,
        kernel_bandwidth=0.3,
        warmstart=False,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        dtype=torch.float32,
        min_sensors_for_update: int = 6,
        update_batch_size: Optional[int] = None,
    ):
        if int(n_sources) != 1:
            raise ValueError("DU-IIBTD adapter currently supports only R=1.")
        if checkpoint_path is None:
            raise ValueError("DU-IIBTD requires a trained checkpoint path.")

        self.R = 1
        self.N1, self.N2 = int(grid_size[0]), int(grid_size[1])
        self.mu = float(mu)
        self.nu = float(nu)
        self.max_iter = int(max(1, max_iter))
        self.tol = float(tol)

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DU-IIBTD checkpoint does not exist: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self._checkpoint_config = dict(checkpoint.get("config", {}))
        self._state_dict = checkpoint.get("model_state_dict", checkpoint)

        if kernel_bandwidth is None or float(kernel_bandwidth) <= 0.0:
            kernel_bandwidth = self._checkpoint_config_value("kernel_bandwidth", 0.3)
        self.h = float(kernel_bandwidth)
        self.warmstart = bool(warmstart)
        self.dim_poly = 6
        if min_sensors_for_update is None or int(min_sensors_for_update) <= 0:
            min_sensors_for_update = self._checkpoint_config_value("min_sensors_for_update", 6)
        self.min_sensors_for_update = int(max(1, min_sensors_for_update))
        if update_batch_size is None or int(update_batch_size) <= 0:
            update_batch_size = self._checkpoint_config_value("update_batch_size", None)
        self.update_batch_size = (
            None if update_batch_size is None or int(update_batch_size) <= 0 else int(update_batch_size)
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device was requested for DU-IIBTD but is unavailable: {self.device}")
        self.dtype = dtype

        self._model: DU_IIBTD | None = None
        self._state: dict[str, torch.Tensor] | None = None
        self._obs: dict | None = None
        self._grid: dict | None = None
        self._bounds = None
        self._I_flat: torch.Tensor | None = None
        self._initialized = False

        self.Theta = None
        self.Phi = None
        self.Sr = None
        self.H_hat = None

    def _checkpoint_config_value(self, key: str, default=None):
        config = getattr(self, "_checkpoint_config", None)
        if not isinstance(config, dict):
            return default
        value = config.get(key, default)
        return default if value is None else value

    def _ensure_model(self, K: int) -> DU_IIBTD:
        """Instantiate the checkpointed network once K is known."""
        K = int(K)
        ckpt_n1 = self._checkpoint_config.get("N1")
        ckpt_n2 = self._checkpoint_config.get("N2")
        if ckpt_n1 is not None and ckpt_n2 is not None:
            if int(ckpt_n1) != self.N1 or int(ckpt_n2) != self.N2:
                raise ValueError(
                    "DU-IIBTD checkpoint grid size does not match the UAV run: "
                    f"checkpoint grid=({int(ckpt_n1)}, {int(ckpt_n2)}), "
                    f"run grid=({self.N1}, {self.N2})."
                )
        ckpt_K = self._checkpoint_config.get("K")
        if ckpt_K is not None and int(ckpt_K) != K:
            raise ValueError(
                "DU-IIBTD checkpoint frequency dimension does not match the UAV run: "
                f"checkpoint K={int(ckpt_K)}, run K={K}. Use --K {int(ckpt_K)} "
                "or a checkpoint trained for the requested K."
            )

        if self._model is None:
            model = DU_IIBTD(
                M=self.N1,
                N=self.N2,
                K=K,
                T=int(self._checkpoint_config.get("T", 2)),
                nu=float(self._checkpoint_config.get("nu", self.nu)),
                hidden=int(self._checkpoint_config.get("hidden", 32)),
                local_sr_update=bool(self._checkpoint_config.get("local_sr_update", False)),
            ).to(device=self.device, dtype=self.dtype)
            model.load_state_dict(self._state_dict, strict=True)
            model.eval()
            self._model = model
        return self._model

    def _make_normalizer(self, bounds):
        (min_x, max_x), (min_y, max_y) = bounds
        self._scale_x = max((float(max_x) - float(min_x)) / 2.0, 1e-12)
        self._scale_y = max((float(max_y) - float(min_y)) / 2.0, 1e-12)
        self._center_x = (float(max_x) + float(min_x)) / 2.0
        self._center_y = (float(max_y) + float(min_y)) / 2.0

    def _normalize_coords(self, coords) -> torch.Tensor:
        arr = torch.as_tensor(coords, dtype=self.dtype, device=self.device).reshape(-1, 2)
        out = torch.empty_like(arr)
        out[:, 0] = (arr[:, 0] - self._center_x) / self._scale_x
        out[:, 1] = (arr[:, 1] - self._center_y) / self._scale_y
        return out

    def _grid_indices_from_locs(self, locs) -> torch.Tensor:
        locs_np = np.asarray(locs, dtype=float).reshape(-1, 2)
        grid_xy = np.floor(locs_np).astype(np.int64)
        grid_xy[:, 0] = np.clip(grid_xy[:, 0], 0, self.N1 - 1)
        grid_xy[:, 1] = np.clip(grid_xy[:, 1], 0, self.N2 - 1)
        return torch.as_tensor(grid_xy[:, 0] * self.N2 + grid_xy[:, 1], dtype=torch.long, device=self.device)

    def _sync_public_state(self):
        if self._state is None:
            return
        self.Theta = self._state["Theta"].detach().cpu().numpy()
        self.Phi = self._state["Phi"].detach().cpu().numpy()
        self.Sr = self._state["Sr"].detach().cpu().numpy()
        self.H_hat = self._state["H_hat"].detach().cpu().numpy()

    def init_sequential(self, grid_coords, bounds, K, I_mask=None):
        """Initialize an empty sequential DU-IIBTD state for one ensemble member."""
        model = self._ensure_model(int(K))
        self._bounds = bounds
        self._make_normalizer(bounds)

        grid_norm = self._normalize_coords(grid_coords)
        n_grid = self.N1 * self.N2
        if grid_norm.shape[0] != n_grid:
            raise ValueError(f"grid_coords must have {n_grid} rows, got {grid_norm.shape[0]}.")

        if I_mask is None:
            I_mask = np.ones((self.N1, self.N2), dtype=bool)
        self._I_flat = torch.as_tensor(
            np.asarray(I_mask, dtype=bool).reshape(-1),
            dtype=torch.bool,
            device=self.device,
        )
        if self._I_flat.numel() != n_grid:
            raise ValueError(f"I_mask must contain {n_grid} values, got {self._I_flat.numel()}.")

        self._grid = {"grid_norm": grid_norm}
        self._state = model.init_state(device=self.device, dtype=self.dtype)
        self._obs = None
        self._initialized = True
        self._sync_public_state()
        return self

    def _consume_observations(self, sensor_locs, gamma, omega, *, passes_per_update=1):
        if not self._initialized or self._state is None or self._grid is None or self._I_flat is None:
            raise RuntimeError("DU-IIBTD adapter must be initialized before observations are consumed.")

        sensor_locs = np.atleast_2d(np.asarray(sensor_locs, dtype=float))
        gamma = np.atleast_2d(np.asarray(gamma, dtype=np.float32))
        omega = np.atleast_2d(np.asarray(omega, dtype=np.float32))
        if sensor_locs.shape[0] == 0:
            return self
        if gamma.shape != omega.shape:
            raise ValueError(f"gamma and omega must have the same shape, got {gamma.shape} vs {omega.shape}.")
        if gamma.shape[0] != sensor_locs.shape[0]:
            raise ValueError("sensor_locs row count must match gamma/omega row count.")

        batch_size = self.update_batch_size or int(sensor_locs.shape[0])
        batch_size = int(max(1, batch_size))
        passes_per_update = int(max(1, passes_per_update))

        # DU-IIBTD is already an unrolled optimizer: one forward applies all T
        # learned layers.  The legacy `max_iter` knobs are therefore interpreted
        # as repeated DU forwards only for incremental refreshes, not as BCD
        # iterations inside each layer.
        with torch.no_grad():
            for start in range(0, int(sensor_locs.shape[0]), batch_size):
                end = min(start + batch_size, int(sensor_locs.shape[0]))
                locs_norm = self._normalize_coords(sensor_locs[start:end])
                gamma_t = torch.as_tensor(gamma[start:end], dtype=self.dtype, device=self.device)
                omega_t = torch.as_tensor(omega[start:end], dtype=self.dtype, device=self.device)
                sample_idx = self._grid_indices_from_locs(sensor_locs[start:end])

                self._obs = append_observation(
                    self._obs,
                    locs_norm,
                    gamma_t,
                    omega_t,
                    self._grid["grid_norm"],
                    (self.N1, self.N2),
                    self.h,
                    I_flat=self._I_flat,
                    sample_grid_idx=sample_idx,
                )
                if int(self._obs["locs_norm"].shape[0]) < self.min_sensors_for_update:
                    continue
                for _ in range(passes_per_update):
                    self._state = self._model(self._state, self._obs, self._grid)

        self._sync_public_state()
        return self

    def fit_2(self, sensor_locs, Gamma, Omega, grid_coords, bounds, I_mask=None, debugFlag=False):
        """Cold-start reconstruction over the provided fused observations."""
        del debugFlag
        K = np.atleast_2d(Gamma).shape[1]
        self.init_sequential(grid_coords, bounds, K=K, I_mask=I_mask)
        return self._consume_observations(sensor_locs, Gamma, Omega, passes_per_update=1)

    def fit_incremental(
        self,
        new_sensor_locs,
        new_gamma,
        new_omega,
        grid_coords=None,
        bounds=None,
        I_mask=None,
        n_outer_iter=2,
        max_svt_iter=20,
        debugFlag=False,
    ):
        """Append UAV measurements and run the trained unrolled update."""
        del max_svt_iter, debugFlag
        if not self._initialized:
            if grid_coords is None or bounds is None:
                raise ValueError("First DU-IIBTD incremental call requires grid_coords and bounds.")
            K = np.atleast_2d(new_gamma).shape[1]
            self.init_sequential(grid_coords, bounds, K=K, I_mask=I_mask)
        elif I_mask is not None:
            self._I_flat = torch.as_tensor(
                np.asarray(I_mask, dtype=bool).reshape(-1),
                dtype=torch.bool,
                device=self.device,
            )
        return self._consume_observations(
            new_sensor_locs,
            new_gamma,
            new_omega,
            passes_per_update=max(1, int(n_outer_iter)),
        )

    def load_state_from(self, src_model):
        """Warm-start from another adapter when the ensemble code requests it."""
        if not isinstance(src_model, DU_IIBTDSolverAdapter):
            for name in ("Theta", "Phi", "Sr", "H_hat"):
                if hasattr(src_model, name):
                    setattr(self, name, np.array(getattr(src_model, name), copy=True))
            return self

        self._ensure_model(src_model.Phi.shape[1])
        self._bounds = src_model._bounds
        for name in ("_scale_x", "_scale_y", "_center_x", "_center_y"):
            if hasattr(src_model, name):
                setattr(self, name, getattr(src_model, name))
        self._grid = (
            None
            if src_model._grid is None
            else {key: value.detach().clone() for key, value in src_model._grid.items()}
        )
        self._I_flat = None if src_model._I_flat is None else src_model._I_flat.detach().clone()
        self._state = (
            None
            if src_model._state is None
            else {key: value.detach().clone() for key, value in src_model._state.items()}
        )
        self._obs = (
            None
            if src_model._obs is None
            else {
                key: (value.detach().clone() if torch.is_tensor(value) else value)
                for key, value in src_model._obs.items()
            }
        )
        self._initialized = bool(src_model._initialized)
        self._sync_public_state()
        return self

    def jitter_state(self, scale: float = 0.0):
        """Optional tiny perturbation used only if the caller enables it."""
        if self._state is None or float(scale) <= 0.0:
            return self
        with torch.no_grad():
            for key in ("Theta", "Phi", "Sr", "H_hat"):
                value = self._state.get(key)
                if value is None or not torch.is_floating_point(value):
                    continue
                jittered = value * (1.0 + torch.randn_like(value) * float(scale))
                if bool(torch.all(value >= 0.0).item()):
                    jittered = torch.clamp(jittered, min=1e-12)
                self._state[key] = jittered
        self._sync_public_state()
        return self

    def get_current_map(self):
        self._sync_public_state()
        if self.H_hat is None:
            raise RuntimeError("DU-IIBTD state is not initialized.")
        return self.H_hat.copy()

    def get_source_maps(self):
        self._sync_public_state()
        if self.Sr is None:
            raise RuntimeError("DU-IIBTD state is not initialized.")
        return self.Sr.copy()

    def get_spectra(self):
        self._sync_public_state()
        if self.Phi is None:
            raise RuntimeError("DU-IIBTD state is not initialized.")
        return self.Phi.copy()

    def close(self):
        self._state = None
        self._obs = None
        self._grid = None
        self._I_flat = None
        self._model = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
