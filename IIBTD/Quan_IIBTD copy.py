import numpy as np
import torch


def make_bin_boundaries_from_map(
    map_data,
    bits=3,
    log_offset=1e-6,
    device=None,
    dtype=torch.float64,
    extend_edges=False,
    min_gap=1e-6,
):
    """
    Build log-domain quantization boundaries from radio-map values.

    Quan_IIBTD models quantized observations of log(gamma + log_offset), so
    these boundaries should be estimated from the same transformed map values.
    """
    bits = int(bits)
    if bits < 1:
        raise ValueError("bits must be a positive integer")

    def _flatten_log_values(values):
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
        tensor = torch.clamp(tensor, min=0.0)
        log_values = torch.log(tensor.reshape(-1) + float(log_offset))
        return log_values[torch.isfinite(log_values)]

    if isinstance(map_data, (list, tuple)):
        value_parts = [_flatten_log_values(item) for item in map_data]
        value_parts = [part for part in value_parts if part.numel() > 0]
        if not value_parts:
            raise ValueError("map_data does not contain any finite values")
        log_values = torch.cat(value_parts)
    else:
        log_values = _flatten_log_values(map_data)

    if log_values.numel() == 0:
        raise ValueError("map_data does not contain any finite values")

    n_edges = 2 ** bits + 1
    probs = torch.linspace(0.0, 1.0, n_edges, dtype=dtype, device=log_values.device)
    if extend_edges:
        inner = torch.quantile(log_values, probs[1:-1])
        boundaries = torch.empty(n_edges, dtype=dtype, device=log_values.device)
        boundaries[0] = -torch.inf
        boundaries[-1] = torch.inf
        boundaries[1:-1] = inner
    else:
        boundaries = torch.quantile(log_values, probs)

    if min_gap is not None and float(min_gap) > 0:
        gap = torch.as_tensor(float(min_gap), dtype=dtype, device=boundaries.device)
        last_finite = None
        for idx in range(boundaries.numel()):
            if not torch.isfinite(boundaries[idx]):
                continue
            if last_finite is not None and boundaries[idx] <= last_finite + gap:
                boundaries[idx] = last_finite + gap
            last_finite = boundaries[idx].clone()

    return boundaries


class Quan_IIBTD:
    """
    Quantized II-BTD solver with a probit/ordinal MLE data term.

    Design choices
    --------------
    - Keep the original II-BTD block structure:
        1. local Theta update
        2. global Phi update
        3. low-rank Sr update via SVT
    - Replace the squared-error data fidelity with a quantized MLE term.

    Observation model
    -----------------
    Given a continuous nonnegative prediction gamma_hat, define

        u = log(gamma_hat + eta)

    and quantized observations

        Y = Q(u + epsilon), epsilon ~ N(0, sigma_q^2)

    The bin likelihood is

        P(Y=q | u) = Phi((b_q - u) / sigma_q) - Phi((b_{q-1} - u) / sigma_q)

    where Phi is the standard Gaussian CDF and the bin edges are stored in
    self.bin_boundaries.
    """

    make_bin_boundaries_from_map = staticmethod(make_bin_boundaries_from_map)

    def __init__(
        self,
        n_sources=2,
        grid_size=(20, 20),
        mu=1.0,
        nu=1.0,
        max_iter=20,
        tol=1e-5,
        kernel_bandwidth=1.0,
        warmstart=False,
        device=None,
        dtype=torch.float64,
        bin_boundaries=None,
        sigma_q=1,
        log_offset=1e-6,
        pred_epsilon=1e-9,
        theta_lr=0.03,
        phi_lr=0.03,
        theta_inner_iter=15,
        phi_inner_iter=25,
        inner_tol=1e-4,
        lambda_phi=1e-4,
        grid_batch_size=128,
        max_block_cache_mb=256.0,
    ):
        self.R = int(n_sources)
        self.N1, self.N2 = int(grid_size[0]), int(grid_size[1])
        self.mu = float(mu)
        self.nu = float(nu)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.h = float(kernel_bandwidth)
        self.warmstart = bool(warmstart)
        self.dim_poly = 6

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        if bin_boundaries is None:
            raise ValueError(
                "bin_boundaries must be provided. Use "
                "Quan_IIBTD.make_bin_boundaries_from_map(map_data, bits=...) "
                "to build data-matched boundaries."
            )
        self.bin_boundaries = self._to_tensor(bin_boundaries)
        if self.bin_boundaries.ndim != 1 or self.bin_boundaries.numel() < 2:
            raise ValueError("bin_boundaries must be a 1D array with at least two entries")

        self.sigma_q = float(sigma_q)
        self.log_offset = float(log_offset)
        self.pred_epsilon = float(pred_epsilon)
        self.theta_lr = float(theta_lr)
        self.phi_lr = float(phi_lr)
        self.theta_inner_iter = int(max(1, theta_inner_iter))
        self.phi_inner_iter = int(max(1, phi_inner_iter))
        self.inner_tol = float(max(0.0, inner_tol))
        self.lambda_phi = float(max(0.0, lambda_phi))
        self.grid_batch_size = int(max(1, grid_batch_size))
        self.max_block_cache_bytes = int(max(0.0, float(max_block_cache_mb)) * 1024 * 1024)

        self._initialized = False
        self._bounds = None

        self.Theta = None
        self.Phi = None
        self.Sr = None
        self.H_hat = None
        self.U_hat = None

        self._Theta_t = None
        self._Phi_t = None
        self._Sr_t = None
        self._H_hat_t = None
        self._U_hat_t = None

        self._sensor_locs_t = None
        self._Y_t = None
        self._Omega_t = None
        self._grid_coords_t = None
        self._locs_norm_t = None
        self._grid_norm_t = None
        self._Weights_raw_t = None
        self._I_mask_bool_t = None
        self._I_mask_t = None
        self._sensor_locs = None
        self._Y = None
        self._Omega = None
        self._grid_coords = None
        self._locs_norm = None
        self._grid_norm = None
        self._Weights_raw = None
        self.I_mask = None

    # =================================================================
    # Tensor helpers
    # =================================================================

    def _to_tensor(self, arr, dtype=None, device=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=dtype)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    def _to_long_tensor(self, arr, device=None):
        if device is None:
            device = self.device
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=torch.long)
        return torch.as_tensor(arr, dtype=torch.long, device=device)

    def _to_bool_tensor(self, arr, device=None):
        if device is None:
            device = self.device
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=torch.bool)
        return torch.as_tensor(arr, dtype=torch.bool, device=device)

    @staticmethod
    def _atleast_2d(arr):
        if isinstance(arr, torch.Tensor):
            if arr.ndim == 1:
                return arr.unsqueeze(0)
            return arr
        return np.atleast_2d(arr)

    @staticmethod
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _require_positive_sigma_q(self):
        if self.sigma_q <= 0:
            raise ValueError(
                "sigma_q must be positive for the quant_BTD-compatible probit MLE. "
                "Use sigma_q > 0 for fitting."
            )

    def _validate_reconstruction_inputs(self, sensor_locs_t, Y_t, Omega_t, grid_coords_t):
        if sensor_locs_t.ndim != 2 or sensor_locs_t.shape[1] != 2:
            raise ValueError("sensor_locs must have shape (M, 2)")
        if Y_t.ndim != 2 or Omega_t.ndim != 2:
            raise ValueError("Y and Omega must both be 2D arrays with shape (M, K)")
        if Y_t.shape != Omega_t.shape:
            raise ValueError("Y and Omega must have the same shape")
        if sensor_locs_t.shape[0] != Y_t.shape[0]:
            raise ValueError("sensor_locs, Y, and Omega must have the same number of rows")
        if grid_coords_t.ndim != 2 or grid_coords_t.shape[1] != 2:
            raise ValueError("grid_coords must have shape (N_grid, 2)")
        if grid_coords_t.shape[0] != self.N1 * self.N2:
            raise ValueError(
                f"grid_coords must have exactly {self.N1 * self.N2} rows for grid_size={self.N1, self.N2}"
            )

    def _prepare_observation_tensors(self, Y, Omega):
        omega_t = self._to_tensor(self._atleast_2d(Omega))
        if Y is None:
            raise ValueError("Y must be provided as integer quantization-bin indices")
        y_t = self._to_long_tensor(self._atleast_2d(Y))
        if y_t.shape != omega_t.shape:
            raise ValueError("Y and Omega must have the same shape")
        return y_t, omega_t

    def _sync_public_state(self):
        self.Theta = None if self._Theta_t is None else self._to_numpy(self._Theta_t)
        self.Phi = None if self._Phi_t is None else self._to_numpy(self._Phi_t)
        self.Sr = None if self._Sr_t is None else self._to_numpy(self._Sr_t)
        self.H_hat = None if self._H_hat_t is None else self._to_numpy(self._H_hat_t)
        self.U_hat = None if self._U_hat_t is None else self._to_numpy(self._U_hat_t)

    def _sync_public_cache(self):
        self._sensor_locs = None if self._sensor_locs_t is None else self._to_numpy(self._sensor_locs_t)
        self._Y = None if self._Y_t is None else self._to_numpy(self._Y_t)
        self._Omega = None if self._Omega_t is None else self._to_numpy(self._Omega_t)
        self._grid_coords = None if self._grid_coords_t is None else self._to_numpy(self._grid_coords_t)
        self._locs_norm = None if self._locs_norm_t is None else self._to_numpy(self._locs_norm_t)
        self._grid_norm = None if self._grid_norm_t is None else self._to_numpy(self._grid_norm_t)
        self._Weights_raw = None if self._Weights_raw_t is None else self._to_numpy(self._Weights_raw_t)
        self.I_mask = None if self._I_mask_bool_t is None else self._to_numpy(self._I_mask_bool_t)

    def _set_i_mask(self, I_mask):
        if I_mask is None:
            I_mask = np.ones((self.N1, self.N2), dtype=bool)
        self._I_mask_bool_t = self._to_bool_tensor(I_mask)
        self._I_mask_t = self._I_mask_bool_t.to(dtype=self.dtype)
        self.I_mask = self._to_numpy(self._I_mask_bool_t)

    @staticmethod
    def _debug_range(tensor):
        if tensor is None or tensor.numel() == 0:
            return "empty"
        return f"[{float(torch.min(tensor)):.4g}, {float(torch.max(tensor)):.4g}]"

    def _debug_boundary_preview(self):
        values = self.bin_boundaries.detach().cpu().tolist()
        if len(values) <= 12:
            shown = values
        else:
            shown = values[:4] + ["..."] + values[-4:]
        return "[" + ", ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in shown) + "]"

    def _print_debug_fit_header(self, Y_t, Omega_t, N_grid, M, K, I_flat):
        num_bins = int(self.bin_boundaries.numel() - 1)
        obs_mask = Omega_t > 0
        obs_count = int(torch.sum(obs_mask).item())
        total_count = int(Omega_t.numel())
        obs_ratio = obs_count / max(1, total_count)
        print(
            f"Starting Quantized II-BTD: N_grid={N_grid}, M={M}, K={K}, R={self.R}, "
            f"device={self.device}, dtype={self.dtype}"
        )
        print(
            f"  Hyperparams: max_iter={self.max_iter}, tol={self.tol}, h={self.h}, "
            f"mu={self.mu}, nu={self.nu}, sigma_q={self.sigma_q}"
        )
        print(
            f"  Optimizer: theta_lr={self.theta_lr}, phi_lr={self.phi_lr}, "
            f"theta_inner_iter={self.theta_inner_iter}, phi_inner_iter={self.phi_inner_iter}, "
            f"inner_tol={self.inner_tol}, lambda_phi={self.lambda_phi}"
        )
        print(f"  Bin boundaries ({num_bins} bins): {self._debug_boundary_preview()}")
        print(
            f"  Observations: {obs_count}/{total_count} observed "
            f"({100.0 * obs_ratio:.2f}%), Omega range={self._debug_range(Omega_t)}"
        )
        if obs_count > 0:
            y_obs = Y_t[obs_mask]
            invalid = int(torch.sum((y_obs < 0) | (y_obs >= num_bins)).item())
            y_clamped = torch.clamp(y_obs, min=0, max=max(0, num_bins - 1))
            counts = torch.bincount(y_clamped.reshape(-1), minlength=num_bins)
            hist = ", ".join(
                f"{idx}:{int(count)}"
                for idx, count in enumerate(counts.detach().cpu().tolist())
                if int(count) > 0
            )
            if not hist:
                hist = "empty"
            print(
                f"  Y observed range={self._debug_range(y_obs.to(dtype=self.dtype))}, "
                f"invalid_bins={invalid}, bin_hist={{ {hist} }}"
            )
        print(f"  |I| = {int(torch.sum(I_flat))}/{N_grid}, warmstart={self.warmstart}")

    def load_state_from(self, src_model):
        for name in ("Theta", "Phi", "Sr"):
            value = getattr(src_model, name, None)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.dtype == object:
                raise TypeError(f"Cannot warm-start from non-numeric {name} state")
            arr = np.array(arr, copy=True)
            setattr(self, name, arr)
            tensor_attr = f"_{name}_t"
            if hasattr(self, tensor_attr):
                setattr(self, tensor_attr, self._to_tensor(arr))
        self._initialized = True
        return self

    # =================================================================
    # Kernel, features, normalization
    # =================================================================

    def _epanechnikov_kernel(self, dists):
        u = dists / self.h
        k = 0.75 * (1 - u ** 2)
        return torch.where(torch.abs(u) <= 1.0, k, torch.zeros_like(k))

    def _poly_features_from_diff(self, diff):
        dx = diff[..., 0]
        dy = diff[..., 1]
        ones = torch.ones_like(dx)
        return torch.stack((ones, dx, dy, dx ** 2, dx * dy, dy ** 2), dim=-1)

    def _get_poly_features(self, coords_norm, center_norm):
        diff = coords_norm - center_norm
        return self._poly_features_from_diff(diff)

    def _make_normalizer(self, bounds):
        (min_x, max_x), (min_y, max_y) = bounds
        self._scale_x = (max_x - min_x) / 2.0
        self._scale_y = (max_y - min_y) / 2.0
        self._center_x = (max_x + min_x) / 2.0
        self._center_y = (max_y + min_y) / 2.0

    def _normalize(self, coords):
        c = self._to_tensor(coords)
        n = c.clone()
        n[:, 0] = (c[:, 0] - self._center_x) / self._scale_x
        n[:, 1] = (c[:, 1] - self._center_y) / self._scale_y
        return n

    # =================================================================
    # Quantization model
    # =================================================================

    def _quant_boundaries(self):
        boundaries = self.bin_boundaries.clone()
        boundaries[-1] = torch.tensor(float("inf"), dtype=self.dtype, device=self.device)
        return boundaries

    def _gaussian_cdf(self, value):
        self._require_positive_sigma_q()
        denom = self.sigma_q * np.sqrt(2.0)
        return 0.5 * (1.0 + torch.erf(value / denom))

    def probit_bin_prob(self, y_idx, u_hat):
        boundaries = self._quant_boundaries()
        y_idx = torch.clamp(y_idx.long(), min=0, max=boundaries.numel() - 2)
        lower = boundaries[y_idx]
        upper = boundaries[y_idx + 1]
        prob = self._gaussian_cdf(upper - u_hat) - self._gaussian_cdf(lower - u_hat)
        return torch.clamp(prob, min=1e-12)

    def _bin_log_representatives(self):
        boundaries = self._quant_boundaries()
        finite = boundaries[torch.isfinite(boundaries)]
        if finite.numel() >= 2:
            widths = torch.diff(finite)
            widths = widths[widths > 0]
            step = torch.median(widths) if widths.numel() > 0 else torch.ones((), dtype=self.dtype, device=self.device)
        else:
            step = torch.ones((), dtype=self.dtype, device=self.device)

        reps = []
        for q in range(boundaries.numel() - 1):
            lower = boundaries[q]
            upper = boundaries[q + 1]
            lower_val = float(lower.detach().cpu())
            upper_val = float(upper.detach().cpu())
            if np.isneginf(lower_val) and np.isposinf(upper_val):
                rep = torch.zeros((), dtype=self.dtype, device=self.device)
            elif np.isneginf(lower_val):
                rep = upper - 0.5 * step
            elif np.isposinf(upper_val):
                rep = lower + 0.5 * step
            else:
                rep = 0.5 * (lower + upper)
            reps.append(rep)
        return torch.stack(reps)

    def _initial_gamma_from_quantized_observations(self, Y, Omega):
        reps = self._bin_log_representatives()
        observed = Omega > 0
        if torch.any(observed):
            y_obs = torch.clamp(Y[observed].long(), min=0, max=reps.numel() - 1)
            u0 = torch.median(reps.index_select(0, y_obs.reshape(-1)))
        else:
            finite = reps[torch.isfinite(reps)]
            u0 = torch.median(finite) if finite.numel() > 0 else torch.zeros((), dtype=self.dtype, device=self.device)
        gamma0 = torch.exp(u0) - float(self.log_offset)
        return torch.clamp(gamma0, min=float(self.pred_epsilon)), u0

    def _initialize_quantized_state(self, N_grid, K, Y, Omega, debugFlag=False):
        gamma0, u0 = self._initial_gamma_from_quantized_observations(Y, Omega)
        sr0 = gamma0 / max(1, self.R)
        self._Theta_t = torch.zeros((N_grid, self.R, self.dim_poly), dtype=self.dtype, device=self.device)
        self._Theta_t[:, :, 0] = sr0
        self._Phi_t = torch.ones((self.R, K), dtype=self.dtype, device=self.device)
        self._Sr_t = torch.full((self.R, self.N1, self.N2), sr0, dtype=self.dtype, device=self.device)
        if debugFlag:
            print(
                f"  Init: u0={float(u0):.4g}, gamma0={float(gamma0):.4g}, "
                f"Sr0 per source={float(sr0):.4g}"
            )

    def quantize_log_observations(self, u, add_noise=True):
        u_t = self._to_tensor(u)
        if add_noise and self.sigma_q > 0:
            u_t = u_t + torch.randn_like(u_t) * self.sigma_q

        boundaries = self._quant_boundaries()
        y = torch.zeros_like(u_t, dtype=torch.long)
        for q in range(1, boundaries.numel() - 1):
            mask = (boundaries[q] < u_t) & (u_t <= boundaries[q + 1])
            y[mask] = q
        return y

    def quantize_measurements(self, gamma, add_noise=True):
        gamma_t = self._to_tensor(gamma)
        gamma_t = torch.clamp(gamma_t, min=0.0)
        u_t = torch.log(gamma_t + self.log_offset)
        return self.quantize_log_observations(u_t, add_noise=add_noise)

    # =================================================================
    # Prediction helpers
    # =================================================================

    def _theta_block_objective(self, theta_var, X_block, Y, Omega, W_block, target_sr):
        src_pred = torch.einsum("gmp,grp->gmr", X_block, theta_var)
        gamma_raw = torch.einsum("gmr,rk->gmk", src_pred, self._Phi_t.detach())
        gamma_pos = torch.clamp(gamma_raw, min=0.0) + self.pred_epsilon
        u_hat = torch.log(gamma_pos + self.log_offset)
        probs = self.probit_bin_prob(Y, u_hat)
        data_term = -(W_block[:, :, None] * Omega.unsqueeze(0) * torch.log(probs)).sum()
        consistency = 0.5 * self.nu * torch.sum((theta_var[:, :, 0] - target_sr) ** 2)
        return data_term + consistency

    def _phi_block_data_term(self, phi_var, pred_block, Omega, W_block, Y):
        gamma_raw = torch.einsum("gmr,rk->gmk", pred_block, phi_var)
        gamma_pos = torch.clamp(gamma_raw, min=0.0) + self.pred_epsilon
        u_hat = torch.log(gamma_pos + self.log_offset)
        probs = self.probit_bin_prob(Y, u_hat)
        return -(W_block[:, :, None] * Omega.unsqueeze(0) * torch.log(probs)).sum()

    def _resolve_active_grid_indices(self, Weights_raw, I_flat, grid_indices=None):
        if grid_indices is None:
            grid_idx_t = torch.arange(self.N1 * self.N2, dtype=torch.long, device=self.device)
        elif isinstance(grid_indices, torch.Tensor):
            grid_idx_t = grid_indices.to(device=self.device, dtype=torch.long).reshape(-1)
        else:
            grid_idx_t = torch.as_tensor(list(grid_indices), dtype=torch.long, device=self.device)

        total = int(grid_idx_t.numel())
        if total == 0:
            return grid_idx_t, 0, 0

        valid_mask = I_flat.index_select(0, grid_idx_t)
        support_mask = torch.any(Weights_raw.index_select(0, grid_idx_t) > 1e-6, dim=1)
        keep_mask = valid_mask & support_mask
        selected = grid_idx_t[keep_mask]
        skip_count = total - int(selected.numel())
        return selected, skip_count, total

    def _iter_grid_blocks(self, grid_idx_t):
        block = int(max(1, self.grid_batch_size))
        total = int(grid_idx_t.numel())
        for start in range(0, total, block):
            yield grid_idx_t[start:start + block]

    def _build_grid_block_features(self, locs_norm, grid_norm, Weights_raw, grid_block):
        centers = grid_norm.index_select(0, grid_block)
        diff = locs_norm.unsqueeze(0) - centers.unsqueeze(1)
        X_block = self._poly_features_from_diff(diff)
        W_block = Weights_raw.index_select(0, grid_block)
        return X_block, W_block

    def _estimate_grid_block_cache_bytes(self, num_grids, num_sensors):
        elem_size = torch.empty((), dtype=self.dtype).element_size()
        return int(num_grids) * int(num_sensors) * (self.dim_poly + 1) * elem_size

    def _should_materialize_grid_blocks(self, num_grids, num_sensors):
        if self.max_block_cache_bytes <= 0:
            return False
        estimated = self._estimate_grid_block_cache_bytes(num_grids, num_sensors)
        return estimated <= self.max_block_cache_bytes

    def _materialize_grid_block(self, grid_block, X_block, W_block, locs_norm, grid_norm, Weights_raw):
        if X_block is not None and W_block is not None:
            return X_block, W_block
        return self._build_grid_block_features(locs_norm, grid_norm, Weights_raw, grid_block)

    def _prepare_grid_blocks(self, locs_norm, grid_norm, Weights_raw, I_flat, grid_indices=None, materialize=None):
        grid_idx_t, skip_count, total = self._resolve_active_grid_indices(
            Weights_raw,
            I_flat,
            grid_indices=grid_indices,
        )
        if materialize is None:
            materialize = self._should_materialize_grid_blocks(int(grid_idx_t.numel()), int(locs_norm.shape[0]))
        blocks = []
        for grid_block in self._iter_grid_blocks(grid_idx_t):
            if materialize:
                X_block, W_block = self._build_grid_block_features(locs_norm, grid_norm, Weights_raw, grid_block)
            else:
                X_block, W_block = None, None
            blocks.append((grid_block, X_block, W_block))
        return blocks, skip_count, total, bool(materialize)

    # =================================================================
    # SVT
    # =================================================================

    def _stable_svd(self, matrix):
        if not bool(torch.all(torch.isfinite(matrix))):
            bad_count = int((~torch.isfinite(matrix)).sum().item())
            raise ValueError(f"SVD input contains {bad_count} non-finite values")

        if matrix.is_cuda:
            try:
                return torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "svd" not in msg and "cusolver" not in msg:
                    raise
                matrix_cpu = matrix.detach().cpu()
                U, s, Vh = torch.linalg.svd(matrix_cpu, full_matrices=False)
                return (
                    U.to(device=matrix.device, dtype=matrix.dtype),
                    s.to(device=matrix.device, dtype=matrix.dtype),
                    Vh.to(device=matrix.device, dtype=matrix.dtype),
                )
        return torch.linalg.svd(matrix, full_matrices=False)

    def _svt_operator(self, matrix, threshold):
        U, s, Vh = self._stable_svd(matrix)
        s_thresh = torch.clamp(s - threshold, min=0.0)
        rec = (U * s_thresh.unsqueeze(0)) @ Vh
        return torch.clamp(rec, min=0.0), s, s_thresh

    def _svt_operator_truncated(self, matrix, threshold, max_rank=None):
        U, s, Vh = self._stable_svd(matrix)
        if max_rank is not None:
            keep = min(int(max_rank), int(s.shape[0]))
            U = U[:, :keep]
            s = s[:keep]
            Vh = Vh[:keep, :]
        s_thresh = torch.clamp(s - threshold, min=0.0)
        nz = s_thresh > 0
        if torch.any(nz):
            rec = (U[:, nz] * s_thresh[nz].unsqueeze(0)) @ Vh[nz, :]
        else:
            rec = torch.zeros_like(matrix)
        return torch.clamp(rec, min=0.0), s, s_thresh

    # =================================================================
    # Step 1: Theta update
    # =================================================================

    def _update_theta_quantized(
        self,
        Y,
        Omega,
        grid_blocks,
        locs_norm,
        grid_norm,
        Weights_raw,
        skip_count,
        total,
        debugFlag=False,
    ):
        if not grid_blocks:
            if debugFlag:
                print(f"  Quantized Step1 done. Skipped {skip_count}/{total} grids.")
            return dict(blocks=0, grids=0, avg_inner=0.0, avg_loss=float("nan"), converged_blocks=0)

        block_count = 0
        grid_count = 0
        total_inner = 0
        total_final_loss = 0.0
        converged_blocks = 0
        for grid_block, X_block, W_block in grid_blocks:
            X_block, W_block = self._materialize_grid_block(
                grid_block,
                X_block,
                W_block,
                locs_norm,
                grid_norm,
                Weights_raw,
            )
            theta_block = self._Theta_t.index_select(0, grid_block).detach().clone().requires_grad_(True)

            i_g = torch.div(grid_block, self.N2, rounding_mode="floor")
            j_g = grid_block % self.N2
            target_sr = self._Sr_t[:, i_g, j_g].permute(1, 0).detach()

            optimizer = torch.optim.Adam([theta_block], lr=self.theta_lr)
            prev_loss = None
            final_loss_val = float("nan")
            inner_steps = 0
            converged = False

            for inner_idx in range(self.theta_inner_iter):
                optimizer.zero_grad()
                loss = self._theta_block_objective(
                    theta_block,
                    X_block,
                    Y,
                    Omega,
                    W_block,
                    target_sr,
                )
                loss.backward()
                optimizer.step()

                loss_val = float(loss.detach())
                final_loss_val = loss_val
                inner_steps = inner_idx + 1
                if prev_loss is not None:
                    rel = abs(loss_val - prev_loss) / (abs(prev_loss) + 1e-12)
                    if rel <= self.inner_tol:
                        converged = True
                        break
                prev_loss = loss_val

            with torch.no_grad():
                self._Theta_t.index_copy_(0, grid_block, theta_block.detach())

            block_count += 1
            grid_count += int(grid_block.numel())
            total_inner += inner_steps
            total_final_loss += final_loss_val
            converged_blocks += int(converged)

        avg_inner = total_inner / max(1, block_count)
        avg_loss = total_final_loss / max(1, block_count)
        summary = dict(
            blocks=block_count,
            grids=grid_count,
            avg_inner=avg_inner,
            avg_loss=avg_loss,
            converged_blocks=converged_blocks,
        )
        if debugFlag:
            print(
                f"  Quantized Step1 Theta: grids={grid_count}, blocks={block_count}, "
                f"skipped={skip_count}/{total}, avg_inner={avg_inner:.2f}, "
                f"avg_loss={avg_loss:.4g}, converged_blocks={converged_blocks}/{block_count}, "
                f"Theta range={self._debug_range(self._Theta_t)}"
            )
        return summary

    # =================================================================
    # Step 2: Phi update
    # =================================================================

    def _update_phi_quantized(
        self,
        Y,
        Omega,
        grid_blocks,
        locs_norm,
        grid_norm,
        Weights_raw,
        debugFlag=False,
    ):
        if not grid_blocks:
            return dict(inner_steps=0, final_loss=float("nan"), final_rel=float("nan"))

        pred_blocks = []
        for grid_block, X_block, W_block in grid_blocks:
            X_block, W_block = self._materialize_grid_block(
                grid_block,
                X_block,
                W_block,
                locs_norm,
                grid_norm,
                Weights_raw,
            )
            theta_block = self._Theta_t.index_select(0, grid_block).detach()
            pred_block = torch.einsum("gmp,grp->gmr", X_block, theta_block)
            pred_blocks.append((pred_block, W_block))

        phi_var = self._Phi_t.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([phi_var], lr=self.phi_lr)
        prev_loss = None
        final_loss_val = float("nan")
        final_rel = float("nan")
        inner_steps = 0

        for inner_idx in range(self.phi_inner_iter):
            optimizer.zero_grad()
            total_loss_val = 0.0

            for pred_block, W_block in pred_blocks:
                block_loss = self._phi_block_data_term(
                    phi_var,
                    pred_block,
                    Omega,
                    W_block,
                    Y=Y,
                )
                block_loss.backward()
                total_loss_val += float(block_loss.detach())

            reg = 0.5 * self.lambda_phi * torch.sum(phi_var ** 2)
            reg.backward()
            total_loss_val += float(reg.detach())

            optimizer.step()
            with torch.no_grad():
                phi_var.clamp_(min=0.0)

            final_loss_val = total_loss_val
            inner_steps = inner_idx + 1
            if prev_loss is not None:
                rel = abs(total_loss_val - prev_loss) / (abs(prev_loss) + 1e-12)
                final_rel = rel
                if rel <= self.inner_tol:
                    break
            prev_loss = total_loss_val

        with torch.no_grad():
            self._Phi_t.copy_(phi_var.detach())

        summary = dict(inner_steps=inner_steps, final_loss=final_loss_val, final_rel=final_rel)
        if debugFlag:
            print(
                f"  Quantized Step2 Phi: inner_steps={inner_steps}, "
                f"final_loss={final_loss_val:.4g}, final_rel={final_rel:.4g}, "
                f"Phi range={self._debug_range(self._Phi_t)}"
            )
        return summary

    # =================================================================
    # Step 3: Sr update
    # =================================================================

    def _update_sr(self, I_mask, max_svt_iter=100, use_truncated_svd=True, max_rank=None, debugFlag=False):
        mask = I_mask if isinstance(I_mask, torch.Tensor) else self._to_tensor(I_mask)
        svt_iters = []
        for r in range(self.R):
            Psi = self._Theta_t[:, r, 0].reshape(self.N1, self.N2)

            if debugFlag:
                psi_min = float(torch.min(Psi))
                psi_max = float(torch.max(Psi))
                print(f"  Quantized Step3 Source {r}: Psi range [{psi_min:.3f}, {psi_max:.3f}]")

            Y_aux = self._Sr_t[r].clone()
            delta_svt = 1.2
            psi_norm = torch.linalg.norm(Psi) + 1e-9

            for t in range(max_svt_iter):
                if use_truncated_svd:
                    S_new, _, _ = self._svt_operator_truncated(Y_aux, self.mu, max_rank)
                else:
                    S_new, _, _ = self._svt_operator(Y_aux, self.mu)

                residual = mask * (Psi - S_new)
                res_norm = torch.linalg.norm(residual)
                if float(res_norm) < float(1e-4 * psi_norm):
                    break
                Y_aux = Y_aux + delta_svt * residual

            self._Sr_t[r] = S_new
            svt_iters.append(t + 1)

            if debugFlag:
                sr_min = float(torch.min(self._Sr_t[r]))
                sr_max = float(torch.max(self._Sr_t[r]))
                print(f"  Quantized Step3 Source {r}: Sr range [{sr_min:.3f}, {sr_max:.3f}], SVT iters={t + 1}")
        avg_svt = float(np.mean(svt_iters)) if svt_iters else 0.0
        max_svt = int(max(svt_iters)) if svt_iters else 0
        summary = dict(avg_svt_iter=avg_svt, max_svt_iter=max_svt)
        if debugFlag:
            print(
                f"  Quantized Step3 Sr summary: avg_svt_iter={avg_svt:.2f}, "
                f"max_svt_iter={max_svt}, Sr range={self._debug_range(self._Sr_t)}"
            )
        return summary

    # =================================================================
    # Main fit
    # =================================================================

    def _post_update_maps(self):
        self._H_hat_t = torch.einsum("rxy,rk->xyk", self._Sr_t, self._Phi_t)
        self._U_hat_t = torch.log(torch.clamp(self._H_hat_t, min=0.0) + self.log_offset)

    def _clear_model_state(self):
        self.Theta = None
        self.Phi = None
        self.Sr = None
        self.H_hat = None
        self.U_hat = None

        self._Theta_t = None
        self._Phi_t = None
        self._Sr_t = None
        self._H_hat_t = None
        self._U_hat_t = None

        self._sensor_locs_t = None
        self._Y_t = None
        self._Omega_t = None
        self._grid_coords_t = None
        self._locs_norm_t = None
        self._grid_norm_t = None
        self._Weights_raw_t = None
        self._initialized = False

    def global_reconstruct(self, sensor_locs, Y, Omega, grid_coords, bounds, I_mask=None, debugFlag=False, reset_state=True):
        """
        Full-batch reconstruction using all currently available quantized observations.

        This is the closest counterpart to the static quantized BTD routine:
        the solver consumes the whole observation set in one optimization call.

        Parameters
        ----------
        reset_state : bool, default=True
            When True, force a fresh global reconstruction instead of warm-starting
            from the current internal state.
        """
        if reset_state:
            self._clear_model_state()
        return self.fit(sensor_locs, Y, Omega, grid_coords, bounds, I_mask=I_mask, debugFlag=debugFlag)

    def fit(self, sensor_locs, Y, Omega, grid_coords, bounds, I_mask=None, debugFlag=False):
        sensor_locs_t = self._to_tensor(self._atleast_2d(sensor_locs))
        grid_coords_t = self._to_tensor(self._atleast_2d(grid_coords))
        Y_t, Omega_t = self._prepare_observation_tensors(Y, Omega)
        self._validate_reconstruction_inputs(sensor_locs_t, Y_t, Omega_t, grid_coords_t)
        self._require_positive_sigma_q()

        M, K = int(Y_t.shape[0]), int(Y_t.shape[1])
        N_grid = self.N1 * self.N2

        self._bounds = bounds
        self._make_normalizer(bounds)
        locs_norm = self._normalize(sensor_locs_t)
        grid_norm = self._normalize(grid_coords_t)

        self._set_i_mask(I_mask)
        I_flat = self._I_mask_bool_t.reshape(-1)

        if not self.warmstart or self._Theta_t is None or not self._initialized:
            self._initialize_quantized_state(N_grid, K, Y_t, Omega_t, debugFlag=debugFlag)

        dists_full = torch.cdist(grid_norm, locs_norm)
        Weights_raw = self._epanechnikov_kernel(dists_full)

        if debugFlag:
            self._print_debug_fit_header(Y_t, Omega_t, N_grid, M, K, I_flat)

        use_truncated = min(self.N1, self.N2) > 15
        svt_max_rank = min(10, min(self.N1, self.N2) - 1) if use_truncated else None
        grid_blocks, skip_count, total, blocks_materialized = self._prepare_grid_blocks(
            locs_norm,
            grid_norm,
            Weights_raw,
            I_flat,
        )
        if debugFlag:
            cache_mode = "cached" if blocks_materialized else "streamed"
            svt_mode = "truncated" if use_truncated else "full"
            print(
                f"  Grid blocks: mode={cache_mode}, block_size={self.grid_batch_size}, "
                f"blocks={len(grid_blocks)}, selected={total - skip_count}, skipped={skip_count}/{total}"
            )
            print(
                f"  Kernel weights range={self._debug_range(Weights_raw)}, "
                f"SVT={svt_mode}, svt_max_rank={svt_max_rank}"
            )

        for iteration in range(self.max_iter):
            Sr_old = self._Sr_t.clone()

            theta_debug = self._update_theta_quantized(
                Y_t,
                Omega_t,
                grid_blocks,
                locs_norm,
                grid_norm,
                Weights_raw,
                skip_count,
                total,
                debugFlag=debugFlag,
            )
            phi_debug = self._update_phi_quantized(
                Y_t,
                Omega_t,
                grid_blocks,
                locs_norm,
                grid_norm,
                Weights_raw,
                debugFlag=debugFlag,
            )
            sr_debug = self._update_sr(
                self._I_mask_t,
                max_svt_iter=40,
                use_truncated_svd=use_truncated,
                max_rank=svt_max_rank,
                debugFlag=debugFlag,
            )

            diff = torch.linalg.norm(self._Sr_t - Sr_old) / (torch.linalg.norm(Sr_old) + 1e-9)
            diff_val = float(diff)
            if debugFlag:
                self._post_update_maps()
                print(
                    f"  Quantized Iter {iteration + 1}/{self.max_iter}: "
                    f"rel_change={diff_val:.6g}, "
                    f"theta_avg_inner={theta_debug['avg_inner']:.2f}, "
                    f"phi_inner={phi_debug['inner_steps']}, "
                    f"avg_svt={sr_debug['avg_svt_iter']:.2f}, "
                    f"H range={self._debug_range(self._H_hat_t)}, "
                    f"U range={self._debug_range(self._U_hat_t)}\n"
                )
            if diff_val < self.tol and diff_val > 0:
                if debugFlag:
                    print(f"  Quantized solver converged at iteration {iteration + 1}.")
                break
            if diff_val == 0 and iteration > 0:
                if debugFlag:
                    print(f"  Quantized solver had no change at iteration {iteration + 1}, stopping.")
                break

        self._post_update_maps()

        self._initialized = True
        self._sensor_locs_t = sensor_locs_t.clone()
        self._Y_t = Y_t.clone()
        self._Omega_t = Omega_t.clone()
        self._grid_coords_t = grid_coords_t.clone()
        self._locs_norm_t = locs_norm.clone()
        self._grid_norm_t = grid_norm.clone()
        self._Weights_raw_t = Weights_raw.clone()

        self._sync_public_state()
        self._sync_public_cache()
        return self

    fit_2 = fit

    # =================================================================
    # Accessors and utilities
    # =================================================================

    def get_current_map(self):
        return None if self.H_hat is None else self.H_hat.copy()

    def get_current_log_map(self):
        return None if self.U_hat is None else self.U_hat.copy()

    def get_source_maps(self):
        return None if self.Sr is None else self.Sr.copy()

    def get_spectra(self):
        return None if self.Phi is None else self.Phi.copy()

    def get_quantized_prediction(self, add_noise=False):
        if self.H_hat is None:
            return None
        h_hat_t = self._to_tensor(self.H_hat)
        return self._to_numpy(self.quantize_measurements(h_hat_t, add_noise=add_noise))
