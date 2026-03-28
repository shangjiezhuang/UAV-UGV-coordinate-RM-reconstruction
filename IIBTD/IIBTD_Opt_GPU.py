import numpy as np
import torch
from scipy.optimize import nnls


class II_BTD_Opt_GPU:
    """
    Torch/CUDA-backed II-BTD implementation.

    Notes
    -----
    - Dense tensor algebra, distance computation, linear solves, and SVD run on
      the selected torch device.
    - The Phi update supports two NNLS backends:
        * "scipy": uses scipy.optimize.nnls on CPU for accuracy parity
        * "pgd": projected-gradient NNLS on the torch device
      The default keeps the heavy tensor work on GPU while preserving close
      agreement with the existing CPU implementation.
    - Public attributes (Theta/Phi/Sr/H_hat) are synchronized back to numpy so
      downstream code can keep the same access pattern as the CPU solver.
    """

    def __init__(
        self,
        n_sources=2,
        grid_size=(20, 20),
        mu=1.0,
        nu=1.0,
        max_iter=30,
        tol=1e-5,
        kernel_bandwidth=1,
        warmstart=False,
        device=None,
        dtype=torch.float64,
        phi_solver="scipy",
        pgd_max_iter=250,
        pgd_tol=1e-10,
    ):
        self.R = n_sources
        self.N1, self.N2 = grid_size
        self.mu = mu
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.h = kernel_bandwidth
        self.warmstart = warmstart
        self.dim_poly = 6
        self.nmse_list = []
        self.GT_PHI = None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.phi_solver = str(phi_solver).lower()
        self.pgd_max_iter = int(max(1, pgd_max_iter))
        self.pgd_tol = float(max(0.0, pgd_tol))

        self._initialized = False
        self._bounds = None

        self.Theta = None
        self.Phi = None
        self.Sr = None
        self.H_hat = None

        self._Theta_t = None
        self._Phi_t = None
        self._Sr_t = None
        self._H_hat_t = None

        self._sensor_locs_t = None
        self._Gamma_t = None
        self._Omega_t = None
        self._grid_coords_t = None
        self._locs_norm_t = None
        self._grid_norm_t = None
        self._Weights_raw_t = None
        self._I_mask_bool_t = None
        self._I_mask_t = None

        self._sensor_locs = None
        self._Gamma = None
        self._Omega = None
        self._grid_coords = None
        self._locs_norm = None
        self._grid_norm = None
        self._Weights_raw = None
        self.I_mask = None

    # =================================================================
    # Helpers
    # =================================================================

    def _to_tensor(self, arr, dtype=None, device=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=dtype)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    def _to_bool_tensor(self, arr, device=None):
        if device is None:
            device = self.device
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=torch.bool)
        return torch.as_tensor(arr, dtype=torch.bool, device=device)

    @staticmethod
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _sync_public_state(self):
        self.Theta = None if self._Theta_t is None else self._to_numpy(self._Theta_t)
        self.Phi = None if self._Phi_t is None else self._to_numpy(self._Phi_t)
        self.Sr = None if self._Sr_t is None else self._to_numpy(self._Sr_t)
        self.H_hat = None if self._H_hat_t is None else self._to_numpy(self._H_hat_t)

    def _sync_public_cache(self):
        self._sensor_locs = None if self._sensor_locs_t is None else self._to_numpy(self._sensor_locs_t)
        self._Gamma = None if self._Gamma_t is None else self._to_numpy(self._Gamma_t)
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

    def load_state_from(self, src_model):
        for name in ("Theta", "Phi", "Sr", "H_hat"):
            if hasattr(src_model, name):
                value = np.array(getattr(src_model, name), copy=True)
                setattr(self, name, value)
                tensor_attr = f"_{name}_t"
                if hasattr(self, tensor_attr):
                    setattr(self, tensor_attr, self._to_tensor(value))
        return self

    # =================================================================
    # Kernel and features
    # =================================================================

    def _epanechnikov_kernel(self, dists):
        u = dists / self.h
        k = 0.75 * (1 - u ** 2)
        return torch.where(torch.abs(u) <= 1.0, k, torch.zeros_like(k))

    def _get_poly_features(self, coords_norm, center_norm):
        diff = coords_norm - center_norm
        return self._poly_features_from_diff(diff)

    def _poly_features_from_diff(self, diff):
        dx = diff[..., 0]
        dy = diff[..., 1]
        ones = torch.ones_like(dx)
        return torch.stack((ones, dx, dy, dx ** 2, dx * dy, dy ** 2), dim=-1)

    def _stable_svd(self, matrix):
        if not bool(torch.all(torch.isfinite(matrix))):
            bad_count = int((~torch.isfinite(matrix)).sum().item())
            raise ValueError(f"SVD input contains {bad_count} non-finite values")

        if matrix.is_cuda:
            try:
                # `gesvd` is slower than the default Jacobi driver but is more
                # stable for the ill-conditioned matrices that appear in SVT.
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
    # Normalization
    # =================================================================

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
    # Precompute
    # =================================================================

    def _precompute_phi_dependent(self, Omega, Gamma):
        self._PhiOmPhiT_all = torch.einsum("rk,mk,sk->mrs", self._Phi_t, Omega, self._Phi_t)
        self._PhiGam_all = torch.einsum("rk,mk->mr", self._Phi_t, Gamma * Omega)

    # =================================================================
    # Step 1: Theta update
    # =================================================================

    def _update_theta(
        self,
        locs_norm,
        grid_norm,
        Gamma,
        Omega,
        Weights_raw,
        I_flat,
        grid_indices=None,
        debugFlag=False,
    ):
        self._precompute_phi_dependent(Omega, Gamma)
        PhiOmPhiT = self._PhiOmPhiT_all
        PhiGam = self._PhiGam_all

        if grid_indices is None:
            grid_idx_t = torch.arange(self.N1 * self.N2, dtype=torch.long, device=self.device)
        elif isinstance(grid_indices, torch.Tensor):
            grid_idx_t = grid_indices.to(device=self.device, dtype=torch.long).reshape(-1)
        else:
            grid_idx_t = torch.as_tensor(list(grid_indices), dtype=torch.long, device=self.device)
        total = int(grid_idx_t.numel())

        if self.R == 1:
            valid_mask = I_flat.index_select(0, grid_idx_t)
            candidate_idx = grid_idx_t[valid_mask]
            if candidate_idx.numel() == 0:
                if debugFlag:
                    print(f"  Step1 GPU done. Skipped {total}/{total} grids.")
                return

            weights_sel = Weights_raw.index_select(0, candidate_idx)
            active_mask = torch.any(weights_sel > 1e-6, dim=1)
            valid_idx = candidate_idx[active_mask]
            skip_count = total - int(valid_idx.numel())
            if valid_idx.numel() == 0:
                if debugFlag:
                    print(f"  Step1 GPU done. Skipped {skip_count}/{total} grids.")
                return

            weights_sel = weights_sel[active_mask]
            grid_sel = grid_norm.index_select(0, valid_idx)
            diff = locs_norm.unsqueeze(0) - grid_sel.unsqueeze(1)
            X = self._poly_features_from_diff(diff)

            coeff = weights_sel * PhiOmPhiT[:, 0, 0].unsqueeze(0)
            ATA = torch.einsum("gm,gmi,gmj->gij", coeff, X, X)
            ATA[:, 0, 0] += self.nu

            coeff_b = weights_sel * PhiGam[:, 0].unsqueeze(0)
            ATb = torch.einsum("gm,gmi->gi", coeff_b, X)
            i_g = torch.div(valid_idx, self.N2, rounding_mode="floor")
            j_g = torch.remainder(valid_idx, self.N2)
            ATb[:, 0] += self.nu * self._Sr_t[0, i_g, j_g]

            eye = torch.eye(self.dim_poly, dtype=self.dtype, device=self.device).unsqueeze(0)
            ATA = ATA + eye * 1e-7
            try:
                theta = torch.linalg.solve(ATA, ATb.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                theta = torch.linalg.lstsq(ATA, ATb.unsqueeze(-1)).solution.squeeze(-1)
            self._Theta_t[valid_idx, 0, :] = theta

            if debugFlag:
                print(f"  Step1 GPU done. Skipped {skip_count}/{total} grids.")
            return

        R = self.R
        d = self.dim_poly
        dR = d * R
        ATA_base = torch.zeros((dR, dR), dtype=self.dtype, device=self.device)
        for r in range(R):
            ATA_base[r * d, r * d] = self.nu
        eye = torch.eye(dR, dtype=self.dtype, device=self.device)

        grid_indices_iter = [int(v) for v in grid_idx_t.detach().cpu().tolist()]

        skip_count = 0
        for idx in grid_indices_iter:
            if not bool(I_flat[idx]):
                skip_count += 1
                continue

            w_vec = Weights_raw[idx]
            active = w_vec > 1e-6
            if not bool(torch.any(active)):
                skip_count += 1
                continue

            active_idx = torch.where(active)[0]
            w_act = w_vec.index_select(0, active_idx)

            X_act = self._get_poly_features(locs_norm.index_select(0, active_idx), grid_norm[idx])
            XXT_act = torch.einsum("ni,nj->nij", X_act, X_act)

            ATA = ATA_base.clone()
            ATb = torch.zeros(dR, dtype=self.dtype, device=self.device)

            i_g, j_g = idx // self.N2, idx % self.N2
            for r in range(R):
                ATb[r * d] += self.nu * self._Sr_t[r, i_g, j_g]

            PhiOmPhiT_act = PhiOmPhiT.index_select(0, active_idx)
            PhiGam_act = PhiGam.index_select(0, active_idx)

            for r1 in range(R):
                row_slice = slice(r1 * d, (r1 + 1) * d)
                for r2 in range(R):
                    col_slice = slice(r2 * d, (r2 + 1) * d)
                    coeffs = w_act * PhiOmPhiT_act[:, r1, r2]
                    ATA[row_slice, col_slice] += torch.einsum("n,nij->ij", coeffs, XXT_act)

            for r in range(R):
                coeffs_b = w_act * PhiGam_act[:, r]
                block = slice(r * d, (r + 1) * d)
                ATb[block] += torch.einsum("n,ni->i", coeffs_b, X_act)

            ATA = ATA + eye * 1e-7
            try:
                theta_flat = torch.linalg.solve(ATA, ATb.unsqueeze(1)).squeeze(1)
            except RuntimeError:
                theta_flat = torch.linalg.lstsq(ATA, ATb.unsqueeze(1)).solution.squeeze(1)

            self._Theta_t[idx] = theta_flat.reshape(R, d)

        if debugFlag:
            print(f"  Step1 GPU done. Skipped {skip_count}/{total} grids.")

    # =================================================================
    # Step 2: Phi update
    # =================================================================

    def _nnls_pgd(self, A, b):
        n_var = A.shape[1]
        if A.numel() == 0:
            return torch.zeros(n_var, dtype=self.dtype, device=self.device)

        try:
            x = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
            x = torch.clamp(x, min=0.0)
        except RuntimeError:
            x = torch.zeros(n_var, dtype=self.dtype, device=self.device)

        AtA = A.T @ A
        Atb = A.T @ b

        if n_var == 1:
            L = float(torch.clamp(AtA[0, 0], min=1e-12).item())
        else:
            eigvals = torch.linalg.eigvalsh(AtA)
            L = float(torch.clamp(eigvals[-1], min=1e-12).item())
        step = 1.0 / (L + 1e-12)

        for _ in range(self.pgd_max_iter):
            grad = AtA @ x - Atb
            x_next = torch.clamp(x - step * grad, min=0.0)
            delta = torch.linalg.norm(x_next - x)
            denom = torch.linalg.norm(x) + 1e-12
            x = x_next
            if float(delta / denom) <= self.pgd_tol:
                break
        return x

    def _solve_nnls(self, A, b):
        if self.phi_solver == "pgd":
            return self._nnls_pgd(A, b)
        if self.phi_solver != "scipy":
            raise ValueError(f"Unsupported phi_solver: {self.phi_solver}")

        A_np = self._to_numpy(A)
        b_np = self._to_numpy(b)
        sol, _ = nnls(A_np, b_np)
        return torch.as_tensor(sol, dtype=self.dtype, device=self.device)

    def _update_phi_vectorized(self, locs_norm, grid_norm, Gamma, Omega, Weights_raw, I_flat, debugFlag=False):
        N_grid = self.N1 * self.N2
        R = self.R
        K = Gamma.shape[1]

        if self.R == 1:
            grid_idx_t = torch.where(I_flat)[0]
            if grid_idx_t.numel() == 0:
                return

            weights_sel = Weights_raw.index_select(0, grid_idx_t)
            active_mask = torch.any(weights_sel > 1e-6, dim=1)
            grid_idx_t = grid_idx_t[active_mask]
            if grid_idx_t.numel() == 0:
                return

            weights_sel = weights_sel[active_mask]
            grid_sel = grid_norm.index_select(0, grid_idx_t)
            diff = locs_norm.unsqueeze(0) - grid_sel.unsqueeze(1)
            X = self._poly_features_from_diff(diff)
            theta = self._Theta_t.index_select(0, grid_idx_t)[:, 0, :]
            pred = torch.einsum("gmd,gd->gm", X, theta)

            weighted_pred_sum = torch.sum(weights_sel * pred, dim=0)
            weighted_pred_sq_sum = torch.sum(weights_sel * pred ** 2, dim=0)

            gamma_obs = Gamma * Omega
            numerator = torch.sum(weighted_pred_sum.unsqueeze(1) * gamma_obs, dim=0)
            denominator = torch.sum(weighted_pred_sq_sum.unsqueeze(1) * Omega, dim=0)
            phi = torch.clamp(numerator / (denominator + 1e-12), min=0.0)
            self._Phi_t[0] = phi

            if float(torch.max(self._Phi_t[0])) < 1e-10:
                self._Phi_t[0] = torch.abs(torch.randn(K, device=self.device, dtype=self.dtype)) * 0.01

            if debugFlag:
                phi_min = float(torch.min(self._Phi_t))
                phi_max = float(torch.max(self._Phi_t))
                print(f"  Step2 GPU done. Phi range: [{phi_min:.4f}, {phi_max:.4f}]")
            return

        all_pred = []
        all_sqrt_w = []
        all_m_global = []

        for idx in range(N_grid):
            if not bool(I_flat[idx]):
                continue

            w_vec = Weights_raw[idx]
            active = w_vec > 1e-6
            if not bool(torch.any(active)):
                continue

            active_idx = torch.where(active)[0]
            w_act = w_vec.index_select(0, active_idx)
            X_act = self._get_poly_features(locs_norm.index_select(0, active_idx), grid_norm[idx])
            Theta_ij_mat = self._Theta_t[idx].T
            Pred = X_act @ Theta_ij_mat

            all_pred.append(Pred)
            all_sqrt_w.append(torch.sqrt(torch.clamp(w_act, min=0.0)))
            all_m_global.append(active_idx)

        if not all_pred:
            return

        all_pred = torch.cat(all_pred, dim=0)
        all_sqrt_w = torch.cat(all_sqrt_w, dim=0)
        all_m_global = torch.cat(all_m_global, dim=0).long()

        for k in range(K):
            mask_k = Omega[all_m_global, k] > 0.5
            if not bool(torch.any(mask_k)):
                self._Phi_t[:, k] = 0.0
                continue

            pred_k = all_pred[mask_k]
            sw_k = all_sqrt_w[mask_k]
            m_k = all_m_global[mask_k]

            A_k = pred_k * sw_k.unsqueeze(1)
            b_k = sw_k * Gamma[m_k, k]
            phi_k = self._solve_nnls(A_k, b_k)
            self._Phi_t[:, k] = phi_k

        for r in range(R):
            if float(torch.max(self._Phi_t[r])) < 1e-10:
                self._Phi_t[r] = torch.abs(torch.randn(K, device=self.device, dtype=self.dtype)) * 0.01

        if debugFlag:
            phi_min = float(torch.min(self._Phi_t))
            phi_max = float(torch.max(self._Phi_t))
            print(f"  Step2 GPU done. Phi range: [{phi_min:.4f}, {phi_max:.4f}]")

    # =================================================================
    # Step 3: Sr update
    # =================================================================

    def _update_sr(self, I_mask, max_svt_iter=100, use_truncated_svd=True, max_rank=None, debugFlag=False):
        mask = I_mask if isinstance(I_mask, torch.Tensor) else self._to_tensor(I_mask)
        for r in range(self.R):
            Psi = self._Theta_t[:, r, 0].reshape(self.N1, self.N2)

            if debugFlag:
                psi_min = float(torch.min(Psi))
                psi_max = float(torch.max(Psi))
                print(f"  Step3 GPU Source {r}: Psi range [{psi_min:.3f}, {psi_max:.3f}]")

            Y = self._Sr_t[r].clone()
            delta_svt = 1.2
            psi_norm = torch.linalg.norm(Psi) + 1e-9

            for t in range(max_svt_iter):
                if use_truncated_svd:
                    S_new, _, _ = self._svt_operator_truncated(Y, self.mu, max_rank)
                else:
                    S_new, _, _ = self._svt_operator(Y, self.mu)

                residual = mask * (Psi - S_new)
                res_norm = torch.linalg.norm(residual)
                if float(res_norm) < float(1e-4 * psi_norm):
                    break
                Y = Y + delta_svt * residual

            self._Sr_t[r] = S_new

            if debugFlag:
                sr_min = float(torch.min(self._Sr_t[r]))
                sr_max = float(torch.max(self._Sr_t[r]))
                print(f"  Step3 GPU Source {r}: Sr range [{sr_min:.3f}, {sr_max:.3f}], SVT iters={t + 1}")

    # =================================================================
    # Main batch fit
    # =================================================================

    def fit_2(self, sensor_locs, Gamma, Omega, grid_coords, bounds, I_mask=None, debugFlag=False):
        sensor_locs_t = self._to_tensor(sensor_locs)
        Gamma_t = self._to_tensor(Gamma)
        Omega_t = self._to_tensor(Omega)
        grid_coords_t = self._to_tensor(grid_coords)

        M, K = Gamma_t.shape
        N_grid = self.N1 * self.N2

        self._bounds = bounds
        self._make_normalizer(bounds)
        locs_norm = self._normalize(sensor_locs_t)
        grid_norm = self._normalize(grid_coords_t)

        self._set_i_mask(I_mask)
        I_flat = self._I_mask_bool_t.reshape(-1)

        if not self.warmstart or self._Theta_t is None or not self._initialized:
            self._Theta_t = torch.zeros((N_grid, self.R, self.dim_poly), dtype=self.dtype, device=self.device)
            self._Phi_t = torch.ones((self.R, K), dtype=self.dtype, device=self.device)
            self._Phi_t = self._Phi_t * (K / torch.sum(self._Phi_t, dim=1, keepdim=True))
            self._Sr_t = torch.zeros((self.R, self.N1, self.N2), dtype=self.dtype, device=self.device)

        dists_full = torch.cdist(grid_norm, locs_norm)
        Weights_raw = self._epanechnikov_kernel(dists_full)

        if debugFlag:
            print(f"Starting GPU Optimization: N_grid={N_grid}, M={M}, K={K}, R={self.R}, device={self.device}")
            print(f"  |I| = {int(torch.sum(I_flat))}/{N_grid}, nu={self.nu}, mu={self.mu}")

        use_truncated = min(self.N1, self.N2) > 15
        svt_max_rank = min(10, min(self.N1, self.N2) - 1) if use_truncated else None

        for iteration in range(self.max_iter):
            Sr_old = self._Sr_t.clone()

            self._update_theta(locs_norm, grid_norm, Gamma_t, Omega_t, Weights_raw, I_flat, debugFlag=debugFlag)
            self._update_phi_vectorized(locs_norm, grid_norm, Gamma_t, Omega_t, Weights_raw, I_flat, debugFlag=debugFlag)
            self._update_sr(self._I_mask_t, max_svt_iter=40, use_truncated_svd=use_truncated, max_rank=svt_max_rank, debugFlag=debugFlag)

            diff = torch.linalg.norm(self._Sr_t - Sr_old) / (torch.linalg.norm(Sr_old) + 1e-9)
            diff_val = float(diff)
            if debugFlag:
                print(f"  GPU Iter {iteration + 1}/{self.max_iter}, Relative Change: {diff_val:.6f}\n")
            if diff_val < self.tol and diff_val > 0:
                if debugFlag:
                    print(f"  GPU converged at iteration {iteration + 1}.")
                break
            if diff_val == 0 and iteration > 0:
                if debugFlag:
                    print(f"  GPU no change at iteration {iteration + 1}, stopping.")
                break

        self._H_hat_t = torch.einsum("rxy,rk->xyk", self._Sr_t, self._Phi_t)

        self._initialized = True
        self._sensor_locs_t = sensor_locs_t.clone()
        self._Gamma_t = Gamma_t.clone()
        self._Omega_t = Omega_t.clone()
        self._grid_coords_t = grid_coords_t.clone()
        self._locs_norm_t = locs_norm.clone()
        self._grid_norm_t = grid_norm.clone()
        self._Weights_raw_t = Weights_raw.clone()

        self._sync_public_state()
        self._sync_public_cache()
        return self

    # =================================================================
    # Sequential / incremental updates
    # =================================================================

    def _get_affected_grids(self, new_sensor_locs_norm):
        dists = torch.cdist(self._grid_norm_t, new_sensor_locs_norm)
        in_support = torch.any(dists / self.h < 1.0, dim=1)
        return torch.where(in_support)[0]

    def init_sequential(self, grid_coords, bounds, K, I_mask=None):
        N_grid = self.N1 * self.N2
        self._bounds = bounds
        self._make_normalizer(bounds)

        self._grid_coords_t = self._to_tensor(grid_coords)
        self._grid_norm_t = self._normalize(self._grid_coords_t)
        self._set_i_mask(I_mask)

        self._Theta_t = torch.zeros((N_grid, self.R, self.dim_poly), dtype=self.dtype, device=self.device)
        self._Phi_t = torch.ones((self.R, K), dtype=self.dtype, device=self.device)
        self._Phi_t = self._Phi_t * (K / torch.sum(self._Phi_t, dim=1, keepdim=True))
        self._Sr_t = torch.zeros((self.R, self.N1, self.N2), dtype=self.dtype, device=self.device)
        self._H_hat_t = torch.zeros((self.N1, self.N2, K), dtype=self.dtype, device=self.device)

        self._sensor_locs_t = torch.empty((0, 2), dtype=self.dtype, device=self.device)
        self._locs_norm_t = torch.empty((0, 2), dtype=self.dtype, device=self.device)
        self._Gamma_t = torch.empty((0, K), dtype=self.dtype, device=self.device)
        self._Omega_t = torch.empty((0, K), dtype=self.dtype, device=self.device)
        self._Weights_raw_t = torch.empty((N_grid, 0), dtype=self.dtype, device=self.device)

        self._initialized = True
        self._sync_public_state()
        self._sync_public_cache()
        return self

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
        if not self._initialized:
            if grid_coords is None or bounds is None:
                raise ValueError("首次调用 fit_incremental 必须提供 grid_coords 和 bounds")
            K = np.atleast_2d(new_gamma).shape[1]
            self.init_sequential(grid_coords, bounds, K=K, I_mask=I_mask)
        elif I_mask is not None:
            self._set_i_mask(I_mask)

        return self.add_measurements(
            new_sensor_locs,
            new_gamma,
            new_omega,
            n_outer_iter=n_outer_iter,
            max_svt_iter=max_svt_iter,
            debugFlag=debugFlag,
        )

    def add_measurements(self, new_sensor_locs, new_gamma, new_omega, n_outer_iter=2, max_svt_iter=20, debugFlag=False):
        if not self._initialized:
            raise RuntimeError("请先调用 init_sequential() 完成初始化")

        new_sensor_locs_t = self._to_tensor(np.atleast_2d(new_sensor_locs))
        new_gamma_t = self._to_tensor(np.atleast_2d(new_gamma))
        new_omega_t = self._to_tensor(np.atleast_2d(new_omega))
        n_new = int(new_sensor_locs_t.shape[0])

        self._sensor_locs_t = torch.cat([self._sensor_locs_t, new_sensor_locs_t], dim=0)
        new_locs_norm = self._normalize(new_sensor_locs_t)
        self._locs_norm_t = torch.cat([self._locs_norm_t, new_locs_norm], dim=0)
        self._Gamma_t = torch.cat([self._Gamma_t, new_gamma_t], dim=0)
        self._Omega_t = torch.cat([self._Omega_t, new_omega_t], dim=0)

        new_dists = torch.cdist(self._grid_norm_t, new_locs_norm)
        new_weights = self._epanechnikov_kernel(new_dists)
        self._Weights_raw_t = torch.cat([self._Weights_raw_t, new_weights], dim=1)

        M_total = int(self._sensor_locs_t.shape[0])
        N_grid = self.N1 * self.N2
        K = int(self._Gamma_t.shape[1])

        affected_grids = self._get_affected_grids(new_locs_norm)
        I_flat = self._I_mask_bool_t.reshape(-1)

        if debugFlag:
            print(f"Sequential GPU update: +{n_new} sensors (total {M_total}), affected {len(affected_grids)}/{N_grid} grids")

        if M_total < self.dim_poly:
            if debugFlag:
                print(f"  Too few sensors ({M_total} < {self.dim_poly}), skipping GPU update.")
            self._sync_public_state()
            self._sync_public_cache()
            return self

        if self._Phi_t.shape[1] != K:
            self._Phi_t = torch.ones((self.R, K), dtype=self.dtype, device=self.device)
            self._Phi_t = self._Phi_t * (K / torch.sum(self._Phi_t, dim=1, keepdim=True))

        use_truncated = min(self.N1, self.N2) > 15
        svt_max_rank = min(10, min(self.N1, self.N2) - 1) if use_truncated else None

        for iteration in range(n_outer_iter):
            Sr_old = self._Sr_t.clone()

            self._update_theta(
                self._locs_norm_t,
                self._grid_norm_t,
                self._Gamma_t,
                self._Omega_t,
                self._Weights_raw_t,
                I_flat,
                grid_indices=affected_grids,
                debugFlag=debugFlag,
            )
            self._update_phi_vectorized(
                self._locs_norm_t,
                self._grid_norm_t,
                self._Gamma_t,
                self._Omega_t,
                self._Weights_raw_t,
                I_flat,
                debugFlag=debugFlag,
            )
            self._update_sr(
                self._I_mask_t,
                max_svt_iter=max_svt_iter,
                use_truncated_svd=use_truncated,
                max_rank=svt_max_rank,
                debugFlag=debugFlag,
            )

            diff = torch.linalg.norm(self._Sr_t - Sr_old) / (torch.linalg.norm(Sr_old) + 1e-9)
            diff_val = float(diff)
            if debugFlag:
                print(f"  GPU Iter {iteration + 1}/{n_outer_iter}, Rel Change: {diff_val:.6f}")
            if diff_val < self.tol and diff_val > 0:
                break

        self._H_hat_t = torch.einsum("rxy,rk->xyk", self._Sr_t, self._Phi_t)
        self._sync_public_state()
        self._sync_public_cache()
        return self

    # =================================================================
    # Evaluation / accessors
    # =================================================================

    def evaluate_reconstruction2(self, S_est, P_est, S_true, P_true, drawFlag=False):
        import matplotlib.pyplot as plt

        S_est = self._to_numpy(S_est)
        P_est = self._to_numpy(P_est)
        S_true = self._to_numpy(S_true)
        P_true = self._to_numpy(P_true)

        R, N1, N2 = self.R, self.N1, self.N2
        K = P_true.shape[1]

        Map_est = np.einsum("rxy,rk->xyk", S_est, P_est)
        Map_true = np.einsum("rxy,rk->xyk", S_true, P_true)

        error_norm = np.linalg.norm(Map_true - Map_est) ** 2
        true_norm = np.linalg.norm(Map_true) ** 2
        nmse = error_norm / (true_norm + 1e-9)
        self.nmse_list.append(nmse)

        Energy_est = np.sum(Map_est, axis=2)
        Energy_true = np.sum(Map_true, axis=2)

        print(f"NMSE: {nmse:.4f}")
        if drawFlag:
            X, Y = np.meshgrid(np.arange(N2), np.arange(N1))
            fig = plt.figure(figsize=(14, 6))

            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            surf1 = ax1.plot_surface(X, Y, Energy_true, cmap="viridis", linewidth=0, antialiased=False)
            ax1.set_title("Ground Truth Energy Map")
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            surf2 = ax2.plot_surface(X, Y, Energy_est, cmap="viridis", linewidth=0, antialiased=False)
            ax2.set_title("Estimated Energy Map")
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

            plt.tight_layout()
            plt.show()

        return nmse

    def get_current_map(self):
        self._sync_public_state()
        return self.H_hat.copy()

    def get_source_maps(self):
        self._sync_public_state()
        return self.Sr.copy()

    def get_spectra(self):
        self._sync_public_state()
        return self.Phi.copy()


II_BTD_Optimized_GPU = II_BTD_Opt_GPU
