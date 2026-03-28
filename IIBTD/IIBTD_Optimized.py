import numpy as np
from scipy.optimize import nnls
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import svds


class II_BTD_Optimized:
    """
    Integrated Interpolation & Block-Term Tensor Decomposition (II-BTD)
    优化版：面向 UAV 序贯采样的高性能实现
    
    相比原始 fit_2 的主要优化：
    ────────────────────────────────────────────────────────────
    Step 1 (Θ更新):
      - 预计算所有传感器的 PhiOmegaPhiT 和 PhiGamma（消除内层冗余计算）
      - 用分块填充代替 np.kron（消除 Kronecker 积调用）
      - 向量化多项式特征计算（批量处理所有网格点）
      - 增量模式下只更新受影响的网格点
    
    Step 2 (Φ更新):
      - 按频段 k 分解为 K 个只有 R 个变量的小 NNLS（替代巨大的全局 NNLS）
      - 消除三层 Python 循环和逐行 append
    
    Step 3 (Sr更新):
      - 截断 SVD 替代完整 SVD
      - 减少 SVT 内层迭代次数（warm start 更有效）
      - 自适应终止阈值
    
    序贯采样:
      - fit_incremental() 支持逐步添加测量点
      - 自动检测受影响的网格区域，只做局部更新
      - 支持缓存和增量更新核权重矩阵
    ────────────────────────────────────────────────────────────
    """

    def __init__(self, n_sources=2, grid_size=(20, 20),
                 mu=1.0, nu=1.0, max_iter=30, tol=1e-5,
                 kernel_bandwidth=1, warmstart=False):
        self.R = n_sources
        self.N1, self.N2 = grid_size
        self.mu = mu
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.h = kernel_bandwidth
        self.warmstart = warmstart
        self.dim_poly = 6  # [1, x, y, x^2, xy, yx, y^2]
        self.nmse_list = []
        self.GT_PHI = None

        # ---- 序贯采样用的持久化缓存 ----
        self._initialized = False
        self._sensor_locs = None
        self._Gamma = None
        self._Omega = None
        self._grid_coords = None
        self._bounds = None
        self._locs_norm = None
        self._grid_norm = None
        self._Weights_raw = None  # (N_grid, M)

    # =================================================================
    # 核函数与多项式特征
    # =================================================================

    def _epanechnikov_kernel(self, dists):
        """[Paper Eq. 7] Epanechnikov Kernel"""
        u = dists / self.h
        k = 0.75 * (1 - u ** 2)
        k[np.abs(u) > 1] = 0
        return k

    def _get_poly_features(self, coords_norm, center_norm):
        """
        [Paper Sec II.A] 构建局部多项式特征向量 p(x_m - c_ij)
        支持 center_norm 为单个点 (2,) 或多个点 (N, 2)
        """
        diff = coords_norm - center_norm
        dx, dy = diff[:, 0], diff[:, 1]
        M = coords_norm.shape[0]
        X = np.zeros((M, self.dim_poly))
        X[:, 0] = 1.0
        X[:, 1] = dx
        X[:, 2] = dy
        X[:, 3] = dx ** 2
        X[:, 4] = dx * dy
        X[:, 5] = dy ** 2
        return X

    def _svt_operator(self, Matrix, threshold):
        """[Paper Eq. 21] 奇异值阈值算子 (SVT) - 完整 SVD 版"""
        U, s, Vt = np.linalg.svd(Matrix, full_matrices=False)
        s_thresh = np.maximum(s - threshold, 0)
        rec = U @ np.diag(s_thresh) @ Vt
        return np.maximum(rec, 0), s, s_thresh

    def _svt_operator_truncated(self, Matrix, threshold, max_rank=None):
        """截断 SVD 版 SVT：只计算前 max_rank 个奇异值"""
        if max_rank is None:
            max_rank = min(10, min(Matrix.shape) - 1)
        max_rank = min(max_rank, min(Matrix.shape) - 1)
        if max_rank < 1:
            return np.maximum(Matrix, 0), np.array([0.0]), np.array([0.0])
        try:
            U, s, Vt = svds(Matrix, k=max_rank)
            # svds 返回的奇异值是升序的，翻转为降序
            idx = np.argsort(s)[::-1]
            s = s[idx]
            U = U[:, idx]
            Vt = Vt[idx, :]
        except Exception:
            # 回退到完整 SVD
            U, s, Vt = np.linalg.svd(Matrix, full_matrices=False)

        s_thresh = np.maximum(s - threshold, 0)
        # 只保留非零奇异值
        nz = s_thresh > 0
        if np.any(nz):
            rec = (U[:, nz] * s_thresh[nz]) @ Vt[nz, :]
        else:
            rec = np.zeros_like(Matrix)
        return np.maximum(rec, 0), s, s_thresh

    # =================================================================
    # 归一化工具
    # =================================================================

    def _make_normalizer(self, bounds):
        (min_x, max_x), (min_y, max_y) = bounds
        self._scale_x = (max_x - min_x) / 2.0
        self._scale_y = (max_y - min_y) / 2.0
        self._center_x = (max_x + min_x) / 2.0
        self._center_y = (max_y + min_y) / 2.0

    def _normalize(self, c):
        n = np.copy(c).astype(float)
        n[:, 0] = (c[:, 0] - self._center_x) / self._scale_x
        n[:, 1] = (c[:, 1] - self._center_y) / self._scale_y
        return n

    # =================================================================
    # 预计算：消除冗余计算
    # =================================================================

    def _precompute_phi_dependent(self, Omega, Gamma):
        """
        预计算依赖于 Phi 的量（每次 Phi 更新后调用一次）：
          PhiOmPhiT_all[m] = (Phi * Omega[m]) @ Phi.T   shape (M, R, R)
          PhiGam_all[m]    = Phi @ (Gamma[m] * Omega[m]) shape (M, R)
        """
        M = Omega.shape[0]
        R = self.R
        # 向量化计算: 对所有 M 个传感器同时计算
        # PhiOmPhiT_all[m, r1, r2] = sum_k Phi[r1,k] * Omega[m,k] * Phi[r2,k]
        # = (Phi * Omega[m]) @ Phi.T 对每个m
        # 使用 einsum: 'rk,mk,sk->mrs'
        self._PhiOmPhiT_all = np.einsum('rk,mk,sk->mrs', self.Phi, Omega, self.Phi)
        # PhiGam_all[m, r] = sum_k Phi[r,k] * Gamma[m,k] * Omega[m,k]
        self._PhiGam_all = np.einsum('rk,mk->mr', self.Phi, Gamma * Omega)

    # =================================================================
    # Step 1: Θ 更新 - 向量化版
    # =================================================================

    def _update_theta(self, locs_norm, grid_norm, Gamma, Omega,
                      Weights_raw, I_flat, grid_indices=None, debugFlag=False):
        """
        [Paper Eq. 15] 更新 Theta
        
        优化点：
        1. 预计算 PhiOmPhiT 和 PhiGam（避免在内层重复计算）
        2. 用分块赋值替代 np.kron
        3. grid_indices 支持只更新部分网格（增量模式）
        """
        N_grid = self.N1 * self.N2
        R = self.R
        d = self.dim_poly

        # 预计算 Phi 相关量
        self._precompute_phi_dependent(Omega, Gamma)
        PhiOmPhiT = self._PhiOmPhiT_all  # (M, R, R)
        PhiGam = self._PhiGam_all          # (M, R)

        # 正则化基础矩阵
        dR = d * R
        ATA_base = np.zeros((dR, dR))
        for r in range(R):
            ATA_base[r * d, r * d] = self.nu

        # 要更新的网格索引
        if grid_indices is None:
            grid_indices = np.arange(N_grid)

        skip_count = 0
        for idx in grid_indices:
            if not I_flat[idx]:
                skip_count += 1
                continue

            w_vec = Weights_raw[idx]
            active = w_vec > 1e-6
            if not np.any(active):
                skip_count += 1
                continue

            active_idx = np.where(active)[0]
            n_act = len(active_idx)
            w_act = w_vec[active_idx]

            # 多项式特征
            X_act = self._get_poly_features(locs_norm[active_idx], grid_norm[idx])
            # XXT_act: (n_act, d, d)
            XXT_act = np.einsum('ni,nj->nij', X_act, X_act)

            # ---- 构建 ATA: 分块填充，替代 np.kron ----
            ATA = ATA_base.copy()
            ATb = np.zeros(dR)

            # 正则项: nu * Sr_{i,j}
            i_g, j_g = idx // self.N2, idx % self.N2
            for r in range(R):
                ATb[r * d] += self.nu * self.Sr[r, i_g, j_g]

            # 数据项: 批量累加
            # ATA[r1*d:(r1+1)*d, r2*d:(r2+1)*d] += Σ_m w_m * PhiOmPhiT[m,r1,r2] * XXT[m]
            PhiOmPhiT_act = PhiOmPhiT[active_idx]  # (n_act, R, R)
            PhiGam_act = PhiGam[active_idx]          # (n_act, R)

            for r1 in range(R):
                for r2 in range(R):
                    # coeffs[m] = w_act[m] * PhiOmPhiT_act[m, r1, r2]
                    coeffs = w_act * PhiOmPhiT_act[:, r1, r2]
                    # 加权求和: Σ_m coeffs[m] * XXT_act[m]
                    ATA[r1 * d:(r1 + 1) * d, r2 * d:(r2 + 1) * d] += \
                        np.einsum('n,nij->ij', coeffs, XXT_act)

            # ATb 数据项
            for r in range(R):
                coeffs_b = w_act * PhiGam_act[:, r]
                ATb[r * d:(r + 1) * d] += np.einsum('n,ni->i', coeffs_b, X_act)

            # 求解
            ATA += np.eye(dR) * 1e-7
            try:
                theta_flat = np.linalg.solve(ATA, ATb)
            except np.linalg.LinAlgError:
                theta_flat = np.linalg.lstsq(ATA, ATb, rcond=1e-6)[0]

            self.Theta[idx] = theta_flat.reshape(R, d)

        if debugFlag:
            total = len(grid_indices)
            print(f"  Step1 done. Skipped {skip_count}/{total} grids.")

    # =================================================================
    # Step 2: Φ 更新 - 按频段分解版
    # =================================================================

    def _update_phi(self, locs_norm, grid_norm, Gamma, Omega,
                    Weights_raw, I_flat, debugFlag=False):
        """
        [Paper Eq. 17-18] 更新 Phi
        
        核心优化：将全局 NNLS 分解为 K 个独立的小 NNLS
        每个子问题只有 R 个变量，极快。
        
        对每个频段 k:
          min_{φ_{·,k} >= 0}  Σ_{idx∈I} Σ_{m∈active(idx)} 
              w_m * omega_{m,k} * (gamma_{m,k} - Σ_r Pred_{m,r} * phi_{r,k})^2
        """
        N_grid = self.N1 * self.N2
        R = self.R
        K = Gamma.shape[1]

        # 对每个频段 k 分别求解
        for k in range(K):
            A_rows_k = []
            b_vals_k = []

            for idx in range(N_grid):
                if not I_flat[idx]:
                    continue

                w_vec = Weights_raw[idx]
                active = w_vec > 1e-6
                if not np.any(active):
                    continue

                active_idx = np.where(active)[0]
                w_act = w_vec[active_idx]

                # 多项式特征
                X_act = self._get_poly_features(locs_norm[active_idx], grid_norm[idx])
                # Theta_ij: (R, d) -> (d, R)
                Theta_ij_mat = self.Theta[idx].T
                # Pred: (n_act, R) = X_act @ Theta_ij_mat
                Pred = X_act @ Theta_ij_mat

                for i_m in range(len(active_idx)):
                    m_global = active_idx[i_m]
                    if Omega[m_global, k] < 0.5:
                        continue
                    sqrt_w = np.sqrt(w_act[i_m])
                    # 该行: [sqrt_w * Pred[i_m, 0], ..., sqrt_w * Pred[i_m, R-1]]
                    A_rows_k.append(sqrt_w * Pred[i_m])  # (R,) 向量
                    b_vals_k.append(sqrt_w * Gamma[m_global, k])

            if len(b_vals_k) > 0:
                A_k = np.array(A_rows_k)   # (n_k, R) — R 很小!
                b_k = np.array(b_vals_k)   # (n_k,)
                phi_k, _ = nnls(A_k, b_k)
                self.Phi[:, k] = phi_k

        # 防止 Phi 全为零
        for r in range(R):
            if np.max(self.Phi[r]) < 1e-10:
                self.Phi[r] = np.abs(np.random.randn(K)) * 0.01

        if debugFlag:
            print(f"  Step2 done. Phi range: [{self.Phi.min():.4f}, {self.Phi.max():.4f}]")

    def _update_phi_vectorized(self, locs_norm, grid_norm, Gamma, Omega,
                               Weights_raw, I_flat, debugFlag=False):
        """
        Φ 更新的进一步向量化版本：
        预先收集所有 (idx, m) 对的 Pred 和权重，避免在 k 循环中重复计算。
        """
        N_grid = self.N1 * self.N2
        R = self.R
        K = Gamma.shape[1]
        M = Gamma.shape[0]

        # ---- 第一步：收集所有 (idx, m) 对的 Pred 和 sqrt_w ----
        # 预分配列表
        all_pred = []     # 每项: (R,)
        all_sqrt_w = []   # 每项: scalar
        all_m_global = [] # 传感器全局索引

        for idx in range(N_grid):
            if not I_flat[idx]:
                continue
            w_vec = Weights_raw[idx]
            active = w_vec > 1e-6
            if not np.any(active):
                continue

            active_idx = np.where(active)[0]
            w_act = w_vec[active_idx]
            X_act = self._get_poly_features(locs_norm[active_idx], grid_norm[idx])
            Theta_ij_mat = self.Theta[idx].T  # (d, R)
            Pred = X_act @ Theta_ij_mat       # (n_act, R)

            for i_m in range(len(active_idx)):
                all_pred.append(Pred[i_m])
                all_sqrt_w.append(np.sqrt(w_act[i_m]))
                all_m_global.append(active_idx[i_m])

        if len(all_pred) == 0:
            return

        all_pred = np.array(all_pred)         # (N_pairs, R)
        all_sqrt_w = np.array(all_sqrt_w)     # (N_pairs,)
        all_m_global = np.array(all_m_global) # (N_pairs,)

        # ---- 第二步：按频段 k 分别求解 NNLS ----
        for k in range(K):
            # 筛选出 Omega[m, k] == 1 的行
            mask_k = Omega[all_m_global, k] > 0.5
            if not np.any(mask_k):
                self.Phi[:, k] = 0.0
                continue

            pred_k = all_pred[mask_k]         # (n_k, R)
            sw_k = all_sqrt_w[mask_k]         # (n_k,)
            m_k = all_m_global[mask_k]        # (n_k,)

            A_k = pred_k * sw_k[:, np.newaxis]        # (n_k, R)
            b_k = sw_k * Gamma[m_k, k]                # (n_k,)

            phi_k, _ = nnls(A_k, b_k)
            self.Phi[:, k] = phi_k

        # 防止 Phi 全为零
        for r in range(R):
            if np.max(self.Phi[r]) < 1e-10:
                self.Phi[r] = np.abs(np.random.randn(K)) * 0.01

        if debugFlag:
            print(f"  Step2 done. Phi range: [{self.Phi.min():.4f}, {self.Phi.max():.4f}]")

    # =================================================================
    # Step 3: Sr 更新 - 截断 SVD + 自适应迭代
    # =================================================================

    def _update_sr(self, I_mask, max_svt_iter=100, use_truncated_svd=True,
                   max_rank=None, debugFlag=False):
        """
        [Paper Eq. 20-21] 更新 Sr
        
        优化点:
        1. 使用截断 SVD（默认只计算前 max_rank 个奇异值）
        2. 热启动 + 自适应终止
        """
        for r in range(self.R):
            Psi = self.Theta[:, r, 0].reshape(self.N1, self.N2)

            if debugFlag:
                print(f"  Step3 Source {r}: Psi range [{Psi.min():.3f}, {Psi.max():.3f}]")

            Y = self.Sr[r].copy()
            delta_svt = 1.2
            psi_norm = np.linalg.norm(Psi) + 1e-9

            for t in range(max_svt_iter):
                if use_truncated_svd:
                    S_new, s_val, s_thresh = self._svt_operator_truncated(Y, self.mu, max_rank)
                else:
                    S_new, s_val, s_thresh = self._svt_operator(Y, self.mu)

                residual = I_mask * (Psi - S_new)
                res_norm = np.linalg.norm(residual)

                if res_norm < 1e-4 * psi_norm:
                    break

                Y = Y + delta_svt * residual

            self.Sr[r] = S_new

            if debugFlag:
                print(f"  Step3 Source {r}: Sr range [{self.Sr[r].min():.3f}, "
                      f"{self.Sr[r].max():.3f}], SVT iters={t + 1}")

    # =================================================================
    # 主入口：批量 fit（兼容原始接口）
    # =================================================================

    def fit_2(self, sensor_locs, Gamma, Omega, grid_coords, bounds,
              I_mask=None, debugFlag=False):
        """
        主训练循环：块坐标下降法 (Block Coordinate Descent)
        完全兼容原始 fit_2 的接口，内部使用优化实现。
        """
        M, K = Gamma.shape
        N_grid = self.N1 * self.N2

        # --- 0. 数据预处理 ---
        self._bounds = bounds
        self._make_normalizer(bounds)
        locs_norm = self._normalize(sensor_locs)
        grid_norm = self._normalize(grid_coords)

        if I_mask is None:
            I_mask = np.ones((self.N1, self.N2), dtype=bool)
        self.I_mask = I_mask
        I_flat = I_mask.ravel()

        # --- 初始化变量 ---
        if not self.warmstart or not hasattr(self, 'Phi') or not self._initialized:
            self.Theta = np.zeros((N_grid, self.R, self.dim_poly))
            self.Phi = np.ones((self.R, K))
            for ii in range(self.R):
                self.Phi[ii] = self.Phi[ii] * K / np.sum(self.Phi[ii])
            self.Sr = np.zeros((self.R, self.N1, self.N2))

        # 预计算核权重矩阵
        dists_full = cdist(grid_norm, locs_norm)
        Weights_raw = self._epanechnikov_kernel(dists_full)

        if debugFlag:
            print(f"Starting Optimization: N_grid={N_grid}, M={M}, K={K}, R={self.R}")
            print(f"  |I| = {np.sum(I_flat)}/{N_grid}, nu={self.nu}, mu={self.mu}")

        # 选择 SVT 模式
        use_truncated = min(self.N1, self.N2) > 15
        svt_max_rank = min(10, min(self.N1, self.N2) - 1) if use_truncated else None

        for iteration in range(self.max_iter):
            Sr_old = self.Sr.copy()

            # Step 1: Theta 更新
            self._update_theta(locs_norm, grid_norm, Gamma, Omega,
                               Weights_raw, I_flat, debugFlag=debugFlag)

            # Step 2: Phi 更新（按频段分解）
            self._update_phi_vectorized(locs_norm, grid_norm, Gamma, Omega,
                                        Weights_raw, I_flat, debugFlag=debugFlag)

            # Step 3: Sr 更新
            self._update_sr(I_mask, max_svt_iter=40,
                            use_truncated_svd=use_truncated,
                            max_rank=svt_max_rank, debugFlag=debugFlag)

            # 收敛检查
            diff = np.linalg.norm(self.Sr - Sr_old) / (np.linalg.norm(Sr_old) + 1e-9)

            if debugFlag:
                print(f"  Iter {iteration + 1}/{self.max_iter}, "
                      f"Relative Change: {diff:.6f}\n")

            if diff < self.tol and diff > 0:
                if debugFlag:
                    print(f"  Converged at iteration {iteration + 1}.")
                break
            elif diff == 0 and iteration > 0:
                if debugFlag:
                    print(f"  No change at iteration {iteration + 1}, stopping.")
                break

        # 构建最终张量
        self.H_hat = np.zeros((self.N1, self.N2, K))
        for r in range(self.R):
            self.H_hat += self.Sr[r][:, :, np.newaxis] * self.Phi[r][np.newaxis, np.newaxis, :]

        # 缓存状态（供序贯采样使用）
        self._initialized = True
        self._sensor_locs = sensor_locs.copy()
        self._Gamma = Gamma.copy()
        self._Omega = Omega.copy()
        self._grid_coords = grid_coords.copy()
        self._locs_norm = locs_norm
        self._grid_norm = grid_norm
        self._Weights_raw = Weights_raw

        return self

    # =================================================================
    # 序贯采样：增量式更新
    # =================================================================

    def _get_affected_grids(self, new_sensor_locs_norm):
        """
        检测受新传感器影响的网格点索引。
        新传感器的 Epanechnikov 核有紧支撑 (||u|| < h)，
        只有核权重 > 0 的网格点需要更新。
        
        Args:
            new_sensor_locs_norm: (n_new, 2) 归一化后的新传感器位置
        Returns:
            affected: 受影响的网格点索引数组
        """
        # 计算新传感器到所有网格点的距离
        dists = cdist(self._grid_norm, new_sensor_locs_norm)  # (N_grid, n_new)
        # 任一新传感器在核窗口内的网格点
        in_support = np.any(dists / self.h < 1.0, axis=1)
        return np.where(in_support)[0]

    def fit_incremental(self, new_sensor_locs, new_gamma, new_omega,
                        grid_coords=None, bounds=None, I_mask=None,
                        n_outer_iter=2, max_svt_iter=20, debugFlag=False):
        """
        UAV 序贯采样的增量式更新。
        
        当 UAV 飞到新位置采集到新的测量数据后调用此方法，
        只更新受影响的网格点，大幅减少计算时间。
        
        Args:
            new_sensor_locs: (n_new, 2) 新测量位置
            new_gamma:       (n_new, K) 新观测数据
            new_omega:       (n_new, K) 新观测掩码
            grid_coords:     首次调用时需提供网格坐标
            bounds:          首次调用时需提供边界
            I_mask:          索引集掩码
            n_outer_iter:    外层迭代次数（增量模式通常 1-3 即可）
            max_svt_iter:    SVT 最大内层迭代（增量模式减少）
            debugFlag:       是否打印调试信息
        
        Returns:
            self
        """
        new_sensor_locs = np.atleast_2d(new_sensor_locs)
        new_gamma = np.atleast_2d(new_gamma)
        new_omega = np.atleast_2d(new_omega)
        n_new = new_sensor_locs.shape[0]
        K = new_gamma.shape[1]

        # ---- 首次调用：需要完整初始化 ----
        if not self._initialized:
            if grid_coords is None or bounds is None:
                raise ValueError("首次调用 fit_incremental 必须提供 grid_coords 和 bounds")
            return self.fit_2(new_sensor_locs, new_gamma, new_omega,
                              grid_coords, bounds, I_mask, debugFlag)

        # ---- 增量更新 ----
        N_grid = self.N1 * self.N2

        # 1. 追加新数据到全局缓存
        self._sensor_locs = np.vstack([self._sensor_locs, new_sensor_locs])
        self._Gamma = np.vstack([self._Gamma, new_gamma])
        self._Omega = np.vstack([self._Omega, new_omega])
        M_total = self._sensor_locs.shape[0]

        # 2. 归一化新传感器坐标
        new_locs_norm = self._normalize(new_sensor_locs)
        self._locs_norm = np.vstack([self._locs_norm, new_locs_norm])

        # 3. 增量更新核权重矩阵（只计算新列）
        new_dists = cdist(self._grid_norm, new_locs_norm)  # (N_grid, n_new)
        new_weights = self._epanechnikov_kernel(new_dists)  # (N_grid, n_new)
        self._Weights_raw = np.hstack([self._Weights_raw, new_weights])

        # 4. 检测受影响的网格
        affected_grids = self._get_affected_grids(new_locs_norm)

        if I_mask is not None:
            self.I_mask = I_mask
        I_flat = self.I_mask.ravel()

        if debugFlag:
            print(f"Incremental update: {n_new} new sensors, "
                  f"M_total={M_total}, affected grids={len(affected_grids)}/{N_grid}")

        # 5. 确保 Phi 维度匹配
        if self.Phi.shape[1] != K:
            # K 发生变化（不常见）
            self.Phi = np.ones((self.R, K))

        # 选择 SVT 模式
        use_truncated = min(self.N1, self.N2) > 15
        svt_max_rank = min(10, min(self.N1, self.N2) - 1) if use_truncated else None

        # 6. 少量外层迭代
        for iteration in range(n_outer_iter):
            Sr_old = self.Sr.copy()

            # Step 1: 只更新受影响的网格点
            self._update_theta(self._locs_norm, self._grid_norm,
                               self._Gamma, self._Omega,
                               self._Weights_raw, I_flat,
                               grid_indices=affected_grids,
                               debugFlag=debugFlag)

            # Step 2: Phi 更新（全局，但用高效的按频段分解）
            self._update_phi_vectorized(self._locs_norm, self._grid_norm,
                                        self._Gamma, self._Omega,
                                        self._Weights_raw, I_flat,
                                        debugFlag=debugFlag)

            # Step 3: Sr 更新（warm start + 减少迭代）
            self._update_sr(self.I_mask, max_svt_iter=max_svt_iter,
                            use_truncated_svd=use_truncated,
                            max_rank=svt_max_rank, debugFlag=debugFlag)

            diff = np.linalg.norm(self.Sr - Sr_old) / (np.linalg.norm(Sr_old) + 1e-9)
            if debugFlag:
                print(f"  Incr iter {iteration + 1}/{n_outer_iter}, "
                      f"Relative Change: {diff:.6f}")

            if diff < self.tol and diff > 0:
                break

        # 7. 重建 H_hat
        self.H_hat = np.zeros((self.N1, self.N2, K))
        for r in range(self.R):
            self.H_hat += self.Sr[r][:, :, np.newaxis] * self.Phi[r][np.newaxis, np.newaxis, :]

        return self

    # =================================================================
    # 序贯采样的便捷方法
    # =================================================================

    def init_sequential(self, grid_coords, bounds, K, I_mask=None):
        """
        为序贯采样模式做初始化（不需要任何测量数据）。
        
        Args:
            grid_coords: (N1*N2, 2) 网格中心坐标
            bounds: ((min_x, max_x), (min_y, max_y))
            K: 频段数
            I_mask: (N1, N2) 索引集掩码
        """
        N_grid = self.N1 * self.N2

        self._bounds = bounds
        self._make_normalizer(bounds)
        self._grid_coords = grid_coords.copy()
        self._grid_norm = self._normalize(grid_coords)

        if I_mask is None:
            I_mask = np.ones((self.N1, self.N2), dtype=bool)
        self.I_mask = I_mask

        # 初始化模型变量
        self.Theta = np.zeros((N_grid, self.R, self.dim_poly))
        self.Phi = np.ones((self.R, K))
        for ii in range(self.R):
            self.Phi[ii] = self.Phi[ii] * K / np.sum(self.Phi[ii])
        self.Sr = np.zeros((self.R, self.N1, self.N2))
        self.H_hat = np.zeros((self.N1, self.N2, K))

        # 初始化空的传感器缓存
        self._sensor_locs = np.empty((0, 2))
        self._locs_norm = np.empty((0, 2))
        self._Gamma = np.empty((0, K))
        self._Omega = np.empty((0, K))
        self._Weights_raw = np.empty((N_grid, 0))

        self._initialized = True

    def add_measurements(self, new_sensor_locs, new_gamma, new_omega,
                         n_outer_iter=2, max_svt_iter=20, debugFlag=False):
        """
        序贯采样的便捷接口：添加新测量并增量更新。
        需要先调用 init_sequential() 完成初始化。
        
        Args:
            new_sensor_locs: (n_new, 2) 新测量位置
            new_gamma:       (n_new, K) 新观测数据
            new_omega:       (n_new, K) 新观测掩码
            n_outer_iter:    外层迭代次数
            max_svt_iter:    SVT 最大迭代
            debugFlag:       是否打印调试信息
        
        Returns:
            self
        """
        if not self._initialized:
            raise RuntimeError("请先调用 init_sequential() 完成初始化")

        new_sensor_locs = np.atleast_2d(new_sensor_locs)
        new_gamma = np.atleast_2d(new_gamma)
        new_omega = np.atleast_2d(new_omega)
        n_new = new_sensor_locs.shape[0]

        # 追加数据
        self._sensor_locs = np.vstack([self._sensor_locs, new_sensor_locs])
        new_locs_norm = self._normalize(new_sensor_locs)
        self._locs_norm = np.vstack([self._locs_norm, new_locs_norm])
        self._Gamma = np.vstack([self._Gamma, new_gamma])
        self._Omega = np.vstack([self._Omega, new_omega])

        # 增量更新核权重
        new_dists = cdist(self._grid_norm, new_locs_norm)
        new_weights = self._epanechnikov_kernel(new_dists)
        self._Weights_raw = np.hstack([self._Weights_raw, new_weights])

        M_total = self._sensor_locs.shape[0]
        N_grid = self.N1 * self.N2
        K = self._Gamma.shape[1]

        # 检测受影响的网格
        affected_grids = self._get_affected_grids(new_locs_norm)
        I_flat = self.I_mask.ravel()

        if debugFlag:
            print(f"Sequential update: +{n_new} sensors (total {M_total}), "
                  f"affected {len(affected_grids)}/{N_grid} grids")

        # 如果传感器太少，至少需要 dim_poly 个测量才有意义
        if M_total < self.dim_poly:
            if debugFlag:
                print(f"  Too few sensors ({M_total} < {self.dim_poly}), skipping update.")
            return self

        use_truncated = min(self.N1, self.N2) > 15
        svt_max_rank = min(10, min(self.N1, self.N2) - 1) if use_truncated else None

        for iteration in range(n_outer_iter):
            Sr_old = self.Sr.copy()

            # Step 1: 局部 Theta 更新
            self._update_theta(self._locs_norm, self._grid_norm,
                               self._Gamma, self._Omega,
                               self._Weights_raw, I_flat,
                               grid_indices=affected_grids,
                               debugFlag=debugFlag)

            # Step 2: Phi 全局更新（按频段分解，仍然很快）
            self._update_phi_vectorized(self._locs_norm, self._grid_norm,
                                        self._Gamma, self._Omega,
                                        self._Weights_raw, I_flat,
                                        debugFlag=debugFlag)

            # Step 3: Sr 更新
            self._update_sr(self.I_mask, max_svt_iter=max_svt_iter,
                            use_truncated_svd=use_truncated,
                            max_rank=svt_max_rank, debugFlag=debugFlag)

            diff = np.linalg.norm(self.Sr - Sr_old) / (np.linalg.norm(Sr_old) + 1e-9)
            if debugFlag:
                print(f"  Iter {iteration + 1}/{n_outer_iter}, Rel Change: {diff:.6f}")
            if diff < self.tol and diff > 0:
                break

        # 重建 H_hat
        self.H_hat = np.zeros((self.N1, self.N2, K))
        for r in range(self.R):
            self.H_hat += self.Sr[r][:, :, np.newaxis] * self.Phi[r][np.newaxis, np.newaxis, :]

        return self

    # =================================================================
    # 评估重建质量
    # =================================================================

    def evaluate_reconstruction2(self, S_est, P_est, S_true, P_true, drawFlag=False):
        """
        评估重建质量：计算 NMSE 并可选绘制对比图。
        兼容 II_BTD_Annotated 的同名方法。
        """
        import matplotlib.pyplot as plt

        R, N1, N2 = self.R, self.N1, self.N2
        K = P_true.shape[1]

        Map_est = np.einsum('rxy, rk->xyk', S_est, P_est)
        Map_true = np.einsum('rxy, rk->xyk', S_true, P_true)

        error_norm = np.linalg.norm(Map_true - Map_est)**2
        true_norm = np.linalg.norm(Map_true)**2
        nmse = error_norm / (true_norm + 1e-9)
        self.nmse_list.append(nmse)

        Energy_est = np.sum(Map_est, axis=2)
        Energy_true = np.sum(Map_true, axis=2)

        print(f"NMSE: {nmse:.4f}")
        if drawFlag:
            X, Y = np.meshgrid(np.arange(N2), np.arange(N1))
            fig = plt.figure(figsize=(14, 6))

            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            surf1 = ax1.plot_surface(X, Y, Energy_true, cmap='viridis', linewidth=0, antialiased=False)
            ax1.set_title('Ground Truth Energy Map')
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            surf2 = ax2.plot_surface(X, Y, Energy_est, cmap='viridis', linewidth=0, antialiased=False)
            ax2.set_title('Estimated Energy Map')
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

            plt.tight_layout()
            plt.show()

        return nmse

    # =================================================================
    # 便捷访问方法
    # =================================================================

    def get_current_map(self):
        """获取当前重建的功率谱图"""
        return self.H_hat.copy()

    def get_source_maps(self):
        """获取各信号源的传播场"""
        return self.Sr.copy()

    def get_spectra(self):
        """获取各信号源的频谱"""
        return self.Phi.copy()