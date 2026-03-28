import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, json

# ============================================================
#  1. Simulation Configuration (Section V parameters)
# ============================================================
class SimConfig:
    def __init__(self, **kwargs):
        # ---------- Area ----------
        self.L = 51                 # side length of the square area (meters)

        # ---------- Grid ----------
        self.N1 = 51                # grid rows
        self.N2 = 51                # grid columns

        # ---------- Sources ----------
        self.R = 2                  # number of signal sources
        self.Pr = 1.0               # transmit power per source (Watt)
        self.C0 = 2.0               # Friis reference-distance parameter
        self.h = 8.6                 # antenna height to avoid d=0 singularity

        # ---------- Frequency ----------
        self.K = 30                 # number of frequency bands

        # ---------- Shadowing ----------
        self.sigma_s = 3.0          # shadowing std-dev (dB)
        self.correlation_distance = 45.0              # correlation distance (meters)

        # ---------- SNR ----------
        self.SNR_dB = 20            # signal-to-noise ratio (dB)

        # ---------- Sampling ----------
        self.rho = 0.05             # sampling ratio (fraction of N1*N2)
        self.M = None               # number of sensors (computed from rho)
        self.n_grid_samples = 10     # number of additional grid points to add to sensors

        # ---------- Spectrum observation ----------
        self.full_obs = True        # True => each sensor observes all K bands
        self.K_obs = None           # bands per sensor when full_obs=False

        # ---------- Algorithm-related ----------
        self.M0 = 14                # min sensors inside kernel window
        self.grid_ratio = 0.8       # fraction of grids selected for index set I

        # ---------- Power spectrum generation ----------
        self.n_sinc = 2             # number of sinc^2 components per source
        self.ratio = 0.5
        # Override defaults with keyword arguments
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f"Unknown parameter: {key}")

        # Compute derived quantities
        if self.M is None:
            self.M = int(self.rho * self.N1 * self.N2)
        if self.K_obs is None:
            self.K_obs = int(self.K * self.ratio)   # default sparse = K/2

    def summary(self):
        lines = [
            "=" * 60,
            " Simulation Configuration (Section V)",
            "=" * 60,
            f"  Area           : {self.L} x {self.L} m",
            f"  Grid           : {self.N1} x {self.N2}  ({self.N1*self.N2} cells)",
            f"  Sources (R)    : {self.R}",
            f"  Freq bands (K) : {self.K}",
            f"  Sensors (M)    : {self.M} total = {max(0, self.M - self.n_grid_samples)} random + {self.n_grid_samples} grid samples",
            f"  Shadowing      : sigma_s={self.sigma_s} dB, correlation_distance={self.correlation_distance} m",
            f"  Observation    : {'Full' if self.full_obs else 'Sparse (K_obs='+str(self.K_obs)+') ratio='+str(self.ratio*100)+'%'}",
            f"  Friis params   : Pr={self.Pr}, C0={self.C0}, h={self.h}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================
#  2. Grid & Location Utilities
# ============================================================
def generate_grid(cfg):
    """
    Discretize the target area D into N1 x N2 grid cells.
    Returns the centre coordinates of each cell.

    Returns
    -------
    grid_points : ndarray (N1*N2, 2)
    gx, gy      : ndarray (N1, N2) meshgrid arrays
    """
    x = np.linspace(0, cfg.L - 1, cfg.N1)
    y = np.linspace(0, cfg.L - 1, cfg.N2)
    gx, gy = np.meshgrid(x, y, indexing='ij')          # (N1, N2)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    return grid_points, gx, gy


def generate_source_locations(cfg, rng):
    """
    Place R sources uniformly at random inside the area, with a 10% margin
    from the boundary to avoid pathological edge cases.
    """
    margin = cfg.L * 0.1
    locs = rng.uniform(margin, cfg.L - 1 - margin, size=(cfg.R, 2))
    return locs


def generate_sensor_locations(cfg, rng, n_sensors=None):
    """
    Place sensors uniformly at random in the L x L continuous space.
    Measurements are therefore *off-grid* in general (Section V, para. 1).
    
    Parameters:
    -----------
    n_sensors : int, optional
        Number of sensors to generate. If None, use cfg.M - cfg.n_grid_samples.
    """
    if n_sensors is None:
        n_sensors = max(0, cfg.M - cfg.n_grid_samples)
    locs = rng.uniform(0, cfg.L - 1, size=(n_sensors, 2))
    return locs


# ============================================================
#  3. Path Gain  -- Friis Transmission Equation (Section V)
# ============================================================
def compute_path_gain(source, locations, Pr=1.0, C0=2.0, h=0.1):
    """
    Friis model: g_r(d) = Pr * (C0 / d)^2   (Eq. in Section V)

    Computed in *dB* so that shadowing (also in dB) can be added directly,
    consistent with the standard large-scale propagation convention.

    Parameters
    ----------
    source    : (2,)  source coordinate
    locations : (N,2) evaluation points
    Pr, C0, h : Friis parameters

    Returns
    -------
    g_dB : (N,) path gain values in dB
    """
    diff = locations - source[np.newaxis, :]
    d = np.sqrt(np.sum(diff ** 2, axis=1) + h ** 2)
    g_linear = Pr * (C0 / d) ** 2
    g_dB = 10.0 * np.log10(np.maximum(g_linear, 1e-30))
    return g_dB


# ============================================================
#  4. Correlated Shadowing  (Gaussian Process, Section V)
# ============================================================
def generate_shadowing(locations, sigma_s, dc, rng):
    """
    Generate spatially correlated log-normal shadowing.

    "The shadowing component in log-scale log10 zeta is modelled using a
     Gaussian process with zero mean and auto-correlation function
     E{ log10 zeta(zi) log10 zeta(zj) } = sigma_s^2 exp(-||zi-zj||/dc)"
                                                     -- Section V

    In practice, this means the *dB-domain* shadowing X(z) satisfies
       Cov(X(zi), X(zj)) = sigma_s^2 * exp(-||zi-zj|| / dc)
    and we sample from this GP via Cholesky decomposition.

    Returns
    -------
    shadow_dB : (N,) shadowing realization in dB
    """
    n = len(locations)
    dist_mat = cdist(locations, locations)
    K_cov = sigma_s ** 2 * np.exp(-dist_mat / dc)
    K_cov += 1e-8 * np.eye(n)                          # numerical stability
    L_chol = cholesky(K_cov, lower=True)
    z = rng.standard_normal(n)
    return L_chol @ z


# ============================================================
#  5. Power Spectrum Synthesis (Section V)
# ============================================================
def generate_power_spectrum(K, n_sinc=2, rng=None):
    """
    phi_k^(r) = sum_{i=1}^{n_sinc} a_i * sinc^2( (k - f_i) / b_i )

    with  a_i ~ U(0.5, 2),  f_i in {1,...,K},  b_i ~ U(2, 4).
    The result is normalised so that  sum_k phi_k = K
    (Section II-A: "the total power sums to K").

    Returns
    -------
    phi : (K,) power spectrum vector
    """
    if rng is None:
        rng = np.random.default_rng()

    k_vals = np.arange(1, K + 1, dtype=float)
    phi = np.zeros(K)

    for _ in range(n_sinc):
        a = rng.uniform(0.5, 2.0)
        f = rng.integers(1, K + 1)              # centre frequency index
        b = rng.uniform(2.0, 4.0)
        phi += a * np.sinc((k_vals - f) / b) ** 2   # numpy sinc(x)=sin(pi*x)/(pi*x)

    # Normalise: sum phi_k = K
    phi = phi * K / np.sum(phi)
    return phi


# ============================================================
#  6. Propagation Field Assembly
# ============================================================
def build_propagation_fields(cfg, grid_pts, sensor_locs, source_locs, rng, addShadow=True):
    """
    For each source r, compute the large-scale propagation field
        rho^(r)(z) = g_r( d(s_r, z) )  +  zeta_r(z)        (Eq. 3)
    at every grid point and every sensor location.

    To ensure consistent spatial correlation between grid and sensor
    locations, we generate the shadowing GP over all locations jointly.

    Returns
    -------
    prop_grid   : list of R arrays, each (N1*N2,) – field at grid centres 
    prop_sensor : list of R arrays, each (M,)     – field at sensor locations 
    """
    N_grid = len(grid_pts)
    all_locs = np.vstack([grid_pts, sensor_locs])       # (N_grid+M, 2)

    prop_grid, prop_sensor = [], []
    for r in range(cfg.R):
        g_all_dB = compute_path_gain(source_locs[r], all_locs,
                                  cfg.Pr, cfg.C0, cfg.h)
        g_all_linear = 10.0 ** (g_all_dB / 10.0)
        if addShadow:
            shadow_all_dB = generate_shadowing(all_locs, cfg.sigma_s, cfg.correlation_distance, rng)
            shadow_all_linear = 10.0 ** (shadow_all_dB / 10.0)
            rho_all = g_all_linear * shadow_all_linear
        else:
            rho_all = g_all_linear
        
        prop_grid.append(rho_all[:N_grid])
        prop_sensor.append(rho_all[N_grid:])

    return prop_grid, prop_sensor


# ============================================================
#  7. Ground-Truth Tensor H  (Section II-B, Eq. 5)
# ============================================================
def construct_tensor(prop_grid_mW, Phi, cfg):
    """
    H = sum_r  S_r  ◦  phi^(r)         (Eq. 5, BTD model)

    where  [S_r]_{i,j} = rho^(r)(c_{ij})  and  ◦ denotes outer product.

    Returns
    -------
    H      : (N1, N2, K) ground-truth tensor
    S_list : list of R matrices each (N1, N2)
    """
    H_mW = np.zeros((cfg.N1, cfg.N2, cfg.K))
    S_mW_list = []
    for r in range(cfg.R):
        Sr_mW = prop_grid_mW[r].reshape(cfg.N1, cfg.N2)
        S_mW_list.append(Sr_mW)
        H_mW += Sr_mW[:, :, np.newaxis] * Phi[r][np.newaxis, np.newaxis, :]
    return H_mW, S_mW_list

def rbf_kernel_freq(f1, f2, length_scale):
    """
    频率核函数
    """
    f1 = np.array(f1).reshape(-1,1)
    f2 = np.array(f2).reshape(-1,1)
    dist = cdist(f1, f2,'sqeuclidean')
    return np.exp(-dist / (2 * length_scale**2))

def rbf_kernel(X1, X2, length_scale):
    """
        SE 核  k(x,x') = exp(-||x - x'||^2 / (2 * l^2))
    """
    dist_matrix = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-dist_matrix / (2 * length_scale**2))

def sample_joint_shadow_fading( position, freq, sigma, l_space, l_freq, rng):
        """
        从GP中采样阴影衰落
        Parameters:
        -----------
        positions : (n_pos, 2), 空间位置
        frequencies : (n_freq,), 频率数组
        sigma : float, 标准差 (dB)
        l_space : float, 空间相关长度 (m)
        l_freq : float, 频率相关长度 (GHz)
        
        Returns:
        --------
        samples : (n_pos, n_freq), 阴影衰落样本 (dB)
    
        """
        n_pos = len(position)
        n_freq = len(freq)

        # 核矩阵 
        K_space = rbf_kernel(position, position, l_space)
        K_freq = rbf_kernel_freq(freq, freq, l_freq)

        K_space += 1e-6 * np.eye(K_space.shape[0])
        K_freq += 1e-6 * np.eye(K_freq.shape[0])

        # Cholesky 分解
        try:
            L_s = np.linalg.cholesky(K_space)
            L_f = np.linalg.cholesky(K_freq)
            
        except np.linalg.LinAlgError:
            print("[DEBUG] Cholesky decomposition failed, using eigenvalue decomposition")

        # 采样
        # 生成 n_pos 个独立的频率响应样本
        # z shape: (n_pos, n_freq)
        z = rng.standard_normal((n_pos, n_freq))
        
        # samples shape: (n_pos, n_freq)
        samples = sigma * L_s @ z @ L_f.T
        
        return samples  
# ============================================================
#  8. Measurement Generation  (Eq. 2 / 4)
# ============================================================
def generate_measurements(prop_sensor, Phi, cfg, rng, sensor_locs):
    """
    gamma_m^(k) = sum_r rho^(r)(z_m) phi_k^(r) + eps_tilde_k     (Eq. 4)

    where eps_tilde_k ~ N(0, sigma^2) with sigma^2 chosen so that
    SNR = E[|signal|^2] / sigma^2 = 10^(SNR_dB/10).        (Section V)

    Also generates the observation indicator matrix psi (Eq. 10):
      psi_{mk} = 1 if sensor m observes band k, else 0.

    Returns
    -------
    Gamma       : (M, K) raw measurements (with noise)
    Gamma_clean : (M, K) noise-free measurements
    psi         : (M, K) observation indicator
    sigma2      : float, noise variance
    """
    M, K, R = cfg.M, cfg.K, cfg.R

    # ---- clean signal ----
    Gamma_clean = np.zeros((M, K))
    for r in range(R):
        Gamma_clean += np.outer(prop_sensor[r], Phi[r])
    fading = np.zeros((M,K)) 
    for r in range(R):
        shadow = sample_joint_shadow_fading(
            sensor_locs, 
            np.linspace(2.4 - 0.1, 2.4 + 0.1, cfg.K), 
            0.5, 
            20, 
            0.01, rng) 
        # shadow = shadow - np.mean(shadow)
        shadow = 10.0 ** (shadow / 10.0)
        shadow /= np.mean(shadow)
        fading += (shadow - 1)* prop_sensor[r][:, np.newaxis] * Phi[r][np.newaxis, :]
        # fading += shadow * Phi[r][np.newaxis, :]
    # ---- noise variance from target SNR ----
    signal_power = np.mean((Gamma_clean + fading) ** 2)
    sigma2_eps = signal_power / (10.0 ** (cfg.SNR_dB / 10.0))

    # ---- additive Gaussian noise ----
    noise_meas = rng.standard_normal((M, K)) * np.sqrt(sigma2_eps)
    Gamma = np.maximum(Gamma_clean + fading + noise_meas, 1e-10)

    # ---- observation pattern psi ----
    Omega = np.ones((M, K), dtype=np.int32)
    if not cfg.full_obs:
        K_obs = cfg.K_obs
        Omega[:] = 0
        for m in range(M):
            bands = rng.choice(K, size=K_obs, replace=False)
            Omega[m, bands] = 1

    # Mask unobserved entries (set to 0 for downstream algorithms)
    Gamma_observed = Gamma * Omega

    return Gamma_observed, Gamma_clean, Omega, sigma2_eps


# ============================================================
#  9. Index Set I  (Section V, para. 5)
# ============================================================
def select_index_set(cfg, rng):
    """
    "The index set I is constructed through randomly and uniformly
     selecting among 80% of the grids, at the same time avoiding
     scenarios where an entire column or row of S_r is missing."
                                                     -- Section V

    Returns
    -------
    I_set : 1-D array of linear indices into the (N1, N2) grid
    I_mask: (N1, N2) boolean mask
    """
    N1, N2 = cfg.N1, cfg.N2
    n_total = N1 * N2
    n_select = int(cfg.grid_ratio * n_total)

    for _ in range(200):                            # retry if row/col missing
        indices = rng.choice(n_total, size=n_select, replace=False)
        mask = np.zeros(n_total, dtype=bool)
        mask[indices] = True
        mask_2d = mask.reshape(N1, N2)
        rows_ok = mask_2d.any(axis=1).all()
        cols_ok = mask_2d.any(axis=0).all()
        if rows_ok and cols_ok:
            break

    return np.sort(indices), mask_2d


# ============================================================
#  10. NMSE Metric (Section V)
# ============================================================
def nmse(H_hat, H_true):
    """Normalized Mean Squared Error: ||H_hat - H||_F^2 / ||H||_F^2"""
    return np.sum((H_hat - H_true) ** 2) / np.sum(H_true ** 2)


# ============================================================
#  11. Master Pipeline
# ============================================================
def generate_data(cfg=None, seed=42, addShadow=True):
    """
    Full data-generation pipeline reproducing the Section V setup.

    Parameters
    ----------
    cfg  : SimConfig (default: standard configuration)
    seed : int, random seed for reproducibility

    Returns
    -------
    data : dict with all generated quantities
    """
    if cfg is None:
        cfg = SimConfig()
    
    # 使用独立的 RNG 流，确保 GT 与 M 无关
    # 不同组件使用不同的子种子，保证各自独立
    rng_sources = np.random.default_rng(seed)        # 信号源位置 (GT)
    rng_sensors = np.random.default_rng(seed + 100)  # 传感器位置 (依赖 M)
    rng_phi = np.random.default_rng(seed + 200)      # 功率谱 (GT)
    rng_meas = np.random.default_rng(seed + 300)     # 测量噪声
    rng_grid = np.random.default_rng(seed + 400)     # 网格采样

    # --- Geometry ---
    grid_coords, gx, gy = generate_grid(cfg)
    source_locs = generate_source_locations(cfg, rng_sources)  # 使用独立 RNG
    sensor_locs_random = generate_sensor_locations(cfg, rng_sensors)  # 原随机传感器

    # --- Propagation fields (先生成网格阴影，作为 GT) ---
    prop_grid, prop_sensor_random = build_propagation_fields(
        cfg, grid_coords, sensor_locs_random, source_locs, rng_sources, addShadow)

    # --- 从网格中采样额外点加入传感器阵列 ---
    if cfg.n_grid_samples > 0:
        N_grid = len(grid_coords)
        # 随机选择网格点索引
        grid_sample_indices = rng_grid.choice(N_grid, size=cfg.n_grid_samples, replace=False)
        grid_sample_locs = grid_coords[grid_sample_indices]
        
        # 从 prop_grid 中提取这些点的传播场（保持空间相关性！）
        prop_sensor_grid = []
        for r in range(cfg.R):
            prop_sensor_grid.append(prop_grid[r][grid_sample_indices])
        
        # 合并传感器位置和传播场
        sensor_locs = np.vstack([sensor_locs_random, grid_sample_locs])
        prop_sensor = []
        for r in range(cfg.R):
            prop_sensor.append(np.concatenate([prop_sensor_random[r], prop_sensor_grid[r]]))
    else:
        sensor_locs = sensor_locs_random
        prop_sensor = prop_sensor_random

    # --- Power spectra ---
    Phi = np.zeros((cfg.R, cfg.K))
    for r in range(cfg.R):
        Phi[r] = generate_power_spectrum(cfg.K, n_sinc=cfg.n_sinc, rng=rng_phi)  # 使用独立 RNG

    # --- Ground-truth tensor ---
    H, S_list = construct_tensor(prop_grid, Phi, cfg)

    # --- Measurements ---
    Gamma_obs, Gamma_clean, Omega, sigma2 = generate_measurements(
        prop_sensor, Phi, cfg, rng_meas, sensor_locs)  # 使用独立 RNG

    # --- Index set I ---
    I_set, I_mask = select_index_set(cfg, rng_grid)  # 使用独立 RNG

    # --- Package ---
    data = dict(
        config        = cfg,
        grid_coords   = grid_coords,           # (N1*N2, 2)
        grid_x        = gx,                 # (N1, N2)
        grid_y        = gy,                 # (N1, N2)
        source_locs   = source_locs,        # (R, 2)
        sensor_locs   = sensor_locs,        # (M, 2)
        H             = H,                  # (N1, N2, K) tensor in mW
        S             = S_list,             # [S1, S2, …] each (N1, N2) in mW
        Phi           = Phi,                # (R, K) power spectrum matrix
        Gamma_obs     = Gamma_obs,          # (M, K) observed measurements in mW
        Gamma_clean   = Gamma_clean,        # (M, K) noise-free measurements in mW
        Omega           = Omega,                # (M, K) observation indicator
        sigma2_noise  = sigma2,             # noise variance
        I_set         = I_set,              # selected grid indices (linear)
        I_mask        = I_mask,             # (N1, N2) boolean mask
        prop_sensor   = prop_sensor,        # (M, K) propagation sensor
    )
    return data


# ============================================================
#  12. Visualization
# ============================================================
def plot_data_summary(data,title = None, save_dir=None):
    """Create a multi-panel figure summarizing the generated data."""
    cfg = data['config']
    R, K = cfg.R, cfg.K

    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # ---------- Row 1: Propagation fields S_r ----------
    for r in range(min(R, 3)):
        ax = fig.add_subplot(gs[0, r])
        im = ax.imshow(data['S'][r].T, origin='lower',
                        extent=[0, cfg.L-1, 0, cfg.L-1], cmap='jet')
        ax.plot(*data['source_locs'][r], 'w*', ms=14, mew=1.5,
                label=f'Source {r+1}')
        ax.set_title(f'Propagation Field $S_{r+1}$ (dB)', fontsize=11)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # sensor locations
    ax = fig.add_subplot(gs[0, min(R, 3)])
    ax.scatter(data['sensor_locs'][:, 0], data['sensor_locs'][:, 1],
               s=8, c='blue', alpha=0.6, label=f'Sensors (M={cfg.M})')
    for r in range(R):
        ax.plot(*data['source_locs'][r], '*', ms=14, mew=1.5,
                label=f'Source {r+1}')
    ax.set_xlim(0, cfg.L-1); ax.set_ylim(0, cfg.L-1)
    ax.set_title('Sensor & Source Layout', fontsize=11)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.legend(fontsize=8); ax.set_aspect('equal')

    # ---------- Row 2: Power spectra & tensor slices ----------
    # Power spectra
    ax_phi = fig.add_subplot(gs[1, 0])
    k_idx = np.arange(1, K + 1)
    for r in range(R):
        ax_phi.plot(k_idx, data['Phi'][r], 'o-', ms=4,
                    label=f"$\\phi^{{({r+1})}}$")
    ax_phi.set_xlabel('Frequency band $k$')
    ax_phi.set_ylabel('$\\phi_k$')
    ax_phi.set_title('Power Spectrum $\\Phi$', fontsize=11)
    ax_phi.legend(fontsize=9)
    ax_phi.grid(True, alpha=0.3)

    # Tensor slices
    bands_to_show = [0, K // 4, K // 2, 3 * K // 4]
    for idx, band in enumerate(bands_to_show[:3]):
        ax = fig.add_subplot(gs[1, idx + 1])
        im = ax.imshow(data['H'][:, :, band].T, origin='lower',
                        extent=[0, cfg.L-1, 0, cfg.L-1], cmap='jet')
        ax.set_title(f'$\\mathcal{{H}}$ slice  k={band+1}', fontsize=11)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # ---------- Row 3: Observation pattern & statistics ----------
    # Observation pattern psi
    ax_psi = fig.add_subplot(gs[2, 0])
    ax_psi.imshow(data['Omega'][:min(50, cfg.M), :], aspect='auto',
                  cmap='Greys', interpolation='nearest')
    ax_psi.set_xlabel('Frequency band $k$')
    ax_psi.set_ylabel('Sensor index $m$')
    ax_psi.set_title(f"Observation pattern $\\Omega$ (first {min(50,cfg.M)} sensors)",
                     fontsize=11)

    # Index set I
    ax_I = fig.add_subplot(gs[2, 1])
    ax_I.imshow(data['I_mask'].T, origin='lower',
                extent=[0, cfg.L-1, 0, cfg.L-1], cmap='Greys_r')
    ax_I.set_title(f'Index set $\\mathcal{{I}}$ ({cfg.grid_ratio*100:.0f}% grids)',
                   fontsize=11)
    ax_I.set_xlabel('x (m)'); ax_I.set_ylabel('y (m)')

    # Histogram of propagation values
    ax_hist = fig.add_subplot(gs[2, 2])
    for r in range(R):
        ax_hist.hist(data['S'][r].ravel(), bins=40, alpha=0.5,
                     label=f'$S_{r+1}$')
    ax_hist.set_xlabel('Propagation field value (dB)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Distribution of $S_r$ values', fontsize=11)
    ax_hist.legend(fontsize=9)

    # SVD of S_r (to verify low-rank property)
    ax_svd = fig.add_subplot(gs[2, 3])
    for r in range(R):
        U, s, Vt = np.linalg.svd(data['S'][r], full_matrices=False)
        s_norm = s / s[0]
        ax_svd.semilogy(np.arange(1, len(s_norm) + 1), s_norm,
                        'o-', ms=3, label=f'$S_{r+1}$')
    ax_svd.set_xlabel('Singular value index')
    ax_svd.set_ylabel('Normalised singular value')
    ax_svd.set_title('SVD of $S_r$ (low-rank check)', fontsize=11)
    ax_svd.legend(fontsize=9)
    ax_svd.grid(True, alpha=0.3)

    plt.suptitle(f"Data Generation Summary {title}", fontsize=14, y=0.98)

    if save_dir:
        path = os.path.join(save_dir, f"data_summary{title}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {path}")
    plt.close(fig)
    return fig


def plot_tensor_bands(data,title = None, save_dir=None):
    """
    Visualize the ground-truth power spectrum map H for representative
    frequency bands (similar to Fig. 5 in the paper).
    """
    cfg = data['config']
    K = cfg.K
    bands = [
        K //2 - 1, K // 2, K // 2 + 1,
        3 * K // 4 - 1, 3 * K // 4, 3 * K // 4 + 1
        ]
    n = len(bands)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    vmin = data['H'].min()
    vmax = data['H'].max()

    for i, band in enumerate(bands):
        im = axes[i].imshow(data['H'][:, :, band].T, origin='lower',
                            extent=[0, cfg.L-1, 0, cfg.L-1],
                            cmap='jet', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Band k={band+1}', fontsize=10)
        axes[i].set_xlabel('x (m)')
        if i == 0:
            axes[i].set_ylabel('y (m)')
    fig.colorbar(im, ax=axes, shrink=0.8, label='dBm')
    fig.suptitle(f"Ground Truth Power Spectrum Map {title}"+" $\\mathcal{H}$", fontsize=12)
    # plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, f"tensor_bands{title}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {path}")
    plt.close(fig)


# ============================================================
#  13. Statistics & Validation
# ============================================================
def print_data_statistics(data):
    """Print key statistics to verify correctness of the generated data."""
    cfg = data['config']
    H = data['H']
    Phi = data['Phi']

    print(cfg.summary())
    print()
    print("-" * 50)
    print(" Data Statistics")
    print("-" * 50)

    for r in range(cfg.R):
        Sr = data['S'][r]
        print(f"  Source {r+1}:")
        print(f"    Location           : ({data['source_locs'][r][0]:.2f}, "
              f"{data['source_locs'][r][1]:.2f})")
        print(f"    S_{r+1} range       : [{Sr.min():.2f}, {Sr.max():.2f}] dB")
        print(f"    S_{r+1} rank (tol=1e-3 of max SV) : "
              f"{np.linalg.matrix_rank(Sr, tol=np.linalg.svd(Sr, compute_uv=False)[0]*1e-3)}")
        print(f"    sum(phi^({r+1}))    : {Phi[r].sum():.4f}  (should be {cfg.K})")

    print(f"\n  Tensor H shape      : {H.shape}")
    print(f"  Tensor H range      : [{H.min():.2f}, {H.max():.2f}]")
    print(f"  Tensor H Frobenius  : {np.linalg.norm(H):.4f}")
    print(f"  Noise variance      : {data['sigma2_noise']:.6f}")
    actual_snr = np.mean(data['Gamma_clean'] ** 2) / data['sigma2_noise']
    print(f"  Actual SNR          : {10*np.log10(actual_snr):.2f} dB "
          f"(target {cfg.SNR_dB} dB)")
    obs_ratio = data['Omega'].sum() / data['Omega'].size
    print(f"  Observation ratio   : {obs_ratio*100:.1f}%")
    print(f"  Index set |I|       : {len(data['I_set'])} / {cfg.N1*cfg.N2}")
    print("-" * 50)


# ============================================================
#  14. Save Data to Disk
# ============================================================
def save_data(data, save_dir):
    """Save all arrays as .npz and config as JSON."""
    os.makedirs(save_dir, exist_ok=True)
    cfg = data['config']

    # Save arrays
    np.savez_compressed(
        os.path.join(save_dir, "spectrum_map_data.npz"),
        grid_points   = data['grid_points'],
        grid_x        = data['grid_x'],
        grid_y        = data['grid_y'],
        source_locs   = data['source_locs'],
        sensor_locs   = data['sensor_locs'],
        H             = data['H'],
        S_0           = data['S'][0],
        S_1           = data['S'][1] if cfg.R > 1 else np.array([]),
        Phi           = data['Phi'],
        Gamma_obs     = data['Gamma_obs'],
        Gamma_clean   = data['Gamma_clean'],
        psi           = data['psi'],
        sigma2_noise  = np.array([data['sigma2_noise']]),
        I_set         = data['I_set'],
        I_mask        = data['I_mask'],
    )
    print(f"[Saved] {os.path.join(save_dir, 'spectrum_map_data.npz')}")

    # Save config
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(cfg_dict, f, indent=2, default=str)
    print(f"[Saved] {os.path.join(save_dir, 'config.json')}")


# ============================================================
#  MAIN — Reproduce Section V Data Generation
# ============================================================
if __name__ == "__main__":
    OUT_DIR = "code/Test/outputs/spectrum_map_data"
    os.makedirs(OUT_DIR, exist_ok=True)

    # =====================================================
    #  Experiment 1: Full Observation  (Section V-A, Fig. 6)
    # =====================================================
    print("\n" + "=" * 60)
    print(" Experiment 1: Full Observation")
    print("=" * 60)
    cfg_full = SimConfig(full_obs=True, rho=0.05)
    data_full = generate_data(cfg_full, seed=42)
    print_data_statistics(data_full)
    plot_data_summary(data_full,title = "Full Observation", save_dir=OUT_DIR)
    plot_tensor_bands(data_full,title = "Full Observation", save_dir=OUT_DIR)
    # save_data(data_full, OUT_DIR)

    # =====================================================
    #  Experiment 2: Sparse Observation  (Section V-A, Fig. 6)
    # =====================================================
    print("\n" + "=" * 60)
    print(" Experiment 2: Sparse Observation (K_obs = K/2)")
    print("=" * 60)
    cfg_sparse = SimConfig(full_obs=False, K_obs=10, rho=0.05)
    data_sparse = generate_data(cfg_sparse, seed=42)
    print_data_statistics(data_sparse)
    plot_data_summary(data_sparse,title = "Sparse Observation", save_dir=OUT_DIR) 
    plot_tensor_bands(data_sparse,title = "Sparse Observation", save_dir=OUT_DIR)

    # =====================================================
    #  Experiment 3: Varying M  (Section V-A, Fig. 6)
    # =====================================================
    print("\n" + "=" * 60)
    print(" Experiment 3: Varying number of sensors M")
    print("=" * 60)
    M_values = [130, 156, 182, 208, 234, 260]        # rho = 5%–10%
    for M_val in M_values:
        cfg_m = SimConfig(M=M_val, full_obs=True)
        data_m = generate_data(cfg_m, seed=42)
        snr_actual = 10 * np.log10(
            np.mean(data_m['Gamma_clean'] ** 2) / data_m['sigma2_noise'])
        print(f"  M = {M_val:4d}  |  tensor ||H||_F = {np.linalg.norm(data_m['H']):.2f}"
              f"  |  SNR = {snr_actual:.1f} dB")

    # =====================================================
    #  Experiment 4: Varying Shadowing  (Section V-C, Fig. 8)
    # =====================================================
    print("\n" + "=" * 60)
    print(" Experiment 4: Varying shadowing parameters")
    print("=" * 60)
    for sigma_s in [1, 2, 3, 4, 5, 6]:
        cfg_sh = SimConfig(sigma_s=sigma_s, correlation_distance=30, rho=0.05)
        data_sh = generate_data(cfg_sh, seed=42)
        print(f"  sigma_s = {sigma_s}  |  S1 range = "
              f"[{data_sh['S'][0].min():.1f}, {data_sh['S'][0].max():.1f}] dB")
    for dc in [10, 20, 30, 40, 50]:
        cfg_dc = SimConfig(sigma_s=4, correlation_distance=dc, rho=0.05)
        data_dc = generate_data(cfg_dc, seed=42)
        print(f"  correlation_distance = {dc:2d} m  |  S1 range = "
              f"[{data_dc['S'][0].min():.1f}, {data_dc['S'][0].max():.1f}] dB")

    # =====================================================
    #  Experiment 5: Off-grid vs On-grid  (Section V-D, Fig. 9)
    # =====================================================
    print("\n" + "=" * 60)
    print(" Experiment 5: Off-grid vs On-grid (varying N)")
    print("=" * 60)
    for N_val in [10, 15, 20, 25, 30]:
        C1 = 2
        M_val = int(C1 * N_val * np.log2(N_val))
        cfg_n = SimConfig(N1=N_val, N2=N_val, K=20, M=M_val, full_obs=True)
        data_n = generate_data(cfg_n, seed=42)
        print(f"  N = {N_val:2d}  |  M = {M_val:4d}  |  "
              f"tensor shape = {data_n['H'].shape}")

    # =====================================================
    #  Experiment 6: Varying R  (Section V-E, Fig. 10)
    # =====================================================
    print("\n" + "=" * 60)
    print(" Experiment 6: Different number of sources R")
    print("=" * 60)
    for R_val in [2, 3]:
        cfg_r = SimConfig(R=R_val, rho=0.05)
        data_r = generate_data(cfg_r, seed=42)
        print(f"  R = {R_val}  |  tensor ||H||_F = "
              f"{np.linalg.norm(data_r['H']):.2f}")

    print("\n✅ All experiments completed. Data and figures saved to:")
    print(f"   {OUT_DIR}")
