import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ---- 路径 ----
_script_dir = os.path.dirname(os.path.abspath(__file__))
_rt_dir = os.path.join(_script_dir, '..', 'RT')
_root_dir = os.path.join(_script_dir, '..')

for p in [_rt_dir, _root_dir, _script_dir]:
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from RT.rtSpectrumGen import SionnaSimConfig, generate_data_rt
from IIBTD.IIBTD_Optimized import II_BTD_Optimized

# ---- 绘图风格 ----
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
SAVE_DIR = os.path.join(_script_dir, 'outputs', 'rt_results')
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
#  UAV 轨迹排序
# ============================================================
def sort_sensors_by_trajectory(locs, trajectory_type='tsp'):
    """
    将随机传感器位置按 UAV 飞行轨迹排序。

    Parameters
    ----------
    locs : (M, 2)
    trajectory_type : 'tsp' | 'snake' | 'angle'

    Returns
    -------
    sorted_locs : (M, 2)
    order : (M,) 排序索引
    """
    M = len(locs)

    if trajectory_type == 'tsp':
        visited = np.zeros(M, dtype=bool)
        order = [0]
        visited[0] = True
        for _ in range(M - 1):
            current_loc = locs[order[-1]]
            dists = np.linalg.norm(locs - current_loc, axis=1)
            dists[visited] = np.inf
            nxt = np.argmin(dists)
            order.append(nxt)
            visited[nxt] = True
        order = np.array(order)

    elif trajectory_type == 'snake':
        y_sorted_idx = np.argsort(locs[:, 1])
        n_rows = int(np.sqrt(M))
        pts_per_row = M // n_rows + 1
        order = []
        for i in range(n_rows + 1):
            s, e = i * pts_per_row, min((i + 1) * pts_per_row, M)
            row = y_sorted_idx[s:e]
            row = row[np.argsort(locs[row, 0])]
            if i % 2 == 1:
                row = row[::-1]
            order.extend(row)
        order = np.array(order[:M])

    elif trajectory_type == 'angle':
        center = locs.mean(axis=0)
        diff = locs - center
        angles = np.arctan2(diff[:, 1], diff[:, 0])
        radii = np.linalg.norm(diff, axis=1)
        order = np.lexsort((radii, angles))

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    return locs[order], order


# ============================================================
#  轨迹可视化
# ============================================================
def plot_trajectory(sensor_locs, x_range, y_range, tx_locs=None,
                    title="UAV Flight Trajectory", save_path=None):
    """绘制传感器 UAV 采样轨迹"""
    fig, ax = plt.subplots(figsize=(8, 8))
    M = len(sensor_locs)

    ax.plot(sensor_locs[:, 0], sensor_locs[:, 1],
            'b-', alpha=0.5, linewidth=1.5, label='Flight Path')
    sc = ax.scatter(sensor_locs[:, 0], sensor_locs[:, 1],
                    c=np.arange(M), cmap='viridis',
                    s=60, edgecolors='k', linewidth=0.5, zorder=5)
    ax.scatter(*sensor_locs[0], c='lime', s=200, marker='^',
               edgecolors='k', linewidth=2, zorder=10, label='Start')
    ax.scatter(*sensor_locs[-1], c='red', s=200, marker='s',
               edgecolors='k', linewidth=2, zorder=10, label='End')

    if tx_locs is not None:
        for i, tx in enumerate(tx_locs):
            ax.scatter(tx[0], tx[1], c='yellow', s=300, marker='*',
                       edgecolors='k', linewidth=2, zorder=10,
                       label=f'TX {i}' if i == 0 else '')

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Sampling Order')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{title}\n(Total: {M} sampling points)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[Saved] {save_path}")
    plt.close(fig)


# ============================================================
#  RT 数据归一化
# ============================================================
def normalize_rt_data(data):
    """
    将 RT 生成的数据归一化到 II-BTD 算法可接受的数值范围。

    RT 的线性 path gain 极小 (1e-15 ~ 1e-7)，而 II-BTD 期望 S ∈ [1e-4, 0.1] 量级。
    我们用一个全局缩放因子使 S 的最大值归一化到 ~0.05 量级，
    然后同步缩放 Gamma（因为 Gamma ~ S * Phi，Phi 不变 → Gamma 也要乘同样的因子）。

    NMSE 是 scale-invariant 的，所以缩放不影响评估结果。

    Parameters
    ----------
    data : dict  — generate_data_rt() 的返回值

    Returns
    -------
    data : dict  — 就地修改并返回（包含 'scale_factor' 字段）
    """
    S_list = data['S']   # list of R arrays, each (N1, N2)
    S_max = max(np.max(np.abs(s)) for s in S_list)
    if S_max < 1e-20:
        print("  [WARNING] S_max ≈ 0, 无法归一化")
        data['scale_factor'] = 1.0
        return data

    # 目标: S_max → 0.05
    target_max = 0.05
    scale = target_max / S_max

    print(f"  [Normalize] S_max={S_max:.2e}, scale_factor={scale:.2e}")

    # 缩放 S (grid propagation fields)
    for i in range(len(S_list)):
        S_list[i] = S_list[i] * scale
    data['S'] = S_list

    # 缩放 prop_grid
    if 'prop_grid' in data:
        data['prop_grid'] = [pg * scale for pg in data['prop_grid']]

    # 缩放 prop_sensor
    if 'prop_sensor' in data:
        data['prop_sensor'] = [ps * scale for ps in data['prop_sensor']]

    # 缩放 Gamma (测量值 = sum_r prop_sensor[r] * Phi[r] + noise)
    # Gamma_clean ~ S * Phi, 所以也要乘 scale
    data['Gamma_obs'] = data['Gamma_obs'] * scale
    if 'Gamma_clean' in data:
        data['Gamma_clean'] = data['Gamma_clean'] * scale

    # 缩放 sigma2 (噪声方差 ∝ signal_power ∝ scale^2)
    if 'sigma2_noise' in data:
        data['sigma2_noise'] = data['sigma2_noise'] * (scale ** 2)

    # 缩放 H (ground truth tensor)
    if 'H' in data:
        data['H'] = data['H'] * scale

    data['scale_factor'] = scale
    return data


# ============================================================
#  频率域观测模式
# ============================================================
def generate_omega_full(M, K, ratio=1.0, rng=None):
    """全频段观测"""
    return np.ones((M, K), dtype=np.int32)


def generate_omega_dual_center(M, K, ratio, rng=None):
    """
    以 K/4 和 3K/4 为中心的双中心频段观测模式。
    每个传感器观测以两个中心频段向外扩展的频段。
    """
    Omega = np.zeros((M, K), dtype=np.int32)
    n_obs = int(K * ratio)
    n_per_center = max(1, n_obs // 2)
    center1 = K // 4
    center2 = 3 * K // 4

    for m in range(M):
        observed = set()
        for offset in range(n_per_center):
            if center1 - offset >= 0:
                observed.add(center1 - offset)
            if center1 + offset < K:
                observed.add(center1 + offset)
            if len(observed) >= n_per_center:
                break
        for offset in range(n_per_center):
            if center2 - offset >= 0:
                observed.add(center2 - offset)
            if center2 + offset < K:
                observed.add(center2 + offset)
            if len(observed) >= n_obs:
                break
        for b in observed:
            Omega[m, b] = 1
    return Omega


def generate_omega_random(M, K, ratio, rng=None):
    """
    随机选择频段观测模式。
    每个传感器随机观测 ratio*K 个频段。
    """
    if rng is None:
        rng = np.random.default_rng()
    Omega = np.zeros((M, K), dtype=np.int32)
    n_obs = max(1, int(K * ratio))
    for m in range(M):
        bands = rng.choice(K, size=n_obs, replace=False)
        Omega[m, bands] = 1
    return Omega


def generate_omega_cyclic(M, K, ratio, overlap=2, start_offset=0, rng=None):
    """
    循环频谱切换模式：每个航点观测不同的频段块，循环覆盖整个频谱。
    B = int(K * ratio)，步长 = B - overlap。
    """
    B = max(1, int(K * ratio))
    overlap = max(0, min(overlap, B - 1))
    step = B - overlap
    n_steps = max(1, (K - overlap) // step)

    Omega = np.zeros((M, K), dtype=np.int32)
    for m in range(M):
        step_idx = (m + start_offset) % n_steps
        start_freq = step_idx * step
        end_freq = min(start_freq + B, K)
        Omega[m, start_freq:end_freq] = 1
    return Omega


OMEGA_GENERATORS = {
    'full':        generate_omega_full,
    'dual_center': generate_omega_dual_center,
    'random':      generate_omega_random,
    'cyclic':      generate_omega_cyclic,
}


# ============================================================
#  单场景序贯采样实验
# ============================================================
def run_rt_sequential_experiment(
    scene_name='simple_canyan_street',
    R=1, K=24, M_max=130, N1=51, N2=51,
    SNR_dB=20, seed=42,
    trajectory_type='tsp',
    step_size=10,
    use_pathsolver=False,
    save_prefix=None,
    omega_mode='full',
    omega_ratio=1.0,
):
    """
    在单个 Sionna RT 场景上运行 UAV 序贯采样 + II-BTD 重建实验。

    Parameters
    ----------
    scene_name : str  — Sionna 场景名
    R, K, M_max, N1, N2 : 仿真参数
    SNR_dB : 信噪比
    seed : 随机种子
    trajectory_type : UAV 轨迹排序方式
    step_size : 每步增加的传感器数
    use_pathsolver : 是否用 PathSolver 精确计算 sensor path gain
    save_prefix : 保存文件名前缀

    Returns
    -------
    M_values : list of int
    nmse_list : list of float
    data : dict  — 生成的完整数据（已归一化）
    """
    if save_prefix is None:
        save_prefix = scene_name

    print(f"\n{'='*60}")
    print(f"  场景: {scene_name}  |  R={R}, K={K}, M={M_max}")
    print(f"  SNR={SNR_dB} dB, Grid={N1}x{N2}")
    print(f"{'='*60}")

    # ---- 1. 生成 RT 数据 ----
    cfg = SionnaSimConfig(
        scene_name=scene_name,
        R=R, K=K, M=M_max,
        N1=N1, N2=N2,
        SNR_dB=SNR_dB,
        max_depth=5,
        samples_per_tx=10**6,
    )

    t0 = time.time()
    data = generate_data_rt(cfg, seed=seed,
                            use_pathsolver_for_sensors=use_pathsolver)
    t_data = time.time() - t0
    print(f"  数据生成耗时: {t_data:.1f}s")

    # ---- 2. 归一化 ----
    data = normalize_rt_data(data)

    # ---- 3. 准备 bounds ----
    x_range = data['x_range']
    y_range = data['y_range']
    bounds = (x_range, y_range)

    # ---- 4. 按轨迹排序 ----
    sorted_locs, order = sort_sensors_by_trajectory(
        data['sensor_locs'], trajectory_type=trajectory_type
    )
    Gamma_sorted = data['Gamma_obs'][order, :]

    # ---- 4b. 生成频率观测掩码 ----
    rng_omega = np.random.default_rng(seed + 2000)
    omega_gen = OMEGA_GENERATORS.get(omega_mode, generate_omega_full)
    Omega_custom = omega_gen(M_max, K, omega_ratio, rng=rng_omega)
    # 应用掩码到 Gamma
    Gamma_sorted = Gamma_sorted * Omega_custom[order, :]
    Omega_sorted = Omega_custom[order, :]
    obs_ratio_actual = Omega_sorted.mean()
    print(f"  [Omega] mode={omega_mode}, ratio={omega_ratio:.2f}, "
          f"actual coverage={obs_ratio_actual:.2f}")

    # 绘制轨迹
    tx_2d = np.array(data['tx_positions_3d'])[:, :2] if 'tx_positions_3d' in data else None
    plot_trajectory(
        sorted_locs, x_range, y_range, tx_locs=tx_2d,
        title=f"UAV {trajectory_type.upper()} — {scene_name}",
        save_path=os.path.join(SAVE_DIR, f'trajectory_{save_prefix}.png'),
    )

    # ---- 5. 准备 ground truth ----
    # S_true: list → (R, N1, N2) ndarray (已归一化)
    S_true = np.stack(data['S'])        # (R, N1, N2)
    Phi_true = data['Phi']              # (R, K)

    # ---- 6. 序贯 II-BTD ----
    M_values = list(range(step_size, M_max + 1, step_size))
    if M_values[-1] != M_max:
        M_values.append(M_max)

    solver = II_BTD_Optimized(
        n_sources=R,
        grid_size=(N1, N2),

        mu=2.1, nu=1.2, # 1.2, 1.5
        max_iter=20,
        kernel_bandwidth=0.38, # 0.46
        warmstart=False,
    )
    solver.init_sequential(
        data['grid_points'], bounds, K=K, I_mask=data['I_mask']
    )

    nmse_list = []
    prev_M = 0

    print(f"\n  {'M':>5s}  {'NMSE':>10s}  {'Time(s)':>8s}")
    print(f"  {'-'*28}")

    for M_cur in M_values:
        new_locs = sorted_locs[prev_M:M_cur]
        new_gamma = Gamma_sorted[prev_M:M_cur, :]
        new_omega = Omega_sorted[prev_M:M_cur, :]

        t1 = time.time()
        solver.add_measurements(
            new_locs, new_gamma, new_omega,
            n_outer_iter=3, max_svt_iter=10, debugFlag=False,
        )
        dt = time.time() - t1

        nmse = solver.evaluate_reconstruction2(
            solver.Sr, solver.Phi,
            S_true, Phi_true,
        )
        nmse_list.append(nmse)
        prev_M = M_cur

        print(f"  {M_cur:5d}  {nmse:10.4f}  {dt:8.2f}")

    return M_values, nmse_list, data


# ============================================================
#  多场景对比实验
# ============================================================
def run_multi_scene_comparison(
    scenes=None,
    R=1, K=30, M_max=130, N1=51, N2=51,
    SNR_dB=20, seed=42,
    trajectory_type='tsp',
    step_size=10,
    omega_mode='full',
    omega_ratio=1.0,
):
    """
    在多个 Sionna RT 场景上运行序贯采样实验并对比 NMSE 曲线。
    """
    if scenes is None:
        scenes = [
            {'name': 'simple_street_canyon', 'label': 'Street Canyon'},
            {'name': 'munich',               'label': 'Munich'},
            {'name': 'etoile',               'label': 'Etoile'},
        ]

    print("=" * 60)
    print("  Multi-Scene Sequential Sampling Comparison")
    print(f"  R={R}, K={K}, M_max={M_max}, SNR={SNR_dB} dB")
    print(f"  Omega: {omega_mode}, ratio={omega_ratio}")
    print("=" * 60)

    all_results = {}

    for sc in scenes:
        sname = sc['name']
        label = sc['label']
        try:
            M_vals, nmse_vals, data = run_rt_sequential_experiment(
                scene_name=sname,
                R=R, K=K, M_max=M_max, N1=N1, N2=N2,
                SNR_dB=SNR_dB, seed=seed,
                trajectory_type=trajectory_type,
                step_size=step_size,
                save_prefix=sname,
                omega_mode=omega_mode,
                omega_ratio=omega_ratio,
            )
            all_results[label] = {
                'M_values': M_vals,
                'nmse': nmse_vals,
                'scene_name': sname,
                'data': data,
            }
        except Exception as e:
            print(f"\n  ✗ 场景 {sname} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("没有任何场景成功运行。")
        return all_results

    # ---- 绘制 NMSE 对比曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: NMSE vs M
    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(all_results)))
    for i, (label, res) in enumerate(all_results.items()):
        mk = MARKERS[i % len(MARKERS)]
        ax.plot(res['M_values'], res['nmse'],
                marker=mk, linestyle='-', color=colors[i],
                linewidth=2, markersize=8, label=label)
    ax.set_xlabel('Number of Sensors (M)', fontsize=13)
    ax.set_ylabel('NMSE', fontsize=13)
    ax.set_title(f'NMSE vs Sensors — Multi-Scene\n'
                 f'R={R}, K={K}, SNR={SNR_dB} dB, Trajectory={trajectory_type}',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 右图: NMSE vs M (log scale)
    ax2 = axes[1]
    for i, (label, res) in enumerate(all_results.items()):
        mk = MARKERS[i % len(MARKERS)]
        ax2.semilogy(res['M_values'], res['nmse'],
                     marker=mk, linestyle='-', color=colors[i],
                     linewidth=2, markersize=8, label=label)
    ax2.set_xlabel('Number of Sensors (M)', fontsize=13)
    ax2.set_ylabel('NMSE (log)', fontsize=13)
    ax2.set_title('NMSE vs Sensors (Log Scale)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'nmse_multi_scene_comparison.png')
    fig.savefig(save_path, dpi=150)
    print(f"\n[Saved] {save_path}")
    plt.close(fig)

    # ---- 绘制每个场景的 source map 对比 ----
    for label, res in all_results.items():
        _plot_source_maps(res, label)

    # ---- 汇总表 ----
    print(f"\n{'='*60}")
    print(f"  Summary (Final NMSE at M={M_max})")
    print(f"{'='*60}")
    print(f"  {'Scene':<25s} {'NMSE':>10s}")
    print(f"  {'-'*38}")
    for label, res in all_results.items():
        final_nmse = res['nmse'][-1]
        print(f"  {label:<25s} {final_nmse:10.4f}")

    return all_results


def _plot_source_maps(res, label):
    """为单个场景绘制 ground truth vs estimated source maps"""
    data = res['data']
    S_true = np.stack(data['S'])
    cfg = data['config']
    R = cfg.R

    # 目前没有 solver 引用，先跳过此图
    # 后续可以将 solver 也存入 res 中
    pass


# ============================================================
#  单场景快速测试
# ============================================================
def run_quick_test(scene_name='munich', M_max=60, step_size=20):
    """快速测试：少量传感器，验证流程"""
    M_vals, nmse_vals, data = run_rt_sequential_experiment(
        scene_name=scene_name,
        R=1, K=20, M_max=M_max, N1=31, N2=31,
        SNR_dB=20, seed=42,
        step_size=step_size,
        save_prefix=f'{scene_name}_quick',
    )

    # 绘制简单结果
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(M_vals, nmse_vals, 'o-', linewidth=2, markersize=10, color='#2196F3')
    ax.fill_between(M_vals, nmse_vals, alpha=0.2, color='#2196F3')
    ax.set_xlabel('Number of Sensors (M)', fontsize=12)
    ax.set_ylabel('NMSE', fontsize=12)
    ax.set_title(f'Quick Test — {scene_name}\nR=1, K=20, Grid=31x31', fontsize=14)
    ax.grid(True, alpha=0.3)

    for m, n in zip(M_vals, nmse_vals):
        ax.annotate(f'{n:.3f}', (m, n), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'nmse_quick_{scene_name}.png')
    fig.savefig(save_path, dpi=150)
    print(f"[Saved] {save_path}")
    plt.close(fig)

    return M_vals, nmse_vals


# ============================================================
#  入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='II-BTD with Sionna RT data')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['quick', 'single', 'multi'],
                        help='quick=快速测试, single=单场景, multi=多场景对比')
    parser.add_argument('--scene', type=str, default='simple_street_canyon',
                        help='单场景模式的场景名')
    parser.add_argument('--M', type=int, default=140, help='最大传感器数')
    parser.add_argument('--R', type=int, default=1, help='信号源数')
    parser.add_argument('--K', type=int, default=30, help='频段数')
    parser.add_argument('--SNR', type=int, default=20, help='SNR (dB)')
    parser.add_argument('--step', type=int, default=10, help='采样步长')
    parser.add_argument('--trajectory', type=str, default='snake',
                        choices=['tsp', 'snake', 'angle'])
    parser.add_argument('--omega', type=str, default='cyclic',
                        choices=['full', 'dual_center', 'random', 'cyclic'],
                        help='频率域观测模式')
    parser.add_argument('--ratio', type=float, default=0.6,
                        help='频段观测比例 (0,1], 仅在 omega!=full 时有效')

    args = parser.parse_args()

    if args.mode == 'quick':
        run_quick_test(args.scene, M_max=60, step_size=20)

    elif args.mode == 'single':
        run_rt_sequential_experiment(
            scene_name=args.scene,
            R=args.R, K=args.K, M_max=args.M,
            SNR_dB=args.SNR, seed=42,
            trajectory_type=args.trajectory,
            step_size=args.step,
            omega_mode=args.omega,
            omega_ratio=args.ratio,
        )

    elif args.mode == 'multi':
        run_multi_scene_comparison(
            R=args.R, K=args.K, M_max=args.M,
            SNR_dB=args.SNR, seed=42,
            trajectory_type=args.trajectory,
            step_size=args.step,
            omega_mode=args.omega,
            omega_ratio=args.ratio,
        )
