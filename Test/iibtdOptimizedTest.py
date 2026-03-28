from spectrumMapTensorGen import SimConfig
from spectrumMapTensorGen import *
import sys
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹出图片
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from IIBTD.IIBTD_Optimized import II_BTD_Optimized

# 定义不同的 markers 和 linestyles 用于绑图
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]


def sort_sensors_by_trajectory(locs, trajectory_type='tsp'):
    """
    将随机传感器位置按照某种轨迹进行排序，模拟 UAV 飞行路径
    
    Parameters:
    -----------
    locs : (M, 2) 原始随机传感器位置
    trajectory_type : str
        'tsp' - 贪婪最近邻 TSP 排序
        'snake' - 蛇形扫描排序
        'angle' - 按角度排序（从中心向外螺旋）
    
    Returns:
    --------
    sorted_locs : (M, 2) 排序后的传感器位置
    order : (M,) 排序索引
    """
    M = len(locs)
    
    if trajectory_type == 'tsp':
        # 贪婪最近邻 TSP：从第一个点开始，每次选最近的未访问点
        visited = np.zeros(M, dtype=bool)
        order = [0]  # 从第 0 个点开始
        visited[0] = True
        
        for _ in range(M - 1):
            current = order[-1]
            current_loc = locs[current]
            
            # 计算到所有未访问点的距离
            dists = np.linalg.norm(locs - current_loc, axis=1)
            dists[visited] = np.inf  # 已访问的设为无穷大
            
            # 选择最近的未访问点
            next_point = np.argmin(dists)
            order.append(next_point)
            visited[next_point] = True
        
        order = np.array(order)
        
    elif trajectory_type == 'snake':
        # 蛇形扫描：按 Y 坐标分组，交替反向
        y_sorted_idx = np.argsort(locs[:, 1])
        n_rows = int(np.sqrt(M))
        points_per_row = M // n_rows + 1
        
        order = []
        for i in range(n_rows + 1):
            start = i * points_per_row
            end = min((i + 1) * points_per_row, M)
            row_indices = y_sorted_idx[start:end]
            
            # 按 X 坐标排序
            row_x_sorted = row_indices[np.argsort(locs[row_indices, 0])]
            
            if i % 2 == 1:  # 奇数行反向
                row_x_sorted = row_x_sorted[::-1]
            order.extend(row_x_sorted)
        
        order = np.array(order[:M])
        
    elif trajectory_type == 'angle':
        # 从中心向外螺旋：按角度和距离排序
        center = locs.mean(axis=0)
        diff = locs - center
        angles = np.arctan2(diff[:, 1], diff[:, 0])
        radii = np.linalg.norm(diff, axis=1)
        
        # 按角度主排序，距离次排序
        order = np.lexsort((radii, angles))
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    sorted_locs = locs[order]
    return sorted_locs, order


def plot_trajectory(sensor_locs, cfg, source_loc=None, title="UAV Flight Trajectory", save_path=None):
    """
    绘制传感器采样轨迹
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    M = len(sensor_locs)
    
    # 绘制轨迹线
    ax.plot(sensor_locs[:, 0], sensor_locs[:, 1], 
            'b-', alpha=0.5, linewidth=1.5, label='Flight Path')
    
    # 绘制采样点，颜色按顺序渐变
    scatter = ax.scatter(sensor_locs[:, 0], sensor_locs[:, 1], 
                         c=np.arange(M), cmap='viridis', 
                         s=60, edgecolors='k', linewidth=0.5, zorder=5)
    
    # 标记起点和终点
    ax.scatter(sensor_locs[0, 0], sensor_locs[0, 1], 
               c='lime', s=200, marker='^', edgecolors='k', 
               linewidth=2, zorder=10, label='Start')
    ax.scatter(sensor_locs[-1, 0], sensor_locs[-1, 1], 
               c='red', s=200, marker='s', edgecolors='k', 
               linewidth=2, zorder=10, label='End')
    
    # 标记源位置
    if source_loc is not None:
        ax.scatter(source_loc[0], source_loc[1], 
                   c='yellow', s=300, marker='*', edgecolors='k', 
                   linewidth=2, zorder=10, label='Source')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Sampling Order', fontsize=10)
    
    ax.set_xlim(-1, cfg.L)
    ax.set_ylim(-1, cfg.L)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{title}\n(Total: {M} sampling points)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def run(M_max):
    """原始的单次运行函数 (使用序贯采样API)"""
    cfg = SimConfig(full_obs=False, rho=0.05, R = 1, M = M_max)
    print(cfg.summary())
    data = generate_data(cfg, seed=42)
    bounds = ((0, cfg.L), (0, cfg.L))
    print("\n" + "="*60)
    print("Running II-BTD (Sequential)...")
    print("="*60)

    solver = II_BTD_Optimized(n_sources = cfg.R,  grid_size=(cfg.N1,cfg.N2), 
                              mu=1.2, nu=1.5, max_iter= 6, kernel_bandwidth=0.46, warmstart= False)
    solver.init_sequential(data['grid_points'], bounds, K=cfg.K, I_mask=data['I_mask'])
    
    # 一次性添加所有传感器
    solver.add_measurements(
        data['sensor_locs'], data['Gamma_obs'], data['Omega'],
        n_outer_iter=6, max_svt_iter=100, debugFlag=True
    )
    
    solver.evaluate_reconstruction2(solver.Sr, solver.Phi, data['S'], data['Phi'])
    print("\nII-BTD completed.")
    print(f"Shape of estimated power spectrum: {solver.Sr.shape}")


def run_incremental_experiment_trajectory(M_max=130, seed=42, trajectory_type='tsp'):
    """
    使用轨迹排序的增量采样实验：逐渐增加传感器数量，观察 NMSE 变化
    
    Parameters:
    -----------
    M_max : int, 最大传感器数量
    seed : int, 随机种子
    trajectory_type : str, 轨迹类型 ('tsp', 'snake', 'angle')
    """
    # 1. 生成完整数据
    cfg = SimConfig(full_obs=False, rho = 0.05, M=M_max, R=1)
    data = generate_data(cfg, seed=seed)
    bounds = ((0, cfg.L), (0, cfg.L))
    
    # 2. 按轨迹排序传感器位置
    original_locs = data['sensor_locs']
    sorted_locs, order = sort_sensors_by_trajectory(original_locs, trajectory_type=trajectory_type)
    
    # 重新排序相关数据
    Gamma_sorted = data['Gamma_obs'][order, :]
    Omega_sorted = data['Omega'][order, :]
    
    print("="*60)
    print(f" 增量采样实验 (轨迹类型: {trajectory_type})")
    print("="*60)
    print(f"总传感器数: {M_max}, 网格: {cfg.N1}x{cfg.N2}, 频段: {cfg.K}")
    
    # 3. 绘制轨迹
    plot_trajectory(sorted_locs, cfg, 
                    source_loc=data['source_locs'][0] if 'source_locs' in data else None,
                    title=f"UAV {trajectory_type.upper()} Trajectory",
                    save_path=f'trajectory_{trajectory_type}.png')
    
    # 4. 定义采样点数量序列
    M_values = []
    for i in range(20, M_max + 1, 10):
        M_values.append(i)
    # M_values = [10, 20, 30, 40, 50, 60, 80, 100, 120,140,150]
    M_values = [m for m in M_values if m <= M_max]
    
    nmse_list = []
    
    # 5. 逐步增加传感器数量
    # 创建序贯采样 solver
    solver = II_BTD_Optimized(
        n_sources=cfg.R,
        grid_size=(cfg.N1, cfg.N2),
        mu=1.2, nu=1.5, 
        max_iter=6, 
        kernel_bandwidth=0.46, 
        warmstart=False
    )
    solver.init_sequential(data['grid_points'], bounds, K=cfg.K, I_mask=data['I_mask'])
    
    prev_M = 0
    for M_current in M_values:
        # 增量添加新传感器
        new_locs = sorted_locs[prev_M:M_current]
        new_gamma = Gamma_sorted[prev_M:M_current, :]
        new_omega = Omega_sorted[prev_M:M_current, :]
        
        solver.add_measurements(
            new_locs, new_gamma, new_omega,
            n_outer_iter=4, max_svt_iter=30, debugFlag=True
        )
        
        nmse = solver.evaluate_reconstruction2(
            solver.Sr, solver.Phi, 
            data['S'], data['Phi']
        )
        nmse_list.append(nmse)
        prev_M = M_current
        
        print(f"  M = {M_current:3d}  |  NMSE = {nmse:.4f}")
    
    # 6. 绘制结果
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：NMSE 曲线
    axes[0].plot(M_values, nmse_list, 'o-', linewidth=2, markersize=10, color='#2196F3')
    axes[0].fill_between(M_values, nmse_list, alpha=0.2, color='#2196F3')
    axes[0].set_xlabel('Number of Sensors (M)', fontsize=12)
    axes[0].set_ylabel('NMSE', fontsize=12)
    axes[0].set_title(f'NMSE vs Sensors ({trajectory_type})', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(M_values)
    
    for m, nmse in zip(M_values, nmse_list):
        axes[0].annotate(f'{nmse:.3f}', (m, nmse), textcoords="offset points", 
                         xytext=(0, 10), ha='center', fontsize=9)
    
    # 右图：增量覆盖可视化
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(M_values)))
    for i, M_current in enumerate(M_values):
        alpha = 0.3 if i < len(M_values) - 1 else 1.0
        axes[1].scatter(sorted_locs[:M_current, 0], sorted_locs[:M_current, 1], 
                        c=[colors[i]], s=20, alpha=alpha, 
                        label=f'M={M_current}' if i % 2 == 0 else '')
    axes[1].plot(sorted_locs[:, 0], sorted_locs[:, 1], 'k-', alpha=0.2, linewidth=0.5)
    axes[1].set_xlim(-1, cfg.L)
    axes[1].set_ylim(-1, cfg.L)
    axes[1].set_xlabel('X (m)', fontsize=12)
    axes[1].set_ylabel('Y (m)', fontsize=12)
    axes[1].set_title('Incremental Sampling Coverage', fontsize=14)
    axes[1].set_aspect('equal')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'nmse_trajectory_{trajectory_type}.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print(f" 实验完成！结果已保存")
    print("="*60)
    
    return M_values, nmse_list


def generate_omega_dual_center(M, K, ratio, rng):
    """
    生成以 K/4 和 3K/4 为中心的双中心频段观测模式
    
    Parameters:
    -----------
    M : int, 传感器数量
    K : int, 总频段数
    ratio : float, 观测频段占比 (0, 1]
    rng : numpy random generator
    
    Returns:
    --------
    Omega : (M, K) 观测指示矩阵
    """
    print(f"观测模式: 以 K/4={K//4} 和 3K/4={3*K//4} 为中心双中心扩展")

    Omega = np.zeros((M, K), dtype=np.int32)
    
    # 计算每个中心观测的频段数
    n_obs = int(K * ratio)
    n_per_center = max(1, n_obs // 2)  # 每个中心的观测数
    
    # 两个中心位置
    center1 = K // 4
    center2 = 3 * K // 4
    
    for m in range(M):
        observed_bands = set()
        
        # 从中心1扩展
        for offset in range(n_per_center):
            left = center1 - offset
            right = center1 + offset
            if left >= 0:
                observed_bands.add(left)
            if right < K:
                observed_bands.add(right)
            if len(observed_bands) >= n_per_center:
                break
        
        # 从中心2扩展
        for offset in range(n_per_center):
            left = center2 - offset
            right = center2 + offset
            if left >= 0:
                observed_bands.add(left)
            if right < K:
                observed_bands.add(right)
            if len(observed_bands) >= n_obs:
                break
        
        # 设置观测指示
        for band in observed_bands:
            Omega[m, band] = 1
    
    return Omega


def generate_omega_random(M, K, ratio, rng):
    """
    生成随机选择的频段观测模式
    
    Parameters:
    -----------
    M : int, 传感器数量
    K : int, 总频段数
    ratio : float, 观测频段占比 (0, 1]，表示每个传感器随机观测的频段比例
    rng : numpy random generator
    
    Returns:
    --------
    Omega : (M, K) 观测指示矩阵
    """
    Omega = np.zeros((M, K), dtype=np.int32)
    
    # 计算每个传感器观测的频段数
    n_obs = max(1, int(K * ratio))
    
    for m in range(M):
        # 随机选择 n_obs 个频段
        observed_bands = rng.choice(K, size=n_obs, replace=False)
        Omega[m, observed_bands] = 1
    
    return Omega


def generate_omega_cyclic(M, K, B, overlap=0, start_offset=0):
    """
    循环频谱切换模式：每个航点观测不同的频段块，循环覆盖整个频谱
    支持相邻频段块之间的重叠
    
    模拟场景：
    - UAV 飞行一个航点，切换一个观测中频
    - 相邻航点的观测频段有部分重叠
    - 继续循环
    
    Parameters:
    -----------
    M : int, 航点数（传感器数量）
    K : int, 总频段数
    B : int, 每次观测的带宽（频段数）
    overlap : int, 相邻频段块的重叠频段数 (0 <= overlap < B)
    start_offset : int, 起始频段偏移（可用于错开不同实验）
    
    Returns:
    --------
    Omega : (M, K) 观测指示矩阵
    
    Example:
    --------
    K=30, B=6, overlap=2 时，步长 = B - overlap = 4:
      航点 0: 观测 [0-5]
      航点 1: 观测 [4-9]    (与上一个重叠 [4,5])
      航点 2: 观测 [8-13]   (与上一个重叠 [8,9])
      航点 3: 观测 [12-17]
      航点 4: 观测 [16-21]
      航点 5: 观测 [20-25]
      航点 6: 观测 [24-29]
      航点 7: 观测 [0-5]    (循环，从头开始)
      ...
    """
    Omega = np.zeros((M, K), dtype=np.int32)
    
    # 确保 overlap 在合理范围内
    overlap = max(0, min(overlap, B - 1))
    
    # 步长 = 带宽 - 重叠
    step = B - overlap
    
    # 计算完成一轮循环需要的步数
    n_steps = max(1, (K - overlap) // step)  # 不超过 K 的步数
    
    for m in range(M):
        # 当前航点的起始频段
        step_idx = (m + start_offset) % n_steps
        start_freq = step_idx * step
        end_freq = min(start_freq + B, K)
        Omega[m, start_freq:end_freq] = 1
    
    return Omega


def run_frequency_ratio_experiment(M_max=160, seed=42, trajectory_type='tsp'):
    """
    频段占比 + 轨迹增量采样实验：测试不同观测频段比例下，随采样点数增加 NMSE 的变化
    
    Parameters:
    -----------
    M_max : int, 最大传感器数量
    seed : int, 随机种子
    trajectory_type : str, 轨迹类型 ('tsp', 'snake', 'angle')
    """
    rng = np.random.default_rng(seed)
    
    # 1. 生成完整数据
    cfg = SimConfig(full_obs=True, rho=0.05, R = 1, M = M_max)
    data = generate_data(cfg, seed=seed)
    bounds = ((0, cfg.L), (0, cfg.L))
    K = cfg.K
    
    # 2. 按轨迹排序传感器位置
    original_locs = data['sensor_locs']
    sorted_locs, order = sort_sensors_by_trajectory(original_locs, trajectory_type=trajectory_type)
    Gamma_sorted = data['Gamma_obs'][order, :]
    
    print("="*60)
    print(f" 频段占比 + 轨迹增量实验 (轨迹类型: {trajectory_type})")
    print("="*60)
    print(f"最大传感器数: {M_max}, 网格: {cfg.N1}x{cfg.N2}, 总频段: {K}")

    
    # 3. 绘制轨迹
    plot_trajectory(sorted_locs, cfg, 
                    source_loc=data['source_locs'][0] if 'source_locs' in data else None,
                    title=f"UAV {trajectory_type.upper()} Trajectory (Freq Ratio Exp)",
                    save_path=f'trajectory_freq_ratio_{trajectory_type}.png')
    
    # 4. 定义参数范围
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    M_values = list(range(10, M_max + 1, 10))  # 每隔 20 个采样一次
    M_values = [m for m in M_values if m <= M_max]
    
    # M_values = [160]
    
    # 反转数据
    # M_values.reverse()
    ratios.reverse()
    
    print(f"\n参数范围:")
    print(f"  ratios: {ratios}")
    print(f"  M_values: {M_values}")
    
    # 5. 2D NMSE 张量: (n_ratio, n_M)
    nmse_matrix = np.zeros((len(ratios), len(M_values)))
    
    total_runs = len(ratios) * len(M_values)
    current_run = 0
    
    print(f"\n开始实验... (共 {total_runs} 次)")
    
    # 6. 运行实验
    for i, ratio in enumerate(ratios):
        # 生成完整的观测掩码
        rng_omega = np.random.default_rng(seed + 2000)
        Omega_full = generate_omega_dual_center(M_max, K, ratio, rng_omega)
        # Omega_full = generate_omega_random(M_max, K, ratio, rng_omega)
        
        # 为每个 ratio 创建序贯 solver
        solver = II_BTD_Optimized(
            n_sources=cfg.R,
            grid_size=(cfg.N1, cfg.N2),
            mu=1.2, nu=1.5, 
            max_iter=6, 
            kernel_bandwidth=0.46, 
            warmstart=False
        )
        solver.init_sequential(data['grid_points'], bounds, K=K, I_mask=data['I_mask'])
        
        prev_M = 0
        for j, M_current in enumerate(M_values):
            current_run += 1
            
            # 增量添加新传感器
            new_locs = sorted_locs[prev_M:M_current]
            new_gamma = Gamma_sorted[prev_M:M_current, :] * Omega_full[prev_M:M_current, :]
            new_omega = Omega_full[prev_M:M_current, :]
            
            solver.add_measurements(
                new_locs, new_gamma, new_omega,
                n_outer_iter=2, max_svt_iter=20, debugFlag=False
            )
            
            nmse = solver.evaluate_reconstruction2(
                solver.Sr, solver.Phi, 
                data['S'], data['Phi'], 
                drawFlag=False
            )
            nmse_matrix[i, j] = nmse
            prev_M = M_current
            
            print(f"  [{current_run:3d}/{total_runs}] ratio={ratio:.1f}, M={M_current:3d}: NMSE={nmse:.4f}")
    
    # 7. 绘制结果 - NMSE vs M (不同 ratio)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：NMSE vs M (不同 ratio 的曲线)
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(ratios)))
    for i, ratio in enumerate(ratios):
        marker = MARKERS[i % len(MARKERS)]
        axes[0].plot(M_values, nmse_matrix[i, :], 
                     marker=marker, linestyle='-',
                     color=colors[i], linewidth=2, markersize=8,
                     label=f'ratio={ratio:.1f}')
    
    axes[0].set_xlabel('Number of Sensors (M)', fontsize=12)
    axes[0].set_ylabel('NMSE', fontsize=12)
    axes[0].set_title(f'NMSE vs Sensors (Different Frequency Ratios)\nTrajectory: {trajectory_type}', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(M_values)
    
    # 右图：热力图
    im = axes[1].imshow(nmse_matrix, aspect='auto', cmap='RdYlGn_r', 
                        extent=[M_values[0]-10, M_values[-1]+10, len(ratios)-0.5, -0.5])
    axes[1].set_xlabel('Number of Sensors (M)', fontsize=12)
    axes[1].set_ylabel('Observation Ratio', fontsize=12)
    axes[1].set_title('NMSE Heatmap (Ratio vs M)', fontsize=14)
    axes[1].set_yticks(range(len(ratios)))
    axes[1].set_yticklabels([f'{r:.1f}' for r in ratios])
    axes[1].set_xticks(M_values)
    
    # 在热力图上标注数值
    for i in range(len(ratios)):
        for j in range(len(M_values)):
            text_color = 'white' if nmse_matrix[i, j] > 0.5 else 'black'
            axes[1].text(M_values[j], i, f'{nmse_matrix[i, j]:.2f}', 
                        ha='center', va='center', color=text_color, fontsize=8)
    
    plt.colorbar(im, ax=axes[1], shrink=0.8, label='NMSE')
    
    plt.tight_layout()
    plt.savefig('nmse_ratio_vs_M.png', dpi=150)
    plt.show()
    
    # 8. 额外绘制：NMSE vs Ratio (不同 M)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    colors_M = plt.cm.viridis(np.linspace(0, 0.9, len(M_values)))
    for j, M_current in enumerate(M_values):
        marker = MARKERS[j % len(MARKERS)]
        ax2.plot(ratios, nmse_matrix[:, j], 
                 marker=marker, linestyle='-',
                 color=colors_M[j], linewidth=2, markersize=8,
                 label=f'M={M_current}')
    
    ax2.set_xlabel('Observation Ratio (K_obs / K)', fontsize=12)
    ax2.set_ylabel('NMSE', fontsize=12)
    ax2.set_title(f'NMSE vs Observation Ratio (Different M)\nTrajectory: {trajectory_type}', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ratios)
    
    plt.tight_layout()
    plt.savefig('nmse_M_vs_ratio.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print(" 实验完成！结果已保存:")
    print("  - nmse_ratio_vs_M.png (NMSE vs M 曲线 + 热力图)")
    print("  - nmse_M_vs_ratio.png (NMSE vs Ratio 曲线)")
    print("="*60)
    
    return ratios, M_values, nmse_matrix


def generate_measurements_with_lfreq(prop_sensor, Phi, cfg, rng, sensor_locs, l_freq=0.01):
    """
    带有可变 l_freq 的测量生成函数
    
    Parameters:
    -----------
    l_freq : float, 频率相关长度 (GHz)，越大表示频率选择性衰落越平坦
    """
    M, K, R = cfg.M, cfg.K, cfg.R
    
    Gamma_clean = np.zeros((M, K))
    for r in range(R):
        # 使用传入的 l_freq 生成频率选择性衰落
        shadow = sample_joint_shadow_fading(
            sensor_locs, 
            np.linspace(2.4 - 0.1, 2.4 + 0.1, cfg.K), 
            0.5,  # sigma
            30,    # l_space
            l_freq,  # 可变的频率相关长度
            rng
        )
        shadow = 10.0 ** (shadow / 10.0)
        Gamma_clean += np.outer(prop_sensor[r], Phi[r]) * shadow
    
    # 计算噪声
    signal_power = np.mean(Gamma_clean ** 2)
    sigma2_eps = signal_power / (10.0 ** (cfg.SNR_dB / 10.0))
    
    noise = rng.normal(0, np.sqrt(sigma2_eps), (M, K))
    Gamma_obs = Gamma_clean + noise
    
    # 默认全观测
    Omega = np.ones((M, K), dtype=np.int32)
    
    return Gamma_obs, Gamma_clean, Omega, sigma2_eps


def run_lfreq_ratio_experiment(M_max=140, seed=42, trajectory_type='tsp'):
    """
    l_freq 和 ratio 联合实验：测试频率选择性衰落强度与观测比例的交互效应
    
    绘制两种图：
    1. 固定 l_freq，不同 ratio 下，NMSE vs 采样数量 M
    2. 固定 ratio，不同 l_freq 下，NMSE vs 采样数量 M
    """
    import numpy as np
    
    # 1. 生成基础数据
    cfg = SimConfig(full_obs=True, M=M_max, R=1)
    data = generate_data(cfg, seed=seed)
    bounds = ((0, cfg.L), (0, cfg.L))
    K = cfg.K
    
    # 2. 按轨迹排序
    original_locs = data['sensor_locs']
    sorted_locs, order = sort_sensors_by_trajectory(original_locs, trajectory_type=trajectory_type)
    Gamma_base = data['Gamma_clean'][order, :]
    
    print("="*60)
    print(" L_freq-Ratio-M 联合实验")
    print("="*60)
    print(f"最大传感器数: {M_max}, 网格: {cfg.N1}x{cfg.N2}, 总频段: {K}")
    
    # 3. 定义参数范围
    l_freq_values = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    M_values = []
    for i in range(10, M_max + 1, 10):
        M_values.append(i)
    # M_values = [20, 40, 60, 80, 100, 120, 140]
    M_values = [m for m in M_values if m <= M_max]
    # M_values = [160]
    # ratios = [0.5]
    ratios.reverse()
    # l_freq_values.reverse()

    # 4. 定义观测策略
    omega_strategies = ['dual_center', 'random', 'cyclic']
    
    # 5. 4D NMSE 张量: (n_strategy, n_lfreq, n_ratio, n_M)
    nmse_tensor = np.zeros((len(omega_strategies), len(l_freq_values), len(ratios), len(M_values)))
    
    print(f"\n参数范围:")
    print(f"  strategies: {omega_strategies}")
    print(f"  l_freq: {l_freq_values}")
    print(f"  ratio:  {ratios}")
    print(f"  M:      {M_values}")
    
    total_runs = len(omega_strategies) * len(l_freq_values) * len(ratios) * len(M_values)
    current_run = 0
    
    # 使用排序后的 prop_sensor 和 sensor_locs 保持一致性
    prop_sensor_sorted = [p[order] for p in data['prop_sensor']]  # 按 order 排序
    sensor_locs_sorted = sorted_locs  # 已排序的传感器位置
    Phi = data['Phi']
    
    # rng = np.random.default_rng(seed)
    # 5. 运行实验
    print(f"\n开始实验... (共 {total_runs} 次)")
    for i, l_freq in enumerate(l_freq_values):
        # 每次 l_freq 循环重置 RNG，确保公平对比
        # rng_noise = np.random.default_rng(seed + 500)
        rng = np.random.default_rng(seed + 300)
        
        fading = np.zeros((cfg.M,cfg.K)) 
        for r in range(cfg.R):
            shadow = sample_joint_shadow_fading(
                sensor_locs_sorted,   # 使用排序后的传感器位置
                np.linspace(2.4 - 0.1, 2.4 + 0.1, cfg.K), 
                2, 
                10, 
                l_freq,
                rng) 
            # shadow = shadow - np.mean(shadow)
            shadow = 10.0 ** (shadow / 10.0)
            shadow /= np.mean(shadow)
            fading += (shadow - 1) * prop_sensor_sorted[r][:, np.newaxis] * Phi[r][np.newaxis, :]
            # fading += shadow * Phi[r][np.newaxis, :]
        # ---- noise variance from target SNR ----
        signal_power = np.mean((Gamma_base + fading) ** 2)
        sigma2_eps = signal_power / (10.0 ** (cfg.SNR_dB / 10.0))

        # ---- additive Gaussian noise ----
        noise_meas = rng.standard_normal((cfg.M, cfg.K)) * np.sqrt(sigma2_eps)
        Gamma_with_fading_full = np.maximum(Gamma_base + fading + noise_meas, 1e-10)
        
        for s, strategy in enumerate(omega_strategies):
            for j, ratio in enumerate(ratios):
                # 每次循环重置 rng_omega，确保公平对比
                rng_omega = np.random.default_rng(seed + 2000)
                
                if strategy == 'dual_center':
                    Omega_full = generate_omega_dual_center(M_max, K, ratio, rng_omega)
                elif strategy == 'random':
                    Omega_full = generate_omega_random(M_max, K, ratio, rng_omega)
                elif strategy == 'cyclic':
                    # 确保 B 至少为 1，且适用于循环策略
                    B = max(1, int(K * ratio))
                    Omega_full = generate_omega_cyclic(M_max, K, B, overlap=min(2, B-1), start_offset=0)
                
                # 为每个 (strategy, ratio) 创建序贯 solver
                solver = II_BTD_Optimized(
                    n_sources=cfg.R,
                    grid_size=(cfg.N1, cfg.N2),
                    mu=1.2, nu=1.5, 
                    max_iter=6, 
                    kernel_bandwidth=0.46, 
                    warmstart=False
                )
                solver.init_sequential(data['grid_points'], bounds, K=K, I_mask=data['I_mask'])
                
                prev_M = 0
                for k, M_current in enumerate(M_values):
                    current_run += 1
                    
                    # 增量添加新传感器
                    new_gamma = Gamma_with_fading_full[prev_M:M_current, :] * Omega_full[prev_M:M_current, :]
                    new_omega = Omega_full[prev_M:M_current, :]
                    new_locs = sorted_locs[prev_M:M_current]
                    
                    solver.add_measurements(
                        new_locs, new_gamma, new_omega,
                        n_outer_iter=2, max_svt_iter=20, debugFlag=False
                    )
                    
                    nmse = solver.evaluate_reconstruction2(
                        solver.Sr, solver.Phi, 
                        data['S'], data['Phi']
                    )
                    nmse_tensor[s, i, j, k] = nmse
                    prev_M = M_current
                    
                    print(f"  [{current_run}/{total_runs}] {strategy}, l_freq={l_freq}, ratio={ratio}, M={M_current}: NMSE={nmse:.4f}")
    
    # 6. 绘制结果 - 为每个策略创建对比图
    n_strategies = len(omega_strategies)
    n_lfreq = len(l_freq_values)
    
    # 图1: 比较不同策略 (固定 l_freq, 固定 ratio, NMSE vs M)
    # 选择中间的 l_freq 和 ratio 作为示例
    mid_lfreq_idx = n_lfreq // 2
    mid_ratio_idx = len(ratios) // 2
    
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 固定 l_freq，比较策略
    ax1 = axes1[0]
    strategy_colors = ['#2196F3', '#FF5722', '#4CAF50']
    strategy_markers = ['o', 's', '^']  # 不同策略用不同 marker
    for s, strategy in enumerate(omega_strategies):
        for j, ratio in enumerate(ratios):
            linestyle = LINESTYLES[j % len(LINESTYLES)]
            ax1.plot(M_values, nmse_tensor[s, mid_lfreq_idx, j, :], 
                    marker=strategy_markers[s], linestyle=linestyle, 
                    color=strategy_colors[s],
                    linewidth=2, markersize=6, 
                    label=f'{strategy}, ratio={ratio}')
    ax1.set_xlabel('Number of Sensors (M)', fontsize=10)
    ax1.set_ylabel('NMSE', fontsize=10)
    ax1.set_title(f'Strategy Comparison (l_freq={l_freq_values[mid_lfreq_idx]})', fontsize=12)
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 右图: 固定 ratio，比较策略
    ax2 = axes1[1]
    for s, strategy in enumerate(omega_strategies):
        for i, l_freq in enumerate(l_freq_values):
            linestyle = LINESTYLES[i % len(LINESTYLES)]
            ax2.plot(M_values, nmse_tensor[s, i, mid_ratio_idx, :], 
                    marker=strategy_markers[s], linestyle=linestyle, 
                    color=strategy_colors[s],
                    linewidth=2, markersize=6, 
                    label=f'{strategy}, l_freq={l_freq}')
    ax2.set_xlabel('Number of Sensors (M)', fontsize=10)
    ax2.set_ylabel('NMSE', fontsize=10)
    ax2.set_title(f'Strategy Comparison (ratio={ratios[mid_ratio_idx]})', fontsize=12)
    ax2.legend(loc='upper right', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Observation Strategy Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('nmse_strategy_comparison.png', dpi=150)
    plt.close()  # 关闭图形，避免阻塞
    
    # 图2: 每个策略的详细结果 (子图矩阵)
    fig2, axes2 = plt.subplots(n_strategies, 2, figsize=(12, 4 * n_strategies))
    
    colors_ratio = plt.cm.plasma(np.linspace(0, 1, len(ratios)))
    colors_lfreq = plt.cm.viridis(np.linspace(0, 1, len(l_freq_values)))
    
    for s, strategy in enumerate(omega_strategies):
        # 左列: 固定 l_freq，不同 ratio
        ax_left = axes2[s, 0]
        for j, ratio in enumerate(ratios):
            marker = MARKERS[j % len(MARKERS)]
            ax_left.plot(M_values, nmse_tensor[s, mid_lfreq_idx, j, :], 
                        marker=marker, linestyle='-',
                        color=colors_ratio[j], linewidth=2, markersize=6, 
                        label=f'ratio={ratio}')
        ax_left.set_xlabel('Number of Sensors (M)', fontsize=10)
        ax_left.set_ylabel('NMSE', fontsize=10)
        ax_left.set_title(f'{strategy}: l_freq={l_freq_values[mid_lfreq_idx]}', fontsize=12)
        ax_left.legend(loc='upper right', fontsize=8)
        ax_left.grid(True, alpha=0.3)
        
        # 右列: 固定 ratio，不同 l_freq
        ax_right = axes2[s, 1]
        for i, l_freq in enumerate(l_freq_values):
            marker = MARKERS[i % len(MARKERS)]
            ax_right.plot(M_values, nmse_tensor[s, i, mid_ratio_idx, :], 
                         marker=marker, linestyle='-',
                         color=colors_lfreq[i], linewidth=2, markersize=6, 
                         label=f'l_freq={l_freq}')
        ax_right.set_xlabel('Number of Sensors (M)', fontsize=10)
        ax_right.set_ylabel('NMSE', fontsize=10)
        ax_right.set_title(f'{strategy}: ratio={ratios[mid_ratio_idx]}', fontsize=12)
        ax_right.legend(loc='upper right', fontsize=8)
        ax_right.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed Results by Strategy', fontsize=14)
    plt.tight_layout()
    plt.savefig('nmse_strategy_details.png', dpi=150)
    plt.close()  # 关闭图形，避免内存泄漏
    
    # 保存数据到文件
    np.savez('lfreq_ratio_results.npz', 
             omega_strategies=omega_strategies,
             l_freq_values=l_freq_values, 
             ratios=ratios, 
             M_values=M_values, 
             nmse_tensor=nmse_tensor)
    
    print("\n" + "="*60)
    print(" 实验完成！结果已保存:")
    print("  - nmse_strategy_comparison.png (策略对比)")
    print("  - nmse_strategy_details.png (各策略详细结果)")
    print("  - lfreq_ratio_results.npz (数据文件)")
    print("="*60)
    
    return omega_strategies, l_freq_values, ratios, M_values, nmse_tensor


def run_cyclic_experiment(M_max=160, seed=42, trajectory_type='tsp'):
    """
    循环频谱切换实验：测试不同带宽 B 下的 NMSE 性能
    
    模拟场景：每个航点切换观测中频，循环覆盖整个频谱
    """
    rng = np.random.default_rng(seed)
    
    # 1. 生成完整数据
    cfg = SimConfig(full_obs=True, M=M_max, R=1)
    data = generate_data(cfg, seed=seed)
    bounds = ((0, cfg.L), (0, cfg.L))
    K = cfg.K  # 总频段数
    
    # 2. 按轨迹排序
    original_locs = data['sensor_locs']
    sorted_locs, order = sort_sensors_by_trajectory(original_locs, trajectory_type=trajectory_type)
    Gamma_sorted = data['Gamma_obs'][order, :]
    
    print("="*60)
    print(" 循环频谱切换实验")
    print("="*60)
    print(f"最大传感器数: {M_max}, 网格: {cfg.N1}x{cfg.N2}, 总频段: {K}")
    
    # 3. 定义参数范围
    # B_values: 每次观测的频段数（带宽）
    B_values = [3, 5, 6, 10, 15, 30]  # K=30 的约数
    B_values = [b for b in B_values if K % b == 0]  # 确保能整除
    
    M_values = list(range(20, M_max + 1, 10))
    M_values = [m for m in M_values if m <= M_max]
    
    print(f"\n参数范围:")
    print(f"  B (带宽): {B_values}")
    print(f"  M (航点数): {M_values}")
    
    # 4. 可视化不同 B 下的 Omega 模式
    fig_omega, axes_omega = plt.subplots(2, 3, figsize=(15, 8))
    axes_omega = axes_omega.flatten()
    sample_M = 60  # 显示前 60 个航点
    
    for idx, B in enumerate(B_values[:6]):
        Omega_sample = generate_omega_cyclic(sample_M, K, B, overlap=2)
        axes_omega[idx].imshow(Omega_sample, aspect='auto', cmap='Blues')
        axes_omega[idx].set_xlabel('Frequency Band')
        axes_omega[idx].set_ylabel('Waypoint Index')
        axes_omega[idx].set_title(f'B={B} (ratio={B/K:.1%})')
        
        # 添加频段块分隔线
        n_blocks = K // B
        for i in range(1, n_blocks):
            axes_omega[idx].axvline(x=i*B - 0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Cyclic Observation Patterns for Different Bandwidths', fontsize=14)
    plt.tight_layout()
    plt.savefig('cyclic_omega_patterns.png', dpi=150)
    plt.show()
    
    # 5. 2D NMSE 矩阵: (n_B, n_M)
    nmse_matrix = np.zeros((len(B_values), len(M_values)))
    
    total_runs = len(B_values) * len(M_values)
    current_run = 0
    
    print(f"\n开始实验... (共 {total_runs} 次)")
    
    # 6. 运行实验
    for i, B in enumerate(B_values):
        # 生成完整的循环观测掩码
        Omega_full = generate_omega_cyclic(M_max, K, B, overlap=2)
        
        # 为每个 B 创建序贯 solver
        solver = II_BTD_Optimized(
            n_sources=cfg.R,
            grid_size=(cfg.N1, cfg.N2),
            mu=1.2, nu=1.5, 
            max_iter=6, 
            kernel_bandwidth=0.46, 
            warmstart=False
        )
        solver.init_sequential(data['grid_points'], bounds, K=K, I_mask=data['I_mask'])
        
        prev_M = 0
        for j, M_current in enumerate(M_values):
            current_run += 1
            
            # 增量添加新传感器
            new_locs = sorted_locs[prev_M:M_current]
            new_gamma = Gamma_sorted[prev_M:M_current, :] * Omega_full[prev_M:M_current, :]
            new_omega = Omega_full[prev_M:M_current, :]
            
            solver.add_measurements(
                new_locs, new_gamma, new_omega,
                n_outer_iter=2, max_svt_iter=20, debugFlag=False
            )
            
            nmse = solver.evaluate_reconstruction2(
                solver.Sr, solver.Phi, 
                data['S'], data['Phi'], 
                drawFlag=False
            )
            nmse_matrix[i, j] = nmse
            prev_M = M_current
            
            print(f"  [{current_run:3d}/{total_runs}] B={B:2d}, M={M_current:3d}: NMSE={nmse:.4f}")
    
    # 7. 绘制结果 - NMSE vs M (不同 B)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：NMSE vs M 曲线
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(B_values)))
    for i, B in enumerate(B_values):
        ratio = B / K
        marker = MARKERS[i % len(MARKERS)]
        axes[0].plot(M_values, nmse_matrix[i, :], 
                     marker=marker, linestyle='-',
                     color=colors[i], linewidth=2, markersize=8,
                     label=f'B={B} ({ratio:.0%})')
    
    axes[0].set_xlabel('Number of Waypoints (M)', fontsize=12)
    axes[0].set_ylabel('NMSE', fontsize=12)
    axes[0].set_title('NMSE vs Waypoints (Cyclic Observation)', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 右图：热力图
    im = axes[1].imshow(nmse_matrix, aspect='auto', cmap='RdYlGn_r')
    axes[1].set_xlabel('Number of Waypoints (M)', fontsize=12)
    axes[1].set_ylabel('Bandwidth B', fontsize=12)
    axes[1].set_title('NMSE Heatmap (Cyclic Observation)', fontsize=14)
    axes[1].set_yticks(range(len(B_values)))
    axes[1].set_yticklabels([f'B={b}' for b in B_values])
    axes[1].set_xticks(range(len(M_values)))
    axes[1].set_xticklabels(M_values)
    
    plt.colorbar(im, ax=axes[1], shrink=0.8, label='NMSE')
    
    plt.tight_layout()
    plt.savefig('nmse_cyclic_vs_M.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print(" 循环频谱切换实验完成！结果已保存:")
    print("  - cyclic_omega_patterns.png (观测模式可视化)")
    print("  - nmse_cyclic_vs_M.png (NMSE vs M 曲线 + 热力图)")
    print("="*60)
    
    return B_values, M_values, nmse_matrix


if __name__ == "__main__":

    # 选择运行的实验类型
    # 可选: 'trajectory', 'frequency_ratio', 'lfreq_ratio', 'cyclic'
    experiment_type = 'lfreq_ratio'
    experiment_type = 'frequency_ratio'
    experiment_type = 'trajectory'
    experiment_type = 'cyclic'
    M_max = 130 

    if experiment_type == 'trajectory':
        # 运行带轨迹排序的增量采样实验
        run_incremental_experiment_trajectory(
            M_max=M_max, 
            seed=42, 
            trajectory_type='snake'
        )
    elif experiment_type == 'frequency_ratio':
        # 运行频段占比 + 轨迹增量实验
        run_frequency_ratio_experiment(
            M_max=M_max,
            seed=42, 
            trajectory_type='snake'
        )
    elif experiment_type == 'lfreq_ratio':
        # 运行 l_freq-ratio 联合实验
        run_lfreq_ratio_experiment(
            M_max=M_max,
            seed=42,
            trajectory_type='snake'
        )
    elif experiment_type == 'cyclic':
        # 运行循环频谱切换实验
        run_cyclic_experiment(
            M_max=M_max,
            seed=42,
            trajectory_type='snake'
        )