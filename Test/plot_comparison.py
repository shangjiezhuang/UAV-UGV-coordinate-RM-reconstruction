"""
绘制 l_freq-ratio 实验结果对比图

根据 lfreq_ratio_results.npz 数据绘制:
1. 按策略分类的对比图
2. 按 l_freq 分类的对比图  
3. 按 ratio 分类的对比图
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义不同的 markers 和 linestyles
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]

def load_data(filepath):
    """加载实验结果数据"""
    data = np.load(filepath, allow_pickle=True)
    print("Loaded data keys:", data.files)
    
    strategies = data['omega_strategies']
    l_freq_values = data['l_freq_values']
    ratios = data['ratios']
    M_values = data['M_values']
    nmse_tensor = data['nmse_tensor']
    
    print(f"Strategies ({type(strategies)}): {strategies}")
    print(f"l_freq_values: {l_freq_values}")
    print(f"ratios: {ratios}")
    print(f"M_values: {M_values}")
    print(f"nmse_tensor shape: {nmse_tensor.shape}")
    
    return strategies, l_freq_values, ratios, M_values, nmse_tensor

def plot_by_strategy(strategies, l_freq_values, ratios, M_values, nmse_tensor, output_dir):
    """
    按策略分类绘制对比图
    每个策略一张图，展示不同 l_freq 和 ratio 组合下 NMSE vs M 的变化
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(ratios)))
    
    for s_idx, strategy in enumerate(strategies):
        fig, axes = plt.subplots(1, len(l_freq_values), figsize=(5 * len(l_freq_values), 4), sharey=True)
        if len(l_freq_values) == 1:
            axes = [axes]
        
        fig.suptitle(f'策略: {strategy}', fontsize=14, fontweight='bold')
        
        for lf_idx, l_freq in enumerate(l_freq_values):
            ax = axes[lf_idx]
            for r_idx, ratio in enumerate(ratios):
                # nmse_tensor shape: (n_strategies, n_lfreq, n_ratios, n_M)
                nmse_values = nmse_tensor[s_idx, lf_idx, r_idx, :]
                marker = MARKERS[r_idx % len(MARKERS)]
                ax.plot(M_values, nmse_values, 
                       marker=marker, linestyle='-',
                       color=colors[r_idx],
                       label=f'ratio={ratio:.2f}', 
                       linewidth=2, markersize=6)
            
            ax.set_xlabel('采样数量 M', fontsize=10)
            ax.set_ylabel('NMSE (dB)' if lf_idx == 0 else '', fontsize=10)
            ax.set_title(f'l_freq = {l_freq}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'strategy_{strategy}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

def plot_by_lfreq(strategies, l_freq_values, ratios, M_values, nmse_tensor, output_dir):
    """
    按 l_freq 分类绘制对比图
    每个 l_freq 值一张图，展示不同策略和 ratio 组合下 NMSE vs M 的变化
    """
    strategy_colors = ['#2196F3', '#FF5722', '#4CAF50']  # 蓝、橙、绿
    strategy_markers = ['o', 's', '^']  # 圆、方、三角
    
    for lf_idx, l_freq in enumerate(l_freq_values):
        fig, axes = plt.subplots(1, len(ratios), figsize=(5 * len(ratios), 4), sharey=True)
        if len(ratios) == 1:
            axes = [axes]
        
        fig.suptitle(f'频率相关长度 l_freq = {l_freq}', fontsize=14, fontweight='bold')
        
        for r_idx, ratio in enumerate(ratios):
            ax = axes[r_idx]
            for s_idx, strategy in enumerate(strategies):
                nmse_values = nmse_tensor[s_idx, lf_idx, r_idx, :]
                ax.plot(M_values, nmse_values, 
                       marker=strategy_markers[s_idx % len(strategy_markers)],
                       linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                       color=strategy_colors[s_idx % len(strategy_colors)], 
                       label=f'{strategy}', linewidth=2, markersize=6)
            
            ax.set_xlabel('采样数量 M', fontsize=10)
            ax.set_ylabel('NMSE (dB)' if r_idx == 0 else '', fontsize=10)
            ax.set_title(f'ratio = {ratio:.2f}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'lfreq_{l_freq}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

def plot_by_ratio(strategies, l_freq_values, ratios, M_values, nmse_tensor, output_dir):
    """
    按 ratio 分类绘制对比图
    每个 ratio 值一张图，展示不同策略和 l_freq 组合下 NMSE vs M 的变化
    """
    strategy_colors = ['#2196F3', '#FF5722', '#4CAF50']  # 蓝、橙、绿
    strategy_markers = ['o', 's', '^']  # 圆、方、三角
    
    for r_idx, ratio in enumerate(ratios):
        fig, axes = plt.subplots(1, len(l_freq_values), figsize=(5 * len(l_freq_values), 4), sharey=True)
        if len(l_freq_values) == 1:
            axes = [axes]
        
        fig.suptitle(f'观测频率比例 ratio = {ratio:.2f}', fontsize=14, fontweight='bold')
        
        for lf_idx, l_freq in enumerate(l_freq_values):
            ax = axes[lf_idx]
            for s_idx, strategy in enumerate(strategies):
                nmse_values = nmse_tensor[s_idx, lf_idx, r_idx, :]
                ax.plot(M_values, nmse_values, 
                       marker=strategy_markers[s_idx % len(strategy_markers)],
                       linestyle=LINESTYLES[s_idx % len(LINESTYLES)],
                       color=strategy_colors[s_idx % len(strategy_colors)], 
                       label=f'{strategy}', linewidth=2, markersize=6)
            
            ax.set_xlabel('采样数量 M', fontsize=10)
            ax.set_ylabel('NMSE (dB)' if lf_idx == 0 else '', fontsize=10)
            ax.set_title(f'l_freq = {l_freq}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'ratio_{ratio:.2f}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

def main():
    # 数据文件路径
    data_path = r'd:\Users\ASUS\Desktop\RA Work\Work1\lfreq_ratio_results.npz'
    
    # 输出目录
    output_dir = r'd:\Users\ASUS\Desktop\RA Work\Work1\code\Test\outputs\comparison_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    strategies, l_freq_values, ratios, M_values, nmse_tensor = load_data(data_path)
    
    # 分别绘制三类对比图
    print("\n--- 按策略分类绘图 ---")
    plot_by_strategy(strategies, l_freq_values, ratios, M_values, nmse_tensor, output_dir)
    
    print("\n--- 按 l_freq 分类绘图 ---")
    plot_by_lfreq(strategies, l_freq_values, ratios, M_values, nmse_tensor, output_dir)
    
    print("\n--- 按 ratio 分类绘图 ---")
    plot_by_ratio(strategies, l_freq_values, ratios, M_values, nmse_tensor, output_dir)
    
    print(f"\n所有图片已保存到: {output_dir}")

if __name__ == '__main__':
    main()
