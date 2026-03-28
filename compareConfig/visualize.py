"""
Visualization module for MAPPO training results.

Provides plotting functions for:
  1. UAV and UGV movement trajectories
  2. Reward components over training
  3. NMSE (radio map reconstruction accuracy) over training
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, List, Optional, Tuple
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _infer_grid_size(
    grid_size: Optional[Tuple[int, int]] = None,
    *,
    occupancy: Optional[np.ndarray] = None,
    trajectories: Tuple[Optional[np.ndarray], ...] = (),
    step_details: Optional[Dict[str, list]] = None,
) -> Tuple[int, int]:
    """Resolve plotting bounds from explicit config or available artifacts."""
    if grid_size is not None:
        return (max(int(grid_size[0]), 1), max(int(grid_size[1]), 1))

    if occupancy is not None:
        occ = np.asarray(occupancy)
        if occ.ndim >= 2:
            return (max(int(occ.shape[0]), 1), max(int(occ.shape[1]), 1))

    max_x = -1.0
    max_y = -1.0
    for traj in trajectories:
        if traj is None:
            continue
        arr = np.asarray(traj, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.size > 0:
            max_x = max(max_x, float(np.nanmax(arr[:, 0])))
            max_y = max(max_y, float(np.nanmax(arr[:, 1])))

    if isinstance(step_details, dict):
        target_x = np.asarray(step_details.get("target_grid_x", []), dtype=float).reshape(-1)
        target_y = np.asarray(step_details.get("target_grid_y", []), dtype=float).reshape(-1)
        if target_x.size > 0:
            valid_x = target_x[target_x >= 0.0]
            if valid_x.size > 0:
                max_x = max(max_x, float(np.nanmax(valid_x)))
        if target_y.size > 0:
            valid_y = target_y[target_y >= 0.0]
            if valid_y.size > 0:
                max_y = max(max_y, float(np.nanmax(valid_y)))

    if max_x >= 0.0 and max_y >= 0.0:
        return (max(int(np.ceil(max_x)) + 1, 1), max(int(np.ceil(max_y)) + 1, 1))

    return (51, 51)


def _grid_size_from_metrics(all_metrics: Dict[str, object]) -> Optional[Tuple[int, int]]:
    """Extract the saved scene grid size from metrics metadata when present."""
    config = all_metrics.get("config", {})
    if not isinstance(config, dict):
        return None
    scene = config.get("scene", {})
    if not isinstance(scene, dict):
        return None
    raw = scene.get("grid_size")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return (max(int(raw[0]), 1), max(int(raw[1]), 1))
        except (TypeError, ValueError):
            return None
    return None


def plot_radio_map(
    radio_map: np.ndarray,
    title: str = "Ground Truth Radio Map (Averaged over Freq)",
    save_path: str = "logs/figures/ground_truth_map.png"
):
    """
    Plot arguably the original radio map or reconstructed map.
    Args:
        radio_map: (Nx, Ny, K) array of power values.
        title: Title of the plot.
        save_path: Path to save the figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Average across frequency bands
    if radio_map.ndim == 3:
        plot_data = np.mean(radio_map, axis=2)
    else:
        plot_data = radio_map
        
    # Plot heatmap
    im = ax.imshow(plot_data.T, origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label="Power")
    
    ax.set_title(title)
    ax.set_xlabel('X Grid Index')
    ax.set_ylabel('Y Grid Index')
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved radio map plot to {save_path}")

def plot_trajectories(
    uav_trajectory: np.ndarray,
    ugv_trajectory: np.ndarray,
    grid_size: Optional[Tuple[int, int]] = None,
    occupancy: Optional[np.ndarray] = None,
    save_path: str = "logs/figures/trajectories.png",
    title: str = "UAV & UGV Trajectories",
):
    """
    Plot UAV and UGV 2D movement trajectories on the grid.

    Args:
        uav_trajectory: (T, 2) array of UAV positions.
        ugv_trajectory: (T, 2) array of UGV positions.
        grid_size: (Nx, Ny) grid dimensions.
        occupancy: (Nx, Ny) building occupancy grid (optional).
        save_path: Path to save the figure.
        title: Plot title.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid_size = _infer_grid_size(
        grid_size,
        occupancy=occupancy,
        trajectories=(uav_trajectory, ugv_trajectory),
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    def _mark_point(x: float, y: float, *, color: str, marker: str, label: str, text: str) -> None:
        ax.scatter(
            [x], [y],
            s=180,
            c=color,
            marker=marker,
            edgecolors='black',
            linewidths=1.0,
            zorder=5,
            label=label,
        )
        ax.annotate(
            text,
            (x, y),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, alpha=0.85),
        )

    # Draw buildings if provided
    if occupancy is not None:
        ax.imshow(
            occupancy.T, origin='lower', cmap='Greys', alpha=0.3,
            extent=[0, grid_size[0], 0, grid_size[1]],
        )

    # UAV trajectory
    ax.plot(
        uav_trajectory[:, 0], uav_trajectory[:, 1],
        color='#1f77b4', marker='o', markersize=3, linewidth=1.8, alpha=0.75, label='UAV Path',
    )
    _mark_point(
        uav_trajectory[0, 0], uav_trajectory[0, 1],
        color='#1f77b4', marker='*', label='UAV Start', text='UAV S',
    )
    _mark_point(
        uav_trajectory[-1, 0], uav_trajectory[-1, 1],
        color='#1f77b4', marker='X', label='UAV End', text='UAV E',
    )

    # UGV trajectory
    ax.plot(
        ugv_trajectory[:, 0], ugv_trajectory[:, 1],
        color='#d62728', marker='s', markersize=3, linewidth=1.8, linestyle='--', alpha=0.75, label='UGV Path',
    )
    _mark_point(
        ugv_trajectory[0, 0], ugv_trajectory[0, 1],
        color='#d62728', marker='D', label='UGV Start', text='UGV S',
    )
    _mark_point(
        ugv_trajectory[-1, 0], ugv_trajectory[-1, 1],
        color='#d62728', marker='P', label='UGV End', text='UGV E',
    )

    ax.set_xlim(-1, grid_size[0])
    ax.set_ylim(-1, grid_size[1])
    ax.set_xlabel('X Grid Index')
    ax.set_ylabel('Y Grid Index')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved trajectory plot to {save_path}")


def plot_trajectory_gif(
    uav_trajectory: np.ndarray,
    ugv_trajectory: np.ndarray,
    grid_size: Optional[Tuple[int, int]] = None,
    occupancy: Optional[np.ndarray] = None,
    save_path: str = "logs/figures/trajectories.gif",
    title: str = "UAV & UGV Trajectories",
    fps: int = 4,
):
    """
    Save an animated GIF showing both trajectories step by step.

    Args:
        uav_trajectory: (T, 2) array of UAV positions.
        ugv_trajectory: (T, 2) array of UGV positions.
        grid_size: (Nx, Ny) grid dimensions.
        occupancy: (Nx, Ny) building occupancy grid (optional).
        save_path: Path to save the gif.
        title: Plot title.
        fps: Frames per second for the exported GIF.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    uav_trajectory = np.asarray(uav_trajectory, dtype=float)
    ugv_trajectory = np.asarray(ugv_trajectory, dtype=float)
    grid_size = _infer_grid_size(
        grid_size,
        occupancy=occupancy,
        trajectories=(uav_trajectory, ugv_trajectory),
    )
    frame_count = int(min(len(uav_trajectory), len(ugv_trajectory)))
    if frame_count <= 0:
        print("[Visualize] Empty trajectory, skip GIF generation.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if occupancy is not None:
        ax.imshow(
            occupancy.T, origin='lower', cmap='Greys', alpha=0.3,
            extent=[0, grid_size[0], 0, grid_size[1]],
        )

    ax.set_xlim(-1, grid_size[0])
    ax.set_ylim(-1, grid_size[1])
    ax.set_xlabel('X Grid Index')
    ax.set_ylabel('Y Grid Index')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax.scatter(
        [uav_trajectory[0, 0]], [uav_trajectory[0, 1]],
        s=180, c='#1f77b4', marker='*', edgecolors='black', linewidths=1.0, zorder=5, label='UAV Start',
    )
    ax.scatter(
        [ugv_trajectory[0, 0]], [ugv_trajectory[0, 1]],
        s=180, c='#d62728', marker='D', edgecolors='black', linewidths=1.0, zorder=5, label='UGV Start',
    )
    ax.scatter(
        [uav_trajectory[-1, 0]], [uav_trajectory[-1, 1]],
        s=140, facecolors='none', edgecolors='#1f77b4', marker='X', linewidths=1.5, zorder=4, label='UAV End',
    )
    ax.scatter(
        [ugv_trajectory[-1, 0]], [ugv_trajectory[-1, 1]],
        s=140, facecolors='none', edgecolors='#d62728', marker='P', linewidths=1.5, zorder=4, label='UGV End',
    )

    uav_line, = ax.plot(
        [], [], color='#1f77b4', marker='o', markersize=3, linewidth=1.8, alpha=0.75, label='UAV Path',
    )
    ugv_line, = ax.plot(
        [], [], color='#d62728', marker='s', markersize=3, linewidth=1.8, linestyle='--', alpha=0.75, label='UGV Path',
    )
    uav_curr, = ax.plot(
        [], [], color='#1f77b4', marker='o', markersize=10, linestyle='None', markeredgecolor='black',
        label='UAV Current',
    )
    ugv_curr, = ax.plot(
        [], [], color='#d62728', marker='s', markersize=10, linestyle='None', markeredgecolor='black',
        label='UGV Current',
    )
    step_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.9),
    )
    status_text = ax.text(
        0.02, 0.92, '', transform=ax.transAxes, va='top', ha='left',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='lightgray', alpha=0.85),
    )

    ax.legend(loc='upper right', fontsize=9)

    def _update(frame_idx: int):
        end_idx = int(frame_idx) + 1
        uav_line.set_data(uav_trajectory[:end_idx, 0], uav_trajectory[:end_idx, 1])
        ugv_line.set_data(ugv_trajectory[:end_idx, 0], ugv_trajectory[:end_idx, 1])
        uav_curr.set_data([uav_trajectory[frame_idx, 0]], [uav_trajectory[frame_idx, 1]])
        ugv_curr.set_data([ugv_trajectory[frame_idx, 0]], [ugv_trajectory[frame_idx, 1]])
        step_text.set_text(f"{title}\nStep {frame_idx}/{frame_count - 1}")
        dist = float(np.linalg.norm(uav_trajectory[frame_idx] - ugv_trajectory[frame_idx]))
        status_text.set_text(
            f"UAV=({uav_trajectory[frame_idx, 0]:.0f}, {uav_trajectory[frame_idx, 1]:.0f})  "
            f"UGV=({ugv_trajectory[frame_idx, 0]:.0f}, {ugv_trajectory[frame_idx, 1]:.0f})\n"
            f"Distance={dist:.2f}"
        )
        return uav_line, ugv_line, uav_curr, ugv_curr, step_text, status_text

    anim = FuncAnimation(
        fig,
        _update,
        frames=frame_count,
        interval=max(int(1000 / max(fps, 1)), 1),
        blit=False,
    )

    plt.tight_layout()
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[Visualize] Saved trajectory GIF to {save_path}")


def _prepare_planner_step_data(
    uav_trajectory: np.ndarray,
    ugv_trajectory: np.ndarray,
    step_details: Dict[str, list],
):
    if not isinstance(step_details, dict):
        return None

    target_x = np.asarray(step_details.get("target_grid_x", []), dtype=float).reshape(-1)
    target_y = np.asarray(step_details.get("target_grid_y", []), dtype=float).reshape(-1)
    target_f = np.asarray(
        step_details.get("target_center_freq", step_details.get("target_freq", [])),
        dtype=float,
    ).reshape(-1)
    if target_x.size == 0 or target_y.size == 0 or target_f.size == 0:
        return None

    uav_trajectory = np.asarray(uav_trajectory, dtype=float)
    ugv_trajectory = np.asarray(ugv_trajectory, dtype=float)
    frame_count = int(min(len(uav_trajectory) - 1, len(ugv_trajectory) - 1, target_x.size, target_y.size, target_f.size))
    if frame_count <= 0:
        return None

    uav_steps = uav_trajectory[1:frame_count + 1]
    ugv_steps = ugv_trajectory[1:frame_count + 1]
    planner_xyz = np.stack([target_x[:frame_count], target_y[:frame_count], target_f[:frame_count]], axis=1)
    planner_valid = (
        (planner_xyz[:, 0] >= 0.0)
        & (planner_xyz[:, 1] >= 0.0)
        & (planner_xyz[:, 2] >= 0.0)
    )
    return uav_steps, ugv_steps, planner_xyz, planner_valid


def _get_step_series(
    step_details: Dict[str, list],
    frame_count: int,
    key: str,
    *,
    fallback_keys: tuple = (),
    fill_value: float = np.nan,
) -> np.ndarray:
    arr = np.asarray([], dtype=float)
    for candidate in (key,) + tuple(fallback_keys):
        values = step_details.get(candidate, [])
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size > 0:
            break

    if frame_count <= 0:
        return np.asarray([], dtype=float)
    if arr.size >= frame_count:
        return arr[:frame_count]

    padded = np.full(frame_count, fill_value, dtype=float)
    if arr.size > 0:
        padded[:arr.size] = arr
    return padded


def _spread_marker_points(points: Dict[str, np.ndarray], radius: float = 0.35) -> Dict[str, np.ndarray]:
    spread_points = {
        name: np.asarray(point, dtype=float).copy()
        for name, point in points.items()
    }
    grouped = {}
    for name, point in spread_points.items():
        xy_key = tuple(np.round(point[:2], 6).tolist())
        grouped.setdefault(xy_key, []).append(name)

    for names in grouped.values():
        if len(names) <= 1:
            continue
        angles = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, len(names), endpoint=False)
        offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        for name, offset in zip(names, offsets):
            spread_points[name][:2] += offset
    return spread_points


def plot_planner_trajectory_3d(
    uav_trajectory: np.ndarray,
    ugv_trajectory: np.ndarray,
    step_details: Dict[str, list],
    grid_size: Optional[Tuple[int, int]] = None,
    save_path: str = "logs/figures/trajectories_3d.png",
    title: str = "UAV / UGV / Planner 3D Trajectory",
):
    prepared = _prepare_planner_step_data(uav_trajectory, ugv_trajectory, step_details)
    if prepared is None:
        print("[Visualize] Missing planner (x, y, f) step data, skip 3D trajectory plot.")
        return

    uav_steps, ugv_steps, planner_xyz, planner_valid = prepared
    grid_size = _infer_grid_size(
        grid_size,
        trajectories=(uav_steps, ugv_steps),
        step_details=step_details,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    zeros = np.zeros(uav_steps.shape[0], dtype=float)
    ax.plot(
        uav_steps[:, 0], uav_steps[:, 1], zeros,
        color='#1f77b4', linewidth=2.0, marker='o', markersize=3, alpha=0.8, label='UAV Path',
    )
    ax.plot(
        ugv_steps[:, 0], ugv_steps[:, 1], zeros,
        color='#d62728', linewidth=2.0, linestyle='--', marker='s', markersize=3, alpha=0.8, label='UGV Path',
    )

    if planner_valid.any():
        valid_idx = np.where(planner_valid)[0]
        scatter = ax.scatter(
            planner_xyz[planner_valid, 0],
            planner_xyz[planner_valid, 1],
            planner_xyz[planner_valid, 2],
            c=valid_idx,
            cmap='viridis',
            s=26,
            alpha=0.95,
            label='Planner (x,y,f)',
        )
        ax.plot(
            planner_xyz[planner_valid, 0],
            planner_xyz[planner_valid, 1],
            planner_xyz[planner_valid, 2],
            color='#2ca02c',
            linewidth=1.2,
            alpha=0.4,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.08, fraction=0.03)
        cbar.set_label('Step')

    ax.scatter([uav_steps[0, 0]], [uav_steps[0, 1]], [0.0], s=180, c='#1f77b4', marker='*', edgecolors='black', label='UAV Start')
    ax.scatter([uav_steps[-1, 0]], [uav_steps[-1, 1]], [0.0], s=140, c='#1f77b4', marker='X', edgecolors='black', label='UAV End')
    ax.scatter([ugv_steps[0, 0]], [ugv_steps[0, 1]], [0.0], s=180, c='#d62728', marker='D', edgecolors='black', label='UGV Start')
    ax.scatter([ugv_steps[-1, 0]], [ugv_steps[-1, 1]], [0.0], s=140, c='#d62728', marker='P', edgecolors='black', label='UGV End')

    if planner_valid.any():
        first_valid = planner_xyz[np.where(planner_valid)[0][0]]
        last_valid = planner_xyz[np.where(planner_valid)[0][-1]]
        ax.scatter([first_valid[0]], [first_valid[1]], [first_valid[2]], s=140, c='#2ca02c', marker='^', edgecolors='black', label='Planner Start')
        ax.scatter([last_valid[0]], [last_valid[1]], [last_valid[2]], s=120, c='#2ca02c', marker='v', edgecolors='black', label='Planner End')

    planner_max_f = float(np.max(planner_xyz[planner_valid, 2])) if planner_valid.any() else 1.0
    ax.set_xlim(-1, grid_size[0])
    ax.set_ylim(-1, grid_size[1])
    ax.set_zlim(-0.5, max(planner_max_f + 1.0, 1.0))
    ax.set_xlabel('X Grid Index')
    ax.set_ylabel('Y Grid Index')
    ax.set_zlabel('Planner Freq Band')
    ax.set_title(title)
    ax.view_init(elev=28, azim=-58)
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 0.98),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
    )

    fig.subplots_adjust(left=0.04, right=0.80, top=0.90, bottom=0.08)
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[Visualize] Saved 3D trajectory plot to {save_path}")


def plot_planner_trajectory_3d_gif(
    uav_trajectory: np.ndarray,
    ugv_trajectory: np.ndarray,
    step_details: Dict[str, list],
    grid_size: Optional[Tuple[int, int]] = None,
    save_path: str = "logs/figures/trajectories_3d.gif",
    title: str = "UAV / UGV / Planner 3D Trajectory",
    fps: int = 4,
):
    prepared = _prepare_planner_step_data(uav_trajectory, ugv_trajectory, step_details)
    if prepared is None:
        print("[Visualize] Missing planner (x, y, f) step data, skip 3D trajectory GIF.")
        return

    uav_steps, ugv_steps, planner_xyz, planner_valid = prepared
    grid_size = _infer_grid_size(
        grid_size,
        trajectories=(uav_steps, ugv_steps),
        step_details=step_details,
    )
    frame_count = uav_steps.shape[0]
    sample_center_freq = _get_step_series(
        step_details,
        frame_count,
        "sample_center_freq",
        fallback_keys=("sensing_ind",),
        fill_value=-1.0,
    )
    target_center_freq = _get_step_series(
        step_details,
        frame_count,
        "target_center_freq",
        fallback_keys=("target_freq",),
        fill_value=-1.0,
    )
    bw_ratio = _get_step_series(step_details, frame_count, "bw_ratio", fill_value=np.nan)
    sensing_band_num = _get_step_series(step_details, frame_count, "sensing_band_num", fill_value=-1.0)
    sensing_bw_units = _get_step_series(step_details, frame_count, "sensing_bw_units", fill_value=-1.0)
    comm_bw_units = _get_step_series(step_details, frame_count, "comm_bw_units", fill_value=-1.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_subplot(111, projection='3d')
    planner_max_f = float(np.max(planner_xyz[planner_valid, 2])) if planner_valid.any() else 1.0

    ax.set_xlim(-1, grid_size[0])
    ax.set_ylim(-1, grid_size[1])
    ax.set_zlim(-0.5, max(planner_max_f + 1.0, 1.0))
    ax.set_xlabel('X Grid Index')
    ax.set_ylabel('Y Grid Index')
    ax.set_zlabel('Planner Freq Band')
    ax.set_title(title)
    ax.view_init(elev=28, azim=-58)
    fig.subplots_adjust(left=0.05, right=0.76, top=0.90, bottom=0.18)

    start_markers = _spread_marker_points(
        {
            "uav_start": np.array([uav_steps[0, 0], uav_steps[0, 1], 0.0], dtype=float),
            "ugv_start": np.array([ugv_steps[0, 0], ugv_steps[0, 1], 0.0], dtype=float),
        },
        radius=0.32,
    )
    uav_start = start_markers["uav_start"]
    ugv_start = start_markers["ugv_start"]
    ax.scatter([uav_start[0]], [uav_start[1]], [uav_start[2]], s=180, c='#1f77b4', marker='*', edgecolors='black', label='UAV Start')
    ax.scatter([ugv_start[0]], [ugv_start[1]], [ugv_start[2]], s=180, c='#d62728', marker='D', edgecolors='black', label='UGV Start')

    uav_line, = ax.plot([], [], [], color='#1f77b4', linewidth=2.0, marker='o', markersize=3, alpha=0.8, label='UAV Path')
    ugv_line, = ax.plot([], [], [], color='#d62728', linewidth=2.0, linestyle='--', marker='s', markersize=3, alpha=0.8, label='UGV Path')
    planner_line, = ax.plot([], [], [], color='#2ca02c', linewidth=1.8, alpha=0.9, label='Planner (x,y,f)')
    planner_proj, = ax.plot([], [], [], color='#2ca02c', linestyle=':', linewidth=1.2, alpha=0.9)

    uav_curr, = ax.plot([], [], [], color='#1f77b4', marker='o', markersize=9, linestyle='None', markeredgecolor='black', label='UAV Current')
    ugv_curr, = ax.plot([], [], [], color='#d62728', marker='s', markersize=9, linestyle='None', markeredgecolor='black', label='UGV Current')
    planner_curr, = ax.plot([], [], [], color='#2ca02c', marker='^', markersize=10, linestyle='None', markeredgecolor='black', label='Planner Current')

    step_text = fig.text(
        0.05, 0.94, '', va='top', ha='left',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.9),
    )
    status_text = fig.text(
        0.05, 0.09, '', va='bottom', ha='left',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='lightgray', alpha=0.85),
    )
    sensing_text = fig.text(
        0.05, 0.03, '', va='bottom', ha='left',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='lightgray', alpha=0.85),
    )

    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 0.98),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
    )

    def _set_point(artist, point):
        artist.set_data([point[0]], [point[1]])
        artist.set_3d_properties([point[2]])

    def _clear_artist(artist):
        artist.set_data([], [])
        artist.set_3d_properties([])

    def _format_int(value: float) -> str:
        if np.isnan(value) or value < 0:
            return "-"
        return f"{int(value):d}"

    def _format_ratio(value: float) -> str:
        if np.isnan(value):
            return "-"
        return f"{value:.2f}"

    def _update(frame_idx: int):
        end_idx = int(frame_idx) + 1
        zeros = np.zeros(end_idx, dtype=float)
        uav_line.set_data(uav_steps[:end_idx, 0], uav_steps[:end_idx, 1])
        uav_line.set_3d_properties(zeros)
        ugv_line.set_data(ugv_steps[:end_idx, 0], ugv_steps[:end_idx, 1])
        ugv_line.set_3d_properties(zeros)

        if frame_idx == 0:
            _clear_artist(uav_curr)
            _clear_artist(ugv_curr)
        else:
            current_markers = _spread_marker_points(
                {
                    "uav_curr": np.array([uav_steps[frame_idx, 0], uav_steps[frame_idx, 1], 0.0], dtype=float),
                    "ugv_curr": np.array([ugv_steps[frame_idx, 0], ugv_steps[frame_idx, 1], 0.0], dtype=float),
                },
                radius=0.24,
            )
            _set_point(uav_curr, current_markers["uav_curr"])
            _set_point(ugv_curr, current_markers["ugv_curr"])

        valid_history = planner_valid[:end_idx]
        if valid_history.any():
            hist = planner_xyz[:end_idx][valid_history]
            planner_line.set_data(hist[:, 0], hist[:, 1])
            planner_line.set_3d_properties(hist[:, 2])
        else:
            _clear_artist(planner_line)

        planner_tuple = "Planner=(-1, -1, -1)"
        if planner_valid[frame_idx]:
            planner_point = planner_xyz[frame_idx]
            _set_point(planner_curr, planner_point)
            planner_proj.set_data([planner_point[0], planner_point[0]], [planner_point[1], planner_point[1]])
            planner_proj.set_3d_properties([0.0, planner_point[2]])
            planner_tuple = f"Planner=({planner_point[0]:.0f}, {planner_point[1]:.0f}, {planner_point[2]:.0f})"
        else:
            _clear_artist(planner_curr)
            _clear_artist(planner_proj)

        dist = float(np.linalg.norm(uav_steps[frame_idx] - ugv_steps[frame_idx]))
        step_text.set_text(f"Step {frame_idx + 1}/{uav_steps.shape[0]}")
        status_text.set_text(
            f"UAV=({uav_steps[frame_idx, 0]:.0f}, {uav_steps[frame_idx, 1]:.0f}, 0)  "
            f"UGV=({ugv_steps[frame_idx, 0]:.0f}, {ugv_steps[frame_idx, 1]:.0f}, 0)  Dist={dist:.2f}\n"
            f"{planner_tuple}"
        )
        sensing_text.set_text(
            f"Sample CF={_format_int(sample_center_freq[frame_idx])}  "
            f"Planner CF={_format_int(target_center_freq[frame_idx])}  "
            f"Ratio={_format_ratio(bw_ratio[frame_idx])}\n"
            f"SenseBands={_format_int(sensing_band_num[frame_idx])}  "
            f"BW Units S/C={_format_int(sensing_bw_units[frame_idx])}/{_format_int(comm_bw_units[frame_idx])}  "
            f"Frame={frame_idx + 1}"
        )
        return (
            uav_line, ugv_line, planner_line, planner_proj,
            uav_curr, ugv_curr, planner_curr, step_text, status_text, sensing_text,
        )

    anim = FuncAnimation(
        fig,
        _update,
        frames=uav_steps.shape[0],
        interval=max(int(1000 / max(fps, 1)), 1),
        blit=False,
    )

    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[Visualize] Saved 3D trajectory GIF to {save_path}")


def plot_reward_components(
    metrics: Dict[str, List[float]],
    save_path: str = "logs/figures/reward_components.png",
    window: int = 10,
):
    """
    Plot reward components over training episodes.

    Args:
        metrics: Dict with per-episode reward component series.
        save_path: Path to save the figure.
        window: Smoothing window size.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plot_specs = [
        ('r_nmse', 'NMSE Reward', '#16a085'),
        ('r_unc', 'Uncertainty Reward', '#2ecc71'),
        ('r_new_freq', 'New-Frequency Reward', '#1abc9c'),
        ('r_new_spatial', 'New-Spatial Reward', '#27ae60'),
        ('r_tx', 'TX Reward', '#3498db'),
        ('r_queue', 'Queue Penalty', '#e74c3c'),
        ('r_progress', 'Progress Reward', '#f39c12'),
        ('r_uav_progress', 'UAV Progress Reward', '#f1c40f'),
        ('r_ugv_progress', 'UGV Progress Reward', '#e67e22'),
        ('r_goal_arrival', 'Goal-Arrival Reward', '#b9770e'),
        ('r_revisit', 'Revisit Penalty', '#d35400'),
        ('r_terminal', 'Terminal Reward', '#8e44ad'),
    ]
    total_reward_keys = [
        'r_nmse',
        'r_unc',
        'r_new_freq',
        'r_new_spatial',
        'r_tx',
        'r_queue',
        'r_progress',
        'r_revisit',
        'r_terminal',
    ]

    num_cols = 2
    num_rows = int(np.ceil(len(plot_specs) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 4.2 * num_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (key, label, color) in enumerate(plot_specs):
        ax = axes[idx]
        if key in metrics and len(metrics[key]) > 0:
            data = np.array(metrics[key])
            # Smooth
            if len(data) > window:
                smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
                ax.plot(smoothed, color=color, linewidth=1.5, label=f'{label} (smoothed)')
                ax.plot(data, color=color, alpha=0.2, linewidth=0.5)
            else:
                ax.plot(data, color=color, linewidth=1.5, label=label)
            ax.set_title(label)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {key}', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_title(label)

    for ax in axes[len(plot_specs):]:
        ax.axis('off')

    plt.suptitle('Reward Components Over Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved reward components plot to {save_path}")

    # Also plot total reward
    all_present = all(k in metrics and len(metrics[k]) > 0 for k in total_reward_keys)
    if all_present:
        min_len = min(len(metrics[k]) for k in total_reward_keys)
        total = sum(np.array(metrics[k][:min_len]) for k in total_reward_keys)

        fig, ax = plt.subplots(figsize=(10, 5))
        if len(total) > window:
            smoothed = np.convolve(total, np.ones(window) / window, mode='valid')
            ax.plot(smoothed, color='#3498db', linewidth=2, label='Total Reward (smoothed)')
            ax.plot(total, color='#3498db', alpha=0.2, linewidth=0.5)
        else:
            ax.plot(total, color='#3498db', linewidth=2, label='Total Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Total Reward Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        total_path = save_path.replace('reward_components', 'total_reward')
        plt.savefig(total_path, dpi=150)
        plt.close()
        print(f"[Visualize] Saved total reward plot to {total_path}")


def plot_nmse_diagnostics(
    metrics: Dict[str, List[float]],
    save_path: str = "logs/figures/nmse_diagnostics.png",
    window: int = 10,
):
    """Plot episode-level target-gap diagnostics without adding them to reward."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metric_specs = [
        ("nmse_target_gap", "Final Target Gap / Episode", "#5d6d7e", "NMSE Gap"),
        (
            "target_gap_penalty_diag",
            "Cumulative Target-Gap Penalty (diag)",
            "#7f8c8d",
            "Diagnostic Penalty",
        ),
    ]

    has_any = False
    for key, _, _, _ in metric_specs:
        values = metrics.get(key, [])
        if isinstance(values, list) and len(values) > 0:
            has_any = True
            break
    if not has_any:
        print("[Visualize] No NMSE diagnostic series found, skip diagnostics plot.")
        return

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(14, 4.8))
    axes = np.atleast_1d(axes).flatten()

    for idx, (key, title, color, ylabel) in enumerate(metric_specs):
        ax = axes[idx]
        values = metrics.get(key, [])
        if isinstance(values, list) and len(values) > 0:
            data = np.asarray(values, dtype=float)
            if len(data) > window:
                smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
                ax.plot(smoothed, color=color, linewidth=1.8, label=f'{title} (smoothed)')
                ax.plot(data, color=color, alpha=0.22, linewidth=0.6)
            else:
                ax.plot(data, color=color, linewidth=1.8, label=title)
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, f'No data for {key}', transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.suptitle('NMSE Target-Gap Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved NMSE diagnostics plot to {save_path}")


def plot_nmse_curve(
    nmse_values: List[float],
    save_path: str = "logs/figures/nmse_curve.png",
    window: int = 10,
    target_nmse: Optional[float] = None,
):
    """
    Plot NMSE over training episodes.

    Args:
        nmse_values: List of NMSE values per episode.
        save_path: Path to save the figure.
        window: Smoothing window size.
        target_nmse: Target NMSE threshold (horizontal line).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    data = np.array(nmse_values)
    if len(data) > window:
        smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
        ax.plot(smoothed, color='#e74c3c', linewidth=2, label='NMSE (smoothed)')
        ax.plot(data, color='#e74c3c', alpha=0.2, linewidth=0.5)
    else:
        ax.plot(data, color='#e74c3c', linewidth=2, label='NMSE')

    if target_nmse is not None:
        ax.axhline(y=target_nmse, color='green', linestyle='--',
                   linewidth=1.5, label=f'Target NMSE = {target_nmse}')

    ax.set_xlabel('Episode')
    ax.set_ylabel('NMSE')
    ax.set_title('Radio Map NMSE Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved NMSE curve to {save_path}")


def plot_eval_metrics(
    training_metrics: Dict[str, List[float]],
    save_path: str = "logs/figures/eval_metrics.png",
):
    """
    Plot key evaluation metrics over evaluation rounds.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metric_specs = [
        ("eval_mean_return", "Eval Mean Return", "#2ecc71"),
        ("eval_mean_nmse", "Eval Mean NMSE", "#e74c3c"),
        ("eval_mean_steps", "Eval Mean Steps", "#3498db"),
        ("eval_mean_uav_energy_remaining", "Eval Mean UAV Energy", "#9b59b6"),
        ("eval_mean_r_nmse", "Eval Mean r_nmse", "#16a085"),
        ("eval_mean_r_unc", "Eval Mean r_unc", "#16a085"),
        ("eval_mean_r_new_freq", "Eval Mean r_new_freq", "#1abc9c"),
        ("eval_mean_r_new_spatial", "Eval Mean r_new_spatial", "#27ae60"),
        ("eval_mean_r_tx", "Eval Mean r_tx", "#2980b9"),
        ("eval_mean_r_queue", "Eval Mean r_queue", "#c0392b"),
        ("eval_mean_r_progress", "Eval Mean r_progress", "#f39c12"),
        ("eval_mean_r_uav_progress", "Eval Mean r_uav_progress", "#f1c40f"),
        ("eval_mean_r_ugv_progress", "Eval Mean r_ugv_progress", "#e67e22"),
        ("eval_mean_r_goal_arrival", "Eval Mean r_goal_arrival", "#b9770e"),
        ("eval_mean_r_revisit", "Eval Mean r_revisit", "#d35400"),
        ("eval_mean_r_terminal", "Eval Mean r_terminal", "#8e44ad"),
        ("eval_mean_nmse_target_gap", "Eval Mean Target Gap", "#5d6d7e"),
        ("eval_mean_target_gap_penalty_diag", "Eval Mean Gap Penalty (diag)", "#7f8c8d"),
        ("eval_mean_data_delivered_bits", "Eval Mean Delivered Bits", "#34495e"),
        ("eval_mean_novel_data_delivered_bits", "Eval Mean Novel Bits", "#7f8c8d"),
        ("eval_mean_uav_move_dist", "Eval Mean UAV Move Dist", "#2980b9"),
        ("eval_mean_ugv_move_dist", "Eval Mean UGV Move Dist", "#8e44ad"),
    ]

    has_any = False
    for key, _, _ in metric_specs:
        values = training_metrics.get(key, [])
        if isinstance(values, list) and len(values) > 0:
            has_any = True
            break
    if not has_any:
        print("[Visualize] No evaluation metric series found, skip eval metrics plot.")
        return

    num_cols = 4
    num_rows = int(np.ceil(len(metric_specs) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (key, label, color) in enumerate(metric_specs):
        ax = axes[idx]
        values = training_metrics.get(key, [])
        if isinstance(values, list) and len(values) > 0:
            data = np.asarray(values, dtype=float)
            x = np.arange(1, len(data) + 1)
            ax.plot(x, data, color=color, linewidth=1.8, marker='o', markersize=2.5)
            ax.set_xlabel('Eval Round')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {key}', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(label)

    for ax in axes[len(metric_specs):]:
        ax.axis('off')

    plt.suptitle('Validation Metrics Over Evaluation Rounds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved eval metrics plot to {save_path}")


def plot_episode_resource_metrics(
    metrics: Dict[str, List[float]],
    save_path: str = "logs/figures/episode_resource_metrics.png",
    window: int = 10,
):
    """Plot episode-level transmission and movement statistics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metric_specs = [
        ("data_delivered_bits", "Delivered Bits / Episode", "#34495e", 1e-9, "Gbits"),
        ("novel_data_delivered_bits", "Novel Delivered Bits / Episode", "#7f8c8d", 1e-9, "Gbits"),
        ("uav_move_dist", "UAV Move Distance / Episode", "#2980b9", 1.0, "Grid Distance"),
        ("ugv_move_dist", "UGV Move Distance / Episode", "#8e44ad", 1.0, "Grid Distance"),
    ]

    has_any = False
    for key, _, _, _, _ in metric_specs:
        values = metrics.get(key, [])
        if isinstance(values, list) and len(values) > 0:
            has_any = True
            break
    if not has_any:
        print("[Visualize] No episode resource series found, skip resource metrics plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = np.atleast_1d(axes).flatten()

    for idx, (key, title, color, scale, ylabel) in enumerate(metric_specs):
        ax = axes[idx]
        values = metrics.get(key, [])
        if isinstance(values, list) and len(values) > 0:
            data = np.asarray(values, dtype=float) * float(scale)
            if len(data) > window:
                smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
                ax.plot(smoothed, color=color, linewidth=1.8, label=f'{title} (smoothed)')
                ax.plot(data, color=color, alpha=0.2, linewidth=0.6)
            else:
                ax.plot(data, color=color, linewidth=1.8, label=title)
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, f'No data for {key}', transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Episode Transmission / Movement Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved episode resource metrics plot to {save_path}")


def _step_reward_specs() -> List[Tuple[str, str, str, object, str]]:
    return [
        ("r_nmse", "NMSE", '#16a085', '-', 'o'),
        ("r_unc", "Uncertainty", '#2ecc71', '--', 's'),
        ("r_new_freq", "New Freq", '#1abc9c', '-.', '^'),
        ("r_new_spatial", "New Spatial", '#27ae60', ':', 'D'),
        ("r_tx", "TX", '#3498db', (0, (5, 2)), 'P'),
        ("r_queue", "Queue", '#e74c3c', (0, (3, 1, 1, 1)), 'X'),
        ("r_progress", "Progress", '#f39c12', (0, (7, 2)), 'v'),
        ("r_uav_progress", "UAV Progress", '#f1c40f', (0, (5, 1)), '^'),
        ("r_ugv_progress", "UGV Progress", '#e67e22', (0, (2, 1)), 'd'),
        ("r_goal_arrival", "Goal Arrival", '#b9770e', (0, (1, 1)), '*'),
        ("r_revisit", "Revisit", '#d35400', (0, (2, 1)), '<'),
        ("r_terminal", "Terminal", '#8e44ad', (0, (4, 1, 1, 1)), '>'),
    ]


def plot_eval_step_reward_components(
    step_details: Dict[str, list],
    save_path: str = "logs/figures/eval_step_rewards.png",
    title: str = "Evaluation Episode - Per-Step Reward Components",
):
    """Plot each reward component on its own subplot for easier comparison."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not isinstance(step_details, dict):
        print("[Visualize] Invalid eval step details format, skip reward-components plot.")
        return

    reward_specs = _step_reward_specs()
    num_cols = 3
    num_rows = int(np.ceil(len(reward_specs) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 11))
    axes = np.atleast_1d(axes).flatten()

    def _series(key: str) -> np.ndarray:
        values = step_details.get(key, [])
        return np.asarray(values, dtype=float).reshape(-1)

    for idx, (key, label, color, linestyle, marker) in enumerate(reward_specs):
        ax = axes[idx]
        data = _series(key)
        if data.size > 0:
            steps = np.arange(1, data.size + 1)
            markevery = max(1, data.size // 12)
            ax.plot(
                steps,
                data,
                color=color,
                linewidth=1.8,
                alpha=0.92,
                linestyle=linestyle,
                marker=marker,
                markersize=4.4,
                markerfacecolor='white',
                markeredgewidth=0.9,
                markevery=markevery,
            )
            ax.axhline(0.0, color='#7f8c8d', linewidth=0.9, alpha=0.6)
        else:
            ax.text(0.5, 0.5, f'No data for {key}', transform=ax.transAxes,
                    ha='center', va='center')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)

    for ax in axes[len(reward_specs):]:
        ax.axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved eval step reward components to {save_path}")


def plot_eval_step_details(
    step_details: Dict[str, list],
    save_path: str = "logs/figures/eval_step_details.png",
    title: str = "Evaluation Episode - Per-Step Details",
):
    """
    Plot per-step evaluation details from the last eval episode.

    Shows: queue length, SNR, bandwidth ratio, sensing index,
           UAV-UGV distance, and per-step rewards.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not isinstance(step_details, dict):
        print("[Visualize] Invalid eval step details format, skip.")
        return

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))

    def _series(key: str) -> np.ndarray:
        values = step_details.get(key, [])
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr

    def _step_axis(size: int) -> np.ndarray:
        return np.arange(1, size + 1)

    # 1. Queue size
    ax = axes[0, 0]
    data = _series("queue_size")
    if data.size > 0:
        steps = _step_axis(data.size)
        ax.plot(steps, data, 'b-', linewidth=1.5)
        ax.fill_between(steps, 0, data, alpha=0.2)
    ax.set_title('UAV Data Queue Length', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Queue Size (packets)')
    ax.grid(True, alpha=0.3)

    # 2. SNR
    ax = axes[0, 1]
    data = _series("snr_db")
    if data.size > 0:
        steps = _step_axis(data.size)
        ax.plot(steps, data, 'g-', linewidth=1.5)
    ax.set_title('Channel SNR (UAV→UGV)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('SNR (dB)')
    ax.grid(True, alpha=0.3)

    # 3. Bandwidth ratio
    ax = axes[1, 0]
    data = _series("bw_ratio")
    if data.size > 0:
        steps = _step_axis(data.size)
        ax.step(steps, data, 'r-', linewidth=1.5, where='mid')
        ax.set_ylim(-0.05, 1.05)
    ax.set_title('UAV Bandwidth Allocation Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Sensing / Total BW')
    ax.grid(True, alpha=0.3)

    # 4. Sensing index (observation center)
    ax = axes[1, 1]
    data = _series("sensing_ind")
    if data.size > 0:
        steps = _step_axis(data.size)
        ax.scatter(steps, data, c='purple', s=15, alpha=0.7)
        ax.plot(steps, data, 'purple', linewidth=0.5, alpha=0.3)
    ax.set_title('Sensing Center (Freq Band Index)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Freq Band Index (0~K-1)')
    ax.grid(True, alpha=0.3)

    # 5. UAV-UGV distance
    ax = axes[2, 0]
    data = _series("uav_ugv_dist")
    if data.size > 0:
        steps = _step_axis(data.size)
        ax.plot(steps, data, 'm-', linewidth=1.5)
        ax.fill_between(steps, 0, data, alpha=0.15, color='magenta')
    ax.set_title('UAV-UGV Distance (grid cells)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3)

    # 6. Per-step rewards
    ax = axes[2, 1]
    reward_keys = _step_reward_specs()
    has_reward = False
    for key, label, color, linestyle, marker in reward_keys:
        data = _series(key)
        if data.size > 0:
            steps = _step_axis(data.size)
            markevery = max(1, data.size // 12)
            ax.plot(
                steps,
                data,
                color=color,
                linewidth=1.6,
                alpha=0.9,
                linestyle=linestyle,
                marker=marker,
                markersize=4.2,
                markerfacecolor='white',
                markeredgewidth=0.9,
                markevery=markevery,
                label=label,
            )
            has_reward = True
    ax.set_title('Per-Step Rewards', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    if has_reward:
        ax.legend(
            loc='upper left',
            fontsize=8.5,
            ncol=2,
            framealpha=0.92,
            columnspacing=1.2,
            handlelength=2.8,
            handletextpad=0.6,
        )

    # 7. Per-step NMSE
    ax = axes[3, 0]
    data = _series("nmse")
    has_nmse_legend = False
    if data.size > 0:
        steps = _step_axis(data.size)
        ax.plot(steps, data, 'darkorange', linewidth=2, label='NMSE')
        has_nmse_legend = True
        target_gap = _series("nmse_target_gap")
        if target_gap.size > 0:
            ax.plot(
                steps,
                target_gap,
                color='#5d6d7e',
                linewidth=1.4,
                linestyle='--',
                alpha=0.92,
                label='Target Gap',
            )
            has_nmse_legend = True
        target_nmse = _series("target_nmse")
        if target_nmse.size > 0:
            ax.axhline(
                y=float(target_nmse[0]),
                color='#27ae60',
                linestyle=':',
                linewidth=1.4,
                alpha=0.95,
                label='Target NMSE',
            )
            has_nmse_legend = True

        bootstrap_active = _series("bootstrap_active")
        if bootstrap_active.size > 0:
            active_indices = np.flatnonzero(bootstrap_active > 0.0).tolist()
            if active_indices:
                segment_start = active_indices[0]
                prev_idx = active_indices[0]
                for idx in active_indices[1:]:
                    if idx != prev_idx + 1:
                        ax.axvspan(
                            segment_start + 0.5,
                            prev_idx + 1.5,
                            color='#5dade2',
                            alpha=0.10,
                            label='Bootstrap Phase' if not has_nmse_legend else None,
                        )
                        segment_start = idx
                    prev_idx = idx
                ax.axvspan(
                    segment_start + 0.5,
                    prev_idx + 1.5,
                    color='#5dade2',
                    alpha=0.10,
                    label='Bootstrap Phase',
                )
                has_nmse_legend = True

        ensemble_steps = []
        ensemble_values = []
        raw_ensemble_events = step_details.get("ensemble_events", [])
        if isinstance(raw_ensemble_events, list) and raw_ensemble_events:
            for event in raw_ensemble_events:
                if not isinstance(event, dict):
                    continue
                step_num = int(event.get("step", 0))
                if step_num < 1 or step_num > data.size:
                    continue
                ensemble_steps.append(step_num)
                ensemble_values.append(float(event.get("nmse", data[step_num - 1])))
        else:
            ensemble_flags = _series("ensemble_triggered")
            if ensemble_flags.size > 0:
                for step_idx in np.flatnonzero(ensemble_flags > 0.0).tolist():
                    step_num = int(step_idx + 1)
                    if step_num <= data.size:
                        ensemble_steps.append(step_num)
                        ensemble_values.append(float(data[step_num - 1]))

        if ensemble_steps:
            ax.vlines(
                ensemble_steps,
                ymin=float(np.min(data)),
                ymax=ensemble_values,
                colors='#8e44ad',
                linestyles='--',
                linewidth=1.0,
                alpha=0.35,
            )
            ax.scatter(
                ensemble_steps,
                ensemble_values,
                color='#8e44ad',
                marker='D',
                s=36,
                alpha=0.9,
                zorder=4,
                label='Ensemble Triggered',
            )
            has_nmse_legend = True

        bootstrap_styles = {
            "bootstrap_target_reached": {
                "color": "#16a085",
                "marker": "*",
                "size": 90,
                "label": "Bootstrap Target Reached",
            },
            "bootstrap_handoff": {
                "color": "#2c3e50",
                "marker": "^",
                "size": 52,
                "label": "Bootstrap Handoff",
            },
        }
        seen_bootstrap_labels = set()
        raw_bootstrap_events = step_details.get("bootstrap_events", [])
        if isinstance(raw_bootstrap_events, list) and raw_bootstrap_events:
            for event in raw_bootstrap_events:
                if not isinstance(event, dict):
                    continue
                event_name = str(event.get("event", ""))
                style = bootstrap_styles.get(event_name)
                if style is None:
                    continue
                step_num = int(event.get("step", 0))
                if step_num < 1 or step_num > data.size:
                    continue
                label = style["label"] if event_name not in seen_bootstrap_labels else None
                ax.scatter(
                    [step_num],
                    [float(event.get("nmse", data[step_num - 1]))],
                    color=style["color"],
                    marker=style["marker"],
                    s=style["size"],
                    alpha=0.95,
                    zorder=5,
                    label=label,
                )
                if label is not None:
                    seen_bootstrap_labels.add(event_name)
                    has_nmse_legend = True
        else:
            fallback_bootstrap_markers = [
                ("bootstrap_target_reached", "#16a085", "*", 90, "Bootstrap Target Reached"),
                ("bootstrap_handoff", "#2c3e50", "^", 52, "Bootstrap Handoff"),
            ]
            for key, color, marker, size, label in fallback_bootstrap_markers:
                flags = _series(key)
                if flags.size <= 0:
                    continue
                event_steps = [int(idx + 1) for idx in np.flatnonzero(flags > 0.0).tolist() if int(idx + 1) <= data.size]
                if not event_steps:
                    continue
                ax.scatter(
                    event_steps,
                    [float(data[step_num - 1]) for step_num in event_steps],
                    color=color,
                    marker=marker,
                    s=size,
                    alpha=0.95,
                    zorder=5,
                    label=label,
                )
                has_nmse_legend = True
    ax.set_title('Per-Step NMSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('NMSE')
    ax.grid(True, alpha=0.3)
    if has_nmse_legend:
        ax.legend(loc='best', fontsize=9)

    # 8. Per-step transmission / movement
    ax = axes[3, 1]
    delivered_bits = _series("data_delivered_bits") * 1e-9
    novel_bits = _series("novel_data_delivered_bits") * 1e-9
    uav_move = _series("uav_move_dist")
    ugv_move = _series("ugv_move_dist")
    has_resource = False
    if delivered_bits.size > 0:
        steps = _step_axis(delivered_bits.size)
        ax.plot(steps, delivered_bits, color='#34495e', linewidth=1.6, label='Delivered Bits (Gbits)')
        has_resource = True
    if novel_bits.size > 0:
        steps = _step_axis(novel_bits.size)
        ax.plot(steps, novel_bits, color='#7f8c8d', linewidth=1.6, linestyle='--', label='Novel Bits (Gbits)')
        has_resource = True
    ax.set_title('Per-Step Transmission / Movement', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Bits (Gbits)')
    ax.grid(True, alpha=0.3)

    ax_move = ax.twinx()
    if uav_move.size > 0:
        steps = _step_axis(uav_move.size)
        ax_move.plot(steps, uav_move, color='#2980b9', linewidth=1.4, alpha=0.85, label='UAV Move')
        has_resource = True
    if ugv_move.size > 0:
        steps = _step_axis(ugv_move.size)
        ax_move.plot(steps, ugv_move, color='#8e44ad', linewidth=1.4, alpha=0.85, linestyle=':', label='UGV Move')
        has_resource = True
    ax_move.set_ylabel('Move Distance')

    if has_resource:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_move.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8.5)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved eval step details to {save_path}")


def _get_eval_artifact_records(training_metrics: Dict[str, List]) -> List[Dict[str, object]]:
    raw_uav = training_metrics.get("eval_uav_trajectory", [])
    raw_ugv = training_metrics.get("eval_ugv_trajectory", [])
    raw_step_details = training_metrics.get("eval_step_details", [])
    raw_updates = training_metrics.get("eval_update", [])

    if not isinstance(raw_uav, list) or not isinstance(raw_ugv, list):
        return []
    if isinstance(raw_step_details, dict):
        raw_step_details = [raw_step_details]
    elif not isinstance(raw_step_details, list):
        return []
    if not isinstance(raw_updates, list):
        raw_updates = []

    eval_count = min(len(raw_uav), len(raw_ugv), len(raw_step_details))
    records: List[Dict[str, object]] = []
    for idx in range(eval_count):
        update_idx = None
        if idx < len(raw_updates):
            try:
                update_idx = int(raw_updates[idx])
            except (TypeError, ValueError):
                update_idx = None
        records.append(
            {
                "eval_idx": idx + 1,
                "update_idx": update_idx,
                "uav_trajectory": raw_uav[idx],
                "ugv_trajectory": raw_ugv[idx],
                "step_details": raw_step_details[idx],
            }
        )
    return records


def _format_eval_artifact_stem(eval_idx: int, update_idx: Optional[int]) -> str:
    stem = f"eval_{eval_idx:02d}"
    if update_idx is not None:
        stem = f"{stem}_update_{update_idx:04d}"
    return stem


def _format_eval_artifact_title(eval_idx: int, update_idx: Optional[int]) -> str:
    if update_idx is None:
        return f"Eval {eval_idx}"
    return f"Eval {eval_idx} (Update {update_idx})"


def plot_all(log_dir: str = "logs", target_nmse: Optional[float] = None):
    """
    Load metrics from log_dir/metrics.json and generate all plots.

    Saves figures to log_dir/figures/.
    """
    metrics_path = os.path.join(log_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"[Visualize] No metrics.json found at {metrics_path}")
        return

    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    episode_metrics = all_metrics.get("episodes", {})
    training_metrics = all_metrics.get("training", {})
    grid_size = _grid_size_from_metrics(all_metrics)
    fig_dir = os.path.join(log_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Reward components
    plot_reward_components(
        episode_metrics,
        save_path=os.path.join(fig_dir, "reward_components.png"),
    )

    plot_episode_resource_metrics(
        episode_metrics,
        save_path=os.path.join(fig_dir, "episode_resource_metrics.png"),
    )

    # 2. NMSE curve
    nmse_key = "nmse" if "nmse" in episode_metrics else "nmse_db"
    if nmse_key in episode_metrics and len(episode_metrics[nmse_key]) > 0:
        plot_nmse_curve(
            episode_metrics[nmse_key],
            save_path=os.path.join(fig_dir, "nmse_curve.png"),
            target_nmse=target_nmse,
        )
    plot_nmse_diagnostics(
        episode_metrics,
        save_path=os.path.join(fig_dir, "nmse_diagnostics.png"),
    )

    eval_records = _get_eval_artifact_records(training_metrics)
    if eval_records:
        eval_dir = os.path.join(fig_dir, "evals")
        os.makedirs(eval_dir, exist_ok=True)

        for record in eval_records:
            eval_idx = int(record["eval_idx"])
            update_idx = record["update_idx"]
            title_suffix = _format_eval_artifact_title(eval_idx, update_idx)
            stem = _format_eval_artifact_stem(eval_idx, update_idx)
            uav_traj = np.asarray(record["uav_trajectory"], dtype=float)
            ugv_traj = np.asarray(record["ugv_trajectory"], dtype=float)
            step_details = record["step_details"]

            plot_planner_trajectory_3d_gif(
                uav_traj,
                ugv_traj,
                step_details,
                grid_size=grid_size,
                save_path=os.path.join(eval_dir, f"{stem}_trajectories_3d.gif"),
                title=f"UAV / UGV / Planner 3D Trajectory - {title_suffix}",
            )
            plot_eval_step_details(
                step_details,
                save_path=os.path.join(eval_dir, f"{stem}_step_details.png"),
                title=f"Evaluation Episode - Per-Step Details - {title_suffix}",
            )
            plot_eval_step_reward_components(
                step_details,
                save_path=os.path.join(eval_dir, f"{stem}_step_rewards.png"),
                title=f"Evaluation Episode - Per-Step Reward Components - {title_suffix}",
            )

        last_record = eval_records[-1]
        last_title_suffix = _format_eval_artifact_title(
            int(last_record["eval_idx"]),
            last_record["update_idx"],
        )
        last_uav_traj = np.asarray(last_record["uav_trajectory"], dtype=float)
        last_ugv_traj = np.asarray(last_record["ugv_trajectory"], dtype=float)
        last_eval_step_details = last_record["step_details"]

        # Keep the latest eval artifacts at the top level for compatibility.
        plot_planner_trajectory_3d_gif(
            last_uav_traj,
            last_ugv_traj,
            last_eval_step_details,
            grid_size=grid_size,
            save_path=os.path.join(fig_dir, "trajectories_3d.gif"),
            title=f"UAV / UGV / Planner 3D Trajectory - {last_title_suffix}",
        )
        plot_eval_step_details(
            last_eval_step_details,
            save_path=os.path.join(fig_dir, "eval_step_details.png"),
            title=f"Evaluation Episode - Per-Step Details - {last_title_suffix}",
        )
        plot_eval_step_reward_components(
            last_eval_step_details,
            save_path=os.path.join(fig_dir, "eval_step_rewards.png"),
            title=f"Evaluation Episode - Per-Step Reward Components - {last_title_suffix}",
        )
        print(f"[Visualize] Exported {len(eval_records)} eval trajectory GIF(s) to {eval_dir}")

    plot_eval_metrics(
        training_metrics,
        save_path=os.path.join(fig_dir, "eval_metrics.png"),
    )

    print(f"[Visualize] All plots saved to {fig_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--target_nmse", type=float, default=None)
    args = parser.parse_args()
    plot_all(args.log_dir, args.target_nmse)
