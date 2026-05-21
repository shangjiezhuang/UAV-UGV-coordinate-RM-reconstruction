"""
Visualization module for total-random baseline results.

Provides plotting functions for:
  1. UAV and UGV movement trajectories
  2. Episode resource metrics
  3. NMSE (radio map reconstruction accuracy) over episodes
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple


def _format_optional_int(value: float) -> str:
    if np.isnan(value) or value < 0:
        return "-"
    return f"{int(value):d}"


def _format_optional_ratio(value: float) -> str:
    if np.isnan(value):
        return "-"
    return f"{value:.2f}"


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


def _apply_config_overrides(target: object, overrides: Dict[str, object]) -> None:
    """Recursively hydrate a default Config from a saved metrics snapshot."""
    if not isinstance(overrides, dict):
        return

    for key, value in overrides.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if hasattr(current, "__dict__") and isinstance(value, dict):
            _apply_config_overrides(current, value)
        else:
            setattr(target, key, value)


def _occupancy_from_metrics(all_metrics: Dict[str, object]) -> Optional[np.ndarray]:
    """Rebuild the saved scene occupancy so exported GIFs can show the 2D building layout."""
    raw_config = all_metrics.get("config", {})
    if not isinstance(raw_config, dict):
        return None

    try:
        from config import Config
        from sim_models import SimDataGen

        config = Config()
        _apply_config_overrides(config, raw_config)
        config.__post_init__()
        sim_data = SimDataGen(config=config, seed=int(config.run.seed))
        return sim_data.get_building_mask()
    except (AttributeError, FileNotFoundError, KeyError, ModuleNotFoundError, TypeError, ValueError) as exc:
        print(f"[Visualize] Warning: failed to reconstruct building layout from metrics: {exc}")
        return None


def _draw_building_layout(
    ax,
    occupancy: Optional[np.ndarray],
    grid_size: Tuple[int, int],
) -> None:
    """Render the 2D building footprint as a light occupancy overlay."""
    if occupancy is None:
        return

    occ = np.asarray(occupancy, dtype=bool)
    if occ.ndim != 2 or occ.size == 0:
        return

    masked = np.ma.masked_where(~occ.T, occ.T.astype(float))
    extent = [-0.5, grid_size[0] - 0.5, -0.5, grid_size[1] - 0.5]
    ax.imshow(
        masked,
        origin='lower',
        cmap='Greys',
        alpha=0.38,
        extent=extent,
        interpolation='nearest',
        vmin=0.0,
        vmax=1.0,
        zorder=0,
    )
    if np.any(occ) and not np.all(occ):
        ax.contour(
            occ.T.astype(float),
            levels=[0.5],
            colors='#5c6770',
            linewidths=0.6,
            origin='lower',
            extent=extent,
            zorder=1,
        )


def _setup_planar_axes(
    ax,
    grid_size: Tuple[int, int],
    title: str,
) -> None:
    ax.set_xlim(-0.5, grid_size[0] - 0.5)
    ax.set_ylim(-0.5, grid_size[1] - 0.5)
    ax.set_xlabel('X Grid Index')
    ax.set_ylabel('Y Grid Index')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25, linewidth=0.6)


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
    target_f = np.asarray(step_details.get("target_center_freq", []), dtype=float).reshape(-1)
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


def _extract_target_change_points(
    planner_xyz: np.ndarray,
    planner_valid: np.ndarray,
    *,
    limit: Optional[int] = None,
) -> np.ndarray:
    """Keep only distinct target assignments so history markers stay readable."""
    planner_xyz = np.asarray(planner_xyz, dtype=float)
    planner_valid = np.asarray(planner_valid, dtype=bool).reshape(-1)
    frame_count = min(planner_xyz.shape[0], planner_valid.shape[0])
    if limit is not None:
        frame_count = min(frame_count, max(int(limit), 0))
    if frame_count <= 0:
        return np.empty((0, 3), dtype=float)

    change_indices: List[int] = []
    prev_point: Optional[np.ndarray] = None
    for idx in range(frame_count):
        if not bool(planner_valid[idx]):
            prev_point = None
            continue
        point = planner_xyz[idx]
        if prev_point is None or not np.allclose(point, prev_point, atol=1e-6, rtol=0.0):
            change_indices.append(idx)
        prev_point = point.copy()

    if not change_indices:
        return np.empty((0, 3), dtype=float)
    return planner_xyz[np.asarray(change_indices, dtype=int)]


def _get_step_series(
    step_details: Dict[str, list],
    frame_count: int,
    key: str,
) -> np.ndarray:
    if frame_count <= 0:
        return np.asarray([], dtype=float)
    if key not in step_details:
        raise KeyError(f"Missing required step-detail series: {key}")

    arr = np.asarray(step_details[key], dtype=float).reshape(-1)
    if arr.size < frame_count:
        raise ValueError(
            f"Step-detail series {key!r} has {arr.size} values; expected at least {frame_count}"
        )
    return arr[:frame_count]


def _get_step_string_series(
    step_details: Dict[str, list],
    frame_count: int,
    key: str,
) -> List[str]:
    if frame_count <= 0:
        return []
    if key not in step_details:
        raise KeyError(f"Missing required step-detail series: {key}")
    values = step_details[key]
    if not isinstance(values, list):
        values = list(values) if values is not None else []
    if len(values) < frame_count:
        raise ValueError(
            f"Step-detail series {key!r} has {len(values)} values; expected at least {frame_count}"
        )
    return [str(value) for value in values[:frame_count]]


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


def plot_planner_trajectory_2d(
    uav_trajectory: np.ndarray,
    ugv_trajectory: np.ndarray,
    step_details: Dict[str, list],
    grid_size: Optional[Tuple[int, int]] = None,
    occupancy: Optional[np.ndarray] = None,
    save_path: str = "logs/figures/trajectories_2d.png",
    title: str = "UAV / UGV / Planner 2D Trajectory",
):
    prepared = _prepare_planner_step_data(uav_trajectory, ugv_trajectory, step_details)
    if prepared is None:
        print("[Visualize] Missing planner (x, y, f) step data, skip 2D trajectory plot.")
        return

    uav_steps, ugv_steps, planner_xyz, planner_valid = prepared
    uav_trajectory = np.asarray(uav_trajectory, dtype=float)
    ugv_trajectory = np.asarray(ugv_trajectory, dtype=float)
    uav_path = uav_trajectory[:uav_steps.shape[0] + 1]
    ugv_path = ugv_trajectory[:ugv_steps.shape[0] + 1]
    grid_size = _infer_grid_size(
        grid_size,
        occupancy=occupancy,
        trajectories=(uav_path, ugv_path),
        step_details=step_details,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    ax.set_facecolor('#f8fafc')
    _draw_building_layout(ax, occupancy, grid_size)
    _setup_planar_axes(ax, grid_size, title)

    ax.plot(
        uav_path[:, 0], uav_path[:, 1],
        color='#1f77b4', linewidth=2.0, marker='o', markersize=3, alpha=0.8, label='UAV Path',
        zorder=3,
    )
    ax.plot(
        ugv_path[:, 0], ugv_path[:, 1],
        color='#d62728', linewidth=2.0, linestyle='--', marker='s', markersize=3, alpha=0.8, label='UGV Path',
        zorder=3,
    )

    if planner_valid.any():
        target_changes = _extract_target_change_points(planner_xyz, planner_valid)
        if target_changes.shape[0] > 1:
            ax.scatter(
                target_changes[:-1, 0],
                target_changes[:-1, 1],
                s=56,
                c='#ff6b6b',
                marker='x',
                linewidths=1.8,
                alpha=0.55,
                label='Past Targets',
                zorder=4,
            )
        current_target = target_changes[-1]
        ax.scatter(
            [current_target[0]],
            [current_target[1]],
            s=140,
            c='#ff2d55',
            marker='x',
            linewidths=2.8,
            label='Current Target',
            zorder=6,
        )
        ax.plot(
            [uav_path[-1, 0], current_target[0]],
            [uav_path[-1, 1], current_target[1]],
            color='#ffb703',
            linestyle='--',
            linewidth=1.5,
            alpha=0.92,
            label='UAV->Target',
            zorder=2,
        )
        ax.plot(
            [ugv_path[-1, 0], current_target[0]],
            [ugv_path[-1, 1], current_target[1]],
            color='#f1948a',
            linestyle=':',
            linewidth=1.4,
            alpha=0.85,
            label='UGV->Target',
            zorder=2,
        )

    start_points = {
        "uav_start": np.array([uav_path[0, 0], uav_path[0, 1], 0.0], dtype=float),
        "ugv_start": np.array([ugv_path[0, 0], ugv_path[0, 1], 0.0], dtype=float),
    }
    end_points = {
        "uav_end": np.array([uav_path[-1, 0], uav_path[-1, 1], 0.0], dtype=float),
        "ugv_end": np.array([ugv_path[-1, 0], ugv_path[-1, 1], 0.0], dtype=float),
    }
    start_markers = _spread_marker_points(start_points, radius=0.28)
    end_markers = _spread_marker_points(end_points, radius=0.24)

    ax.scatter([start_markers["uav_start"][0]], [start_markers["uav_start"][1]], s=180, c='#1f77b4', marker='*', edgecolors='black', label='UAV Start', zorder=5)
    ax.scatter([end_markers["uav_end"][0]], [end_markers["uav_end"][1]], s=140, c='#1f77b4', marker='X', edgecolors='black', label='UAV End', zorder=5)
    ax.scatter([start_markers["ugv_start"][0]], [start_markers["ugv_start"][1]], s=180, c='#d62728', marker='D', edgecolors='black', label='UGV Start', zorder=5)
    ax.scatter([end_markers["ugv_end"][0]], [end_markers["ugv_end"][1]], s=140, c='#d62728', marker='P', edgecolors='black', label='UGV End', zorder=5)

    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 0.98),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
    )

    fig.subplots_adjust(left=0.07, right=0.80, top=0.90, bottom=0.08)
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[Visualize] Saved 2D trajectory plot to {save_path}")


def plot_planner_trajectory_2d_gif(
    uav_trajectory: np.ndarray,
    ugv_trajectory: np.ndarray,
    step_details: Dict[str, list],
    grid_size: Optional[Tuple[int, int]] = None,
    occupancy: Optional[np.ndarray] = None,
    save_path: str = "logs/figures/trajectories_2d.gif",
    title: str = "UAV / UGV / Planner 2D Trajectory",
    fps: int = 4,
):
    prepared = _prepare_planner_step_data(uav_trajectory, ugv_trajectory, step_details)
    if prepared is None:
        print("[Visualize] Missing planner (x, y, f) step data, skip 2D trajectory GIF.")
        return

    uav_steps, ugv_steps, planner_xyz, planner_valid = prepared
    uav_trajectory = np.asarray(uav_trajectory, dtype=float)
    ugv_trajectory = np.asarray(ugv_trajectory, dtype=float)
    uav_path = uav_trajectory[:uav_steps.shape[0] + 1]
    ugv_path = ugv_trajectory[:ugv_steps.shape[0] + 1]
    grid_size = _infer_grid_size(
        grid_size,
        occupancy=occupancy,
        trajectories=(uav_path, ugv_path),
        step_details=step_details,
    )
    frame_count = uav_steps.shape[0]
    sample_center_freq = _get_step_series(
        step_details,
        frame_count,
        "sample_center_freq",
    )
    target_center_freq = _get_step_series(
        step_details,
        frame_count,
        "target_center_freq",
    )
    bw_ratio = _get_step_series(step_details, frame_count, "bw_ratio")
    sensing_band_num = _get_step_series(step_details, frame_count, "sensing_band_num")
    sensing_bw_units = _get_step_series(step_details, frame_count, "sensing_bw_units")
    comm_bw_units = _get_step_series(step_details, frame_count, "comm_bw_units")
    planner_submode = _get_step_string_series(step_details, frame_count, "planner_submode")
    planner_mode_switch = _get_step_string_series(step_details, frame_count, "planner_mode_switch")
    global_top_x = [
        _get_step_series(step_details, frame_count, f"global_top{rank}_x")
        for rank in range(1, 4)
    ]
    global_top_y = [
        _get_step_series(step_details, frame_count, f"global_top{rank}_y")
        for rank in range(1, 4)
    ]
    global_top_f = [
        _get_step_series(step_details, frame_count, f"global_top{rank}_freq")
        for rank in range(1, 4)
    ]
    global_top_score = [
        _get_step_series(step_details, frame_count, f"global_top{rank}_score")
        for rank in range(1, 4)
    ]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 8.5))
    ax.set_facecolor('#f8fafc')
    _draw_building_layout(ax, occupancy, grid_size)
    _setup_planar_axes(ax, grid_size, title)
    fig.subplots_adjust(left=0.07, right=0.80, top=0.89, bottom=0.25)

    start_markers = _spread_marker_points(
        {
            "uav_start": np.array([uav_path[0, 0], uav_path[0, 1], 0.0], dtype=float),
            "ugv_start": np.array([ugv_path[0, 0], ugv_path[0, 1], 0.0], dtype=float),
        },
        radius=0.30,
    )
    ax.scatter([start_markers["uav_start"][0]], [start_markers["uav_start"][1]], s=180, c='#1f77b4', marker='*', edgecolors='black', label='UAV Start', zorder=5)
    ax.scatter([start_markers["ugv_start"][0]], [start_markers["ugv_start"][1]], s=180, c='#d62728', marker='D', edgecolors='black', label='UGV Start', zorder=5)

    uav_line, = ax.plot([], [], color='#1f77b4', linewidth=2.0, marker='o', markersize=3, alpha=0.8, label='UAV Path', zorder=3)
    ugv_line, = ax.plot([], [], color='#d62728', linewidth=2.0, linestyle='--', marker='s', markersize=3, alpha=0.8, label='UGV Path', zorder=3)
    planner_hist = ax.scatter(
        [], [],
        s=56,
        c='#ff6b6b',
        marker='x',
        alpha=0.55,
        linewidths=1.8,
        label='Past Targets',
        zorder=4,
    )

    uav_curr, = ax.plot([], [], color='#1f77b4', marker='o', markersize=9, linestyle='None', markeredgecolor='black', label='UAV Current', zorder=6)
    ugv_curr, = ax.plot([], [], color='#d62728', marker='s', markersize=9, linestyle='None', markeredgecolor='black', label='UGV Current', zorder=6)
    planner_curr, = ax.plot([], [], color='#ff2d55', marker='x', markersize=12, linestyle='None', markeredgecolor='#ff2d55', markeredgewidth=2.6, label='Current Target', zorder=6)
    global_top1, = ax.plot([], [], color='#ff8c00', marker='o', markersize=9, linestyle='None', markeredgecolor='black', label='Global Top-1', zorder=6)
    global_top2, = ax.plot([], [], color='#ffb347', marker='o', markersize=8, linestyle='None', markeredgecolor='black', label='Global Top-2', zorder=6)
    global_top3, = ax.plot([], [], color='#ffd166', marker='o', markersize=7, linestyle='None', markeredgecolor='black', label='Global Top-3', zorder=6)
    uav_target_line, = ax.plot([], [], color='#ffb703', linestyle='--', linewidth=1.5, alpha=0.92, label='UAV->Target', zorder=2)
    ugv_target_line, = ax.plot([], [], color='#f1948a', linestyle=':', linewidth=1.6, alpha=0.95, label='UGV->Target', zorder=2)


    step_text = fig.text(
        0.07, 0.94, '', va='top', ha='left',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.9),
    )
    status_text = fig.text(
        0.07, 0.09, '', va='bottom', ha='left',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='lightgray', alpha=0.85),
    )
    sensing_text = fig.text(
        0.07, 0.03, '', va='bottom', ha='left',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='lightgray', alpha=0.85),
    )
    global_text = fig.text(
        0.07, 0.145, '', va='bottom', ha='left',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='lightgray', alpha=0.85),
    )

    legend_handles = [
        Line2D([], [], color='#1f77b4', marker='*', markersize=12, linestyle='None', markeredgecolor='black', label='UAV Start'),
        Line2D([], [], color='#d62728', marker='D', markersize=10, linestyle='None', markeredgecolor='black', label='UGV Start'),
        Line2D([], [], color='#1f77b4', linewidth=2.0, marker='o', markersize=4, alpha=0.8, label='UAV Path'),
        Line2D([], [], color='#d62728', linewidth=2.0, linestyle='--', marker='s', markersize=4, alpha=0.8, label='UGV Path'),
        Line2D([], [], color='#ff6b6b', marker='x', markersize=8, linestyle='None', markeredgewidth=1.8, alpha=0.55, label='Past Targets'),
        Line2D([], [], color='#1f77b4', marker='o', markersize=9, linestyle='None', markeredgecolor='black', label='UAV Current'),
        Line2D([], [], color='#d62728', marker='s', markersize=9, linestyle='None', markeredgecolor='black', label='UGV Current'),
        Line2D([], [], color='#ff2d55', marker='x', markersize=10, linestyle='None', markeredgewidth=2.2, label='Current Target'),
        Line2D([], [], color='#ff8c00', marker='o', markersize=8, linestyle='None', markeredgecolor='black', label='Global Top-1'),
        Line2D([], [], color='#ffb347', marker='o', markersize=7, linestyle='None', markeredgecolor='black', label='Global Top-2'),
        Line2D([], [], color='#ffd166', marker='o', markersize=6, linestyle='None', markeredgecolor='black', label='Global Top-3'),
        Line2D([], [], color='#ffb703', linewidth=1.5, linestyle='--', label='UAV->Target'),
        Line2D([], [], color='#f1948a', linewidth=1.6, linestyle=':', label='UGV->Target'),
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.02, 0.98),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
    )

    def _set_point(artist, point: np.ndarray) -> None:
        artist.set_data([point[0]], [point[1]])

    def _clear_artist(artist) -> None:
        artist.set_data([], [])

    def _set_segment(artist, point_a: np.ndarray, point_b: np.ndarray) -> None:
        artist.set_data([point_a[0], point_b[0]], [point_a[1], point_b[1]])

    def _clear_scatter(scatter) -> None:
        scatter.set_offsets(np.empty((0, 2), dtype=float))


    def _format_score(value: float) -> str:
        if np.isnan(value):
            return "-"
        return f"{value:.3f}"

    def _update(frame_idx: int):
        end_idx = int(frame_idx) + 1
        uav_line.set_data(uav_path[:end_idx + 1, 0], uav_path[:end_idx + 1, 1])
        ugv_line.set_data(ugv_path[:end_idx + 1, 0], ugv_path[:end_idx + 1, 1])

        current_markers = _spread_marker_points(
            {
                "uav_curr": np.array([uav_steps[frame_idx, 0], uav_steps[frame_idx, 1], 0.0], dtype=float),
                "ugv_curr": np.array([ugv_steps[frame_idx, 0], ugv_steps[frame_idx, 1], 0.0], dtype=float),
            },
            radius=0.22,
        )
        _set_point(uav_curr, current_markers["uav_curr"])
        _set_point(ugv_curr, current_markers["ugv_curr"])

        target_changes = _extract_target_change_points(
            planner_xyz,
            planner_valid,
            limit=end_idx,
        )
        if target_changes.shape[0] > 1:
            planner_hist.set_offsets(target_changes[:-1, :2])
        else:
            _clear_scatter(planner_hist)

        planner_tuple = "Local=(-1, -1, -1)"
        uav_target_text = "d(UAV,local)=-"
        ugv_target_text = "d(UGV,local)=-"
        if planner_valid[frame_idx]:
            planner_point = planner_xyz[frame_idx]
            _set_point(planner_curr, planner_point)
            _set_segment(uav_target_line, uav_steps[frame_idx], planner_point)
            _set_segment(ugv_target_line, ugv_steps[frame_idx], planner_point)
            uav_target_dist = float(np.abs(uav_steps[frame_idx, 0] - planner_point[0]) + np.abs(uav_steps[frame_idx, 1] - planner_point[1]))
            ugv_target_dist = float(np.abs(ugv_steps[frame_idx, 0] - planner_point[0]) + np.abs(ugv_steps[frame_idx, 1] - planner_point[1]))
            planner_tuple = f"Local=({planner_point[0]:.0f}, {planner_point[1]:.0f}, {planner_point[2]:.0f})"
            uav_target_text = f"d(UAV,local)={uav_target_dist:.0f}"
            ugv_target_text = f"d(UGV,local)={ugv_target_dist:.0f}"
        else:
            _clear_artist(planner_curr)
            _clear_artist(uav_target_line)
            _clear_artist(ugv_target_line)

        global_entries = []
        global_artists = (global_top1, global_top2, global_top3)
        for rank_idx, artist in enumerate(global_artists):
            gx = global_top_x[rank_idx][frame_idx]
            gy = global_top_y[rank_idx][frame_idx]
            gf = global_top_f[rank_idx][frame_idx]
            gs = global_top_score[rank_idx][frame_idx]
            if gx >= 0.0 and gy >= 0.0 and gf >= 0.0:
                _set_point(artist, np.array([gx, gy], dtype=float))
                global_entries.append(
                    f"#{rank_idx + 1}=({gx:.0f},{gy:.0f},{gf:.0f}|s={_format_score(gs)})"
                )
            else:
                _clear_artist(artist)

        active_mode = planner_submode[frame_idx] or "-"
        mode_switch = planner_mode_switch[frame_idx]
        dist = float(np.linalg.norm(uav_steps[frame_idx] - ugv_steps[frame_idx]))
        step_text.set_text(f"{title}\nStep {frame_idx + 1}/{uav_steps.shape[0]}")
        status_text.set_text(
            f"UAV=({uav_steps[frame_idx, 0]:.0f}, {uav_steps[frame_idx, 1]:.0f})  "
            f"UGV=({ugv_steps[frame_idx, 0]:.0f}, {ugv_steps[frame_idx, 1]:.0f})  Dist={dist:.2f}\n"
            f"Mode={active_mode}{'  Switch=' + mode_switch if mode_switch else ''}\n"
            f"{planner_tuple}  {uav_target_text}  {ugv_target_text}"
        )
        global_text.set_text(
            "Global Top-3: " + ("  ".join(global_entries) if global_entries else "-")
        )
        sensing_text.set_text(
            f"Sample CF={_format_optional_int(sample_center_freq[frame_idx])}  "
            f"Local CF={_format_optional_int(target_center_freq[frame_idx])}  "
            f"Ratio={_format_optional_ratio(bw_ratio[frame_idx])}\n"
            f"SenseBands={_format_optional_int(sensing_band_num[frame_idx])}  "
            f"BW Units S/C={_format_optional_int(sensing_bw_units[frame_idx])}/{_format_optional_int(comm_bw_units[frame_idx])}  "
            f"Frame={frame_idx + 1}"
        )
        return (
            uav_line, ugv_line, planner_hist,
            uav_curr, ugv_curr, planner_curr,
            global_top1, global_top2, global_top3,
            uav_target_line, ugv_target_line,
            step_text, status_text, global_text, sensing_text,
        )

    anim = FuncAnimation(
        fig,
        _update,
        frames=frame_count,
        interval=max(int(1000 / max(fps, 1)), 1),
        blit=False,
    )

    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[Visualize] Saved 2D trajectory GIF to {save_path}")


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
    uav_trajectory = np.asarray(uav_trajectory, dtype=float)
    ugv_trajectory = np.asarray(ugv_trajectory, dtype=float)
    uav_path = uav_trajectory[:uav_steps.shape[0] + 1]
    ugv_path = ugv_trajectory[:ugv_steps.shape[0] + 1]
    grid_size = _infer_grid_size(
        grid_size,
        trajectories=(uav_path, ugv_path),
        step_details=step_details,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    zeros = np.zeros(uav_path.shape[0], dtype=float)
    ax.plot(
        uav_path[:, 0], uav_path[:, 1], zeros,
        color='#1f77b4', linewidth=2.0, marker='o', markersize=3, alpha=0.8, label='UAV Path',
    )
    ax.plot(
        ugv_path[:, 0], ugv_path[:, 1], zeros,
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

    ax.scatter([uav_path[0, 0]], [uav_path[0, 1]], [0.0], s=180, c='#1f77b4', marker='*', edgecolors='black', label='UAV Start')
    ax.scatter([uav_path[-1, 0]], [uav_path[-1, 1]], [0.0], s=140, c='#1f77b4', marker='X', edgecolors='black', label='UAV End')
    ax.scatter([ugv_path[0, 0]], [ugv_path[0, 1]], [0.0], s=180, c='#d62728', marker='D', edgecolors='black', label='UGV Start')
    ax.scatter([ugv_path[-1, 0]], [ugv_path[-1, 1]], [0.0], s=140, c='#d62728', marker='P', edgecolors='black', label='UGV End')

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
    uav_trajectory = np.asarray(uav_trajectory, dtype=float)
    ugv_trajectory = np.asarray(ugv_trajectory, dtype=float)
    uav_path = uav_trajectory[:uav_steps.shape[0] + 1]
    ugv_path = ugv_trajectory[:ugv_steps.shape[0] + 1]
    grid_size = _infer_grid_size(
        grid_size,
        trajectories=(uav_path, ugv_path),
        step_details=step_details,
    )
    frame_count = uav_steps.shape[0]
    sample_center_freq = _get_step_series(
        step_details,
        frame_count,
        "sample_center_freq",
    )
    target_center_freq = _get_step_series(
        step_details,
        frame_count,
        "target_center_freq",
    )
    bw_ratio = _get_step_series(step_details, frame_count, "bw_ratio")
    sensing_band_num = _get_step_series(step_details, frame_count, "sensing_band_num")
    sensing_bw_units = _get_step_series(step_details, frame_count, "sensing_bw_units")
    comm_bw_units = _get_step_series(step_details, frame_count, "comm_bw_units")
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
            "uav_start": np.array([uav_path[0, 0], uav_path[0, 1], 0.0], dtype=float),
            "ugv_start": np.array([ugv_path[0, 0], ugv_path[0, 1], 0.0], dtype=float),
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


    def _update(frame_idx: int):
        end_idx = int(frame_idx) + 1
        zeros = np.zeros(end_idx + 1, dtype=float)
        uav_line.set_data(uav_path[:end_idx + 1, 0], uav_path[:end_idx + 1, 1])
        uav_line.set_3d_properties(zeros)
        ugv_line.set_data(ugv_path[:end_idx + 1, 0], ugv_path[:end_idx + 1, 1])
        ugv_line.set_3d_properties(zeros)
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
            f"Sample CF={_format_optional_int(sample_center_freq[frame_idx])}  "
            f"Planner CF={_format_optional_int(target_center_freq[frame_idx])}  "
            f"Ratio={_format_optional_ratio(bw_ratio[frame_idx])}\n"
            f"SenseBands={_format_optional_int(sensing_band_num[frame_idx])}  "
            f"BW Units S/C={_format_optional_int(sensing_bw_units[frame_idx])}/{_format_optional_int(comm_bw_units[frame_idx])}  "
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
    Plot NMSE over completed episodes.

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
    ax.set_title('Radio Map NMSE Over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved NMSE curve to {save_path}")


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
        ("uav_move_dist", "UAV Move Distance / Episode", "#2980b9", 1.0, "Meters"),
        ("ugv_move_dist", "UGV Move Distance / Episode", "#8e44ad", 1.0, "Meters"),
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


def plot_eval_step_details(
    step_details: Dict[str, list],
    save_path: str = "logs/figures/eval_step_details.png",
    title: str = "Evaluation Episode - Per-Step Details",
):
    """
    Plot per-step evaluation details from the last eval episode.

    Shows: queue length, SNR, bandwidth ratio, sensing index,
           UAV-UGV distance, NMSE, and resource usage.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not isinstance(step_details, dict):
        print("[Visualize] Invalid eval step details format, skip.")
        return

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))

    def _series(key: str) -> np.ndarray:
        values = step_details[key]
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

    # 4. Sample center frequency
    ax = axes[1, 1]
    data = _series("sample_center_freq")
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

    # 6. Per-step transmission / movement
    ax = axes[2, 1]
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

        executed_target_source = step_details["executed_target_source"]
        bootstrap_active = np.asarray(
            [1.0 if str(source) == "bootstrap" else 0.0 for source in executed_target_source],
            dtype=float,
        ).reshape(-1)
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
        for event in step_details["ensemble_events"]:
            if not isinstance(event, dict):
                continue
            step_num = int(event.get("step", 0))
            if step_num < 1 or step_num > data.size:
                continue
            ensemble_steps.append(step_num)
            ensemble_values.append(float(event["nmse"]))

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
        raw_bootstrap_events = step_details["bootstrap_events"]
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
                    [float(event["nmse"])],
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
    ax.set_title('Per-Step NMSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('NMSE')
    ax.grid(True, alpha=0.3)
    if has_nmse_legend:
        ax.legend(loc='best', fontsize=9)

    axes[3, 1].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualize] Saved eval step details to {save_path}")


def _prefixed_eval_history_key(prefix: str, suffix: str) -> str:
    if prefix == "eval":
        return f"eval_{suffix}"
    return f"{prefix}_{suffix}"


def _get_eval_artifact_records(
    run_metrics: Dict[str, List],
    prefix: str = "eval",
) -> List[Dict[str, object]]:
    raw_uav = run_metrics.get(_prefixed_eval_history_key(prefix, "uav_trajectory"), [])
    raw_ugv = run_metrics.get(_prefixed_eval_history_key(prefix, "ugv_trajectory"), [])
    raw_step_details = run_metrics.get(_prefixed_eval_history_key(prefix, "step_details"), [])
    raw_iterations = run_metrics.get(f"{prefix}_iteration", [])

    if not isinstance(raw_uav, list) or not isinstance(raw_ugv, list):
        return []
    if isinstance(raw_step_details, dict):
        raw_step_details = [raw_step_details]
    elif not isinstance(raw_step_details, list):
        return []
    if not isinstance(raw_iterations, list):
        raw_iterations = []

    eval_count = min(len(raw_uav), len(raw_ugv), len(raw_step_details))
    records: List[Dict[str, object]] = []
    for idx in range(eval_count):
        iteration_idx = None
        if idx < len(raw_iterations):
            try:
                iteration_idx = int(raw_iterations[idx])
            except (TypeError, ValueError):
                iteration_idx = None
        records.append(
            {
                "eval_idx": idx + 1,
                "iteration_idx": iteration_idx,
                "uav_trajectory": raw_uav[idx],
                "ugv_trajectory": raw_ugv[idx],
                "step_details": raw_step_details[idx],
            }
        )
    return records


def _format_eval_artifact_stem(eval_idx: int, iteration_idx: Optional[int]) -> str:
    stem = f"eval_{eval_idx:02d}"
    if iteration_idx is not None:
        stem = f"{stem}_iteration_{iteration_idx:04d}"
    return stem


def _format_eval_artifact_title(eval_idx: int, iteration_idx: Optional[int]) -> str:
    if iteration_idx is None:
        return f"Eval {eval_idx}"
    return f"Eval {eval_idx} (Iteration {iteration_idx})"


def _format_final_eval_artifact_title(iteration_idx: Optional[int]) -> str:
    if iteration_idx is None:
        return "Final Eval"
    return f"Final Eval (Iteration {iteration_idx})"


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
    run_metrics = all_metrics.get("run", {})
    grid_size = _grid_size_from_metrics(all_metrics)
    occupancy = _occupancy_from_metrics(all_metrics)
    fig_dir = os.path.join(log_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

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

    eval_records = _get_eval_artifact_records(run_metrics, prefix="eval")
    if eval_records:
        eval_dir = os.path.join(fig_dir, "evals")
        os.makedirs(eval_dir, exist_ok=True)

        for record in eval_records:
            eval_idx = int(record["eval_idx"])
            iteration_idx = record["iteration_idx"]
            title_suffix = _format_eval_artifact_title(eval_idx, iteration_idx)
            stem = _format_eval_artifact_stem(eval_idx, iteration_idx)
            uav_traj = np.asarray(record["uav_trajectory"], dtype=float)
            ugv_traj = np.asarray(record["ugv_trajectory"], dtype=float)
            step_details = record["step_details"]

            plot_planner_trajectory_2d_gif(
                uav_traj,
                ugv_traj,
                step_details,
                grid_size=grid_size,
                occupancy=occupancy,
                save_path=os.path.join(eval_dir, f"{stem}_trajectories_2d.gif"),
                title=f"UAV / UGV / Planner 2D Trajectory - {title_suffix}",
            )
            plot_eval_step_details(
                step_details,
                save_path=os.path.join(eval_dir, f"{stem}_step_details.png"),
                title=f"Evaluation Episode - Per-Step Details - {title_suffix}",
            )
        print(f"[Visualize] Exported {len(eval_records)} eval trajectory GIF(s) to {eval_dir}")

    final_eval_records = _get_eval_artifact_records(run_metrics, prefix="final_eval")
    top_level_record: Optional[Dict[str, object]] = None
    top_level_title_suffix: Optional[str] = None
    if final_eval_records:
        top_level_record = final_eval_records[-1]
        top_level_title_suffix = _format_final_eval_artifact_title(
            top_level_record["iteration_idx"],
        )
    elif eval_records:
        top_level_record = eval_records[-1]
        top_level_title_suffix = _format_eval_artifact_title(
            int(top_level_record["eval_idx"]),
            top_level_record["iteration_idx"],
        )

    if top_level_record is not None and top_level_title_suffix is not None:
        top_level_uav_traj = np.asarray(top_level_record["uav_trajectory"], dtype=float)
        top_level_ugv_traj = np.asarray(top_level_record["ugv_trajectory"], dtype=float)
        top_level_step_details = top_level_record["step_details"]

        # Keep the top-level eval artifacts reserved for the final eval run.
        plot_planner_trajectory_2d_gif(
            top_level_uav_traj,
            top_level_ugv_traj,
            top_level_step_details,
            grid_size=grid_size,
            occupancy=occupancy,
            save_path=os.path.join(fig_dir, "trajectories_2d.gif"),
            title=f"UAV / UGV / Planner 2D Trajectory - {top_level_title_suffix}",
        )
        plot_eval_step_details(
            top_level_step_details,
            save_path=os.path.join(fig_dir, "eval_step_details.png"),
            title=f"Evaluation Episode - Per-Step Details - {top_level_title_suffix}",
        )

    print(f"[Visualize] All plots saved to {fig_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--target_nmse", type=float, default=None)
    args = parser.parse_args()
    plot_all(args.log_dir, args.target_nmse)
