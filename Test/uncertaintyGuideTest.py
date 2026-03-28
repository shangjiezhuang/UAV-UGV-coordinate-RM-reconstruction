import argparse
import os
import sys
import time
from typing import Callable, Dict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MAPPO_DIR = os.path.join(ROOT_DIR, "MAPPO")

for p in [ROOT_DIR, MAPPO_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


from RT.rtSpectrumGen import SionnaSimConfig, generate_data_rt
from config import Config
from interfaces import SpectrumSample

try:
    from real_models import RealSionnaRT, RealTensorDecomposition
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise ModuleNotFoundError(
            "real_models.py imports torch at module level. "
            "Install torch first to run this test, even when BNN is not used."
        ) from exc
    raise


SAVE_DIR = os.path.join(SCRIPT_DIR, "outputs", "uncertainty_guide")
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers matching iibtdOptRTTest.py logic
# ---------------------------------------------------------------------------
def normalize_rt_data(data: dict) -> dict:
    """
    Same global scaling policy as Test/iibtdOptRTTest.py and RealSionnaRT:
    scale S max magnitude to ~0.05 for II-BTD stability.
    """
    if "S" not in data:
        data["scale_factor"] = 1.0
        return data

    s_list = data["S"]
    s_max = max(np.max(np.abs(s)) for s in s_list)
    if s_max < 1e-20:
        data["scale_factor"] = 1.0
        return data

    scale = 0.05 / s_max
    data["S"] = [s * scale for s in s_list]

    if "prop_grid" in data:
        data["prop_grid"] = [pg * scale for pg in data["prop_grid"]]
    if "prop_sensor" in data:
        data["prop_sensor"] = [ps * scale for ps in data["prop_sensor"]]
    if "Gamma_obs" in data:
        data["Gamma_obs"] = data["Gamma_obs"] * scale
    if "Gamma_clean" in data:
        data["Gamma_clean"] = data["Gamma_clean"] * scale
    if "sigma2_noise" in data:
        data["sigma2_noise"] = data["sigma2_noise"] * (scale ** 2)
    if "H" in data:
        data["H"] = data["H"] * scale

    data["scale_factor"] = scale
    return data


def generate_omega_full(m: int, k: int, ratio: float = 1.0, rng: np.random.Generator = None) -> np.ndarray:
    _ = ratio, rng
    return np.ones((m, k), dtype=np.int32)


def generate_omega_dual_center(m: int, k: int, ratio: float, rng: np.random.Generator = None) -> np.ndarray:
    _ = rng
    omega = np.zeros((m, k), dtype=np.int32)
    n_obs = int(k * ratio)
    n_per_center = max(1, n_obs // 2)
    c1 = k // 4
    c2 = 3 * k // 4
    for i in range(m):
        observed = set()
        for off in range(n_per_center):
            if c1 - off >= 0:
                observed.add(c1 - off)
            if c1 + off < k:
                observed.add(c1 + off)
            if len(observed) >= n_per_center:
                break
        for off in range(n_per_center):
            if c2 - off >= 0:
                observed.add(c2 - off)
            if c2 + off < k:
                observed.add(c2 + off)
            if len(observed) >= n_obs:
                break
        for b in observed:
            omega[i, b] = 1
    return omega


def generate_omega_random(m: int, k: int, ratio: float, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    omega = np.zeros((m, k), dtype=np.int32)
    n_obs = max(1, int(k * ratio))
    for i in range(m):
        bands = rng.choice(k, size=n_obs, replace=False)
        omega[i, bands] = 1
    return omega


def generate_omega_cyclic(
    m: int,
    k: int,
    ratio: float,
    overlap: int = 2,
    start_offset: int = 0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    _ = rng
    block = max(1, int(k * ratio))
    overlap = max(0, min(overlap, block - 1))
    step = block - overlap
    n_steps = max(1, (k - overlap) // step)

    omega = np.zeros((m, k), dtype=np.int32)
    for i in range(m):
        step_idx = (i + start_offset) % n_steps
        start = step_idx * step
        end = min(start + block, k)
        omega[i, start:end] = 1
    return omega


OMEGA_GENERATORS: Dict[str, Callable[..., np.ndarray]] = {
    "full": generate_omega_full,
    "dual_center": generate_omega_dual_center,
    "random": generate_omega_random,
    "cyclic": generate_omega_cyclic,
}


def compute_nmse(h_hat: np.ndarray, h_true: np.ndarray) -> float:
    return float(np.sum((h_hat - h_true) ** 2) / (np.sum(h_true ** 2) + 1e-10))


def generate_grid_path(start_ix: int, start_iy: int, nx: int, ny: int, m: int) -> np.ndarray:
    """
    Generate a grid path of length m with directional priority:
    Up -> Down -> Left -> Right.
    """
    start_ix = int(np.clip(start_ix, 0, nx - 1))
    start_iy = int(np.clip(start_iy, 0, ny - 1))

    # Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    path = [(start_ix, start_iy)]
    visited = {(start_ix, start_iy)}
    cur_ix, cur_iy = start_ix, start_iy

    while len(path) < m:
        moved = False

        # First try an unvisited neighbor in the fixed direction order.
        for dx, dy in directions:
            nx_i = cur_ix + dx
            ny_i = cur_iy + dy
            if 0 <= nx_i < nx and 0 <= ny_i < ny and (nx_i, ny_i) not in visited:
                cur_ix, cur_iy = nx_i, ny_i
                path.append((cur_ix, cur_iy))
                visited.add((cur_ix, cur_iy))
                moved = True
                break

        if moved:
            continue

        # If all neighbors are visited, still move in the same priority order.
        for dx, dy in directions:
            nx_i = cur_ix + dx
            ny_i = cur_iy + dy
            if 0 <= nx_i < nx and 0 <= ny_i < ny:
                cur_ix, cur_iy = nx_i, ny_i
                path.append((cur_ix, cur_iy))
                moved = True
                break

        if not moved:
            # Degenerate case; cannot move (should not happen for nx, ny >= 1).
            path.append((cur_ix, cur_iy))

    return np.asarray(path, dtype=np.int32)


def choose_uncertainty_target(
    current: np.ndarray,
    observed_mask: np.ndarray,
    visit_count: np.ndarray,
    dist_weight: float = 0.35,
    revisit_penalty: float = 0.15,
) -> np.ndarray:
    """
    Simple uncertainty-based destination selector.
    Uncertainty is the unobserved band ratio at each grid cell.
    """
    nx, ny, k = observed_mask.shape
    observed_ratio = observed_mask.sum(axis=2) / max(k, 1)
    uncertainty = 1.0 - observed_ratio

    xs = np.arange(nx)[:, None]
    ys = np.arange(ny)[None, :]
    dist = np.abs(xs - int(current[0])) + np.abs(ys - int(current[1]))
    norm_dist = dist / max(nx + ny - 2, 1)
    norm_visit = np.minimum(visit_count, 5) / 5.0

    score = uncertainty - dist_weight * norm_dist - revisit_penalty * norm_visit

    # Avoid selecting current cell when there are alternatives with uncertainty.
    if np.any(uncertainty > 0.0):
        score[int(current[0]), int(current[1])] = -np.inf

    best = np.unravel_index(np.argmax(score), score.shape)
    return np.array([int(best[0]), int(best[1])], dtype=np.int32)


def step_towards_target(current: np.ndarray, target: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Move one grid step toward target with U/D/L/R priority."""
    cx, cy = int(current[0]), int(current[1])
    tx, ty = int(target[0]), int(target[1])

    if cx > tx:
        cx -= 1  # Up
    elif cx < tx:
        cx += 1  # Down
    elif cy > ty:
        cy -= 1  # Left
    elif cy < ty:
        cy += 1  # Right

    cx = int(np.clip(cx, 0, nx - 1))
    cy = int(np.clip(cy, 0, ny - 1))
    return np.array([cx, cy], dtype=np.int32)


def grid_index_to_xy(grid_points: np.ndarray, ny: int, ix: int, iy: int) -> np.ndarray:
    flat_idx = int(ix) * ny + int(iy)
    return grid_points[flat_idx]


def build_spectrum_samples_from_grid(
    grid_indices: np.ndarray,
    gamma_rows: np.ndarray,
    omega_rows: np.ndarray,
    t0: int,
) -> list:
    samples = []
    for idx in range(grid_indices.shape[0]):
        omega_row = omega_rows[idx]
        freq_indices = np.where(omega_row > 0.5)[0].astype(np.int32)
        measurements = gamma_rows[idx, freq_indices] if freq_indices.size > 0 else np.array([], dtype=np.float64)
        samples.append(
            SpectrumSample(
                position=grid_indices[idx].astype(np.int32),
                freq_group_idx=0,
                freq_band_indices=freq_indices,
                measurements=measurements,
                gamma=gamma_rows[idx].copy(),
                omega=omega_row.copy(),
                timestamp=t0 + idx,
            )
        )
    return samples


def save_nmse_plot(m_values: list, nmse_values: list, save_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m_values, nmse_values, "o-", linewidth=2)
    ax.set_xlabel("Number of Sensors (M)")
    ax.set_ylabel("NMSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {save_path}")


def run_uncertainty_guide_test(args: argparse.Namespace) -> tuple:
    start_x = args.start_x if args.start_x is not None else (args.N1 // 2)
    start_y = args.start_y if args.start_y is not None else (args.N2 // 2)
    start_x = int(np.clip(start_x, 0, args.N1 - 1))
    start_y = int(np.clip(start_y, 0, args.N2 - 1))

    print("=" * 64)
    print("Uncertainty Guide Test (No BNN Inference)")
    print(
        f"Scene={args.scene}, R={args.R}, K={args.K}, M={args.M}, "
        f"Grid={args.N1}x{args.N2}, SNR={args.SNR} dB"
    )
    print(
        f"Start=({start_x}, {start_y}), MoveOrder=Up->Down->Left->Right, "
        f"Planner={'uncertainty' if args.use_uncertainty_planner else 'fixed'}, "
        f"Omega={args.omega}, ratio={args.ratio}, step={args.step}, "
        f"mode={'interface' if args.use_interface_sampling else 'direct'}"
    )
    print("=" * 64)

    # 1) RT data generation (same pipeline as iibtdOptRTTest.py)
    sim_cfg = SionnaSimConfig(
        scene_name=args.scene,
        R=args.R,
        K=args.K,
        M=args.M,
        N1=args.N1,
        N2=args.N2,
        SNR_dB=args.SNR,
        max_depth=5,
        samples_per_tx=10 ** 6,
    )
    t0 = time.time()
    data = generate_data_rt(sim_cfg, seed=args.seed, use_pathsolver_for_sensors=args.use_pathsolver)
    data = normalize_rt_data(data)
    print(f"[RT] Data ready in {time.time() - t0:.1f}s")

    # 2) Build real_models wrappers (no BNN)
    cfg = Config()
    cfg.scene.grid_size = (args.N1, args.N2)
    cfg.scene.total_freq_bands_nums = args.K

    sionna = RealSionnaRT(config=cfg, seed=args.seed, rt_data=data)
    td_model = RealTensorDecomposition(
        config=cfg,
        grid_coords=sionna.grid_points,
        bounds=sionna.get_bounds(),
        I_mask=sionna.I_mask,
        n_sources=args.R,
    )
    td_model.set_ground_truth(sionna.get_full_ground_truth_map())

    # 3) Build grid path + frequency observation mask
    omega_gen = OMEGA_GENERATORS.get(args.omega, generate_omega_full)
    rng_omega = np.random.default_rng(args.seed + 2000)
    omega_all = omega_gen(args.M, args.K, args.ratio, rng=rng_omega)

    if args.use_uncertainty_planner:
        grid_path = np.zeros((args.M, 2), dtype=np.int32)
    else:
        grid_path = generate_grid_path(start_x, start_y, args.N1, args.N2, args.M)

    # Measurements sampled along the grid path.
    locs_all = np.zeros((args.M, 2), dtype=np.float64)
    gamma_all = np.zeros((args.M, args.K), dtype=np.float64)
    observed_mask = np.zeros((args.N1, args.N2, args.K), dtype=np.int8)
    visit_count = np.zeros((args.N1, args.N2), dtype=np.int32)

    current = np.array([start_x, start_y], dtype=np.int32)
    target = current.copy()
    move_steps = 0

    for i in range(args.M):
        if args.use_uncertainty_planner:
            grid_path[i] = current
        ix, iy = int(grid_path[i, 0]), int(grid_path[i, 1])
        locs_all[i] = grid_index_to_xy(sionna.grid_points, args.N2, ix, iy)

        obs_idx = np.where(omega_all[i] > 0.5)[0].astype(np.int32)
        if obs_idx.size > 0:
            measured = sionna.get_spectrum_ground_truth(np.array([ix, iy], dtype=np.int32), obs_idx)
            gamma_all[i, obs_idx] = measured
            observed_mask[ix, iy, obs_idx] = 1
        visit_count[ix, iy] += 1

        if i == args.M - 1:
            break

        if args.use_uncertainty_planner:
            if i % max(1, args.replan_interval) == 0 or np.array_equal(current, target):
                target = choose_uncertainty_target(
                    current=current,
                    observed_mask=observed_mask,
                    visit_count=visit_count,
                    dist_weight=args.dist_weight,
                    revisit_penalty=args.revisit_penalty,
                )
            nxt = step_towards_target(current, target, args.N1, args.N2)
            if not np.array_equal(nxt, current):
                move_steps += 1
            current = nxt

    print(f"[Path] first points: {grid_path[:min(8, len(grid_path))].tolist()}")
    if args.use_uncertainty_planner:
        print(f"[Planner] total_move_steps={move_steps} / max_possible={max(args.M - 1, 0)}")
    print(f"[Omega] coverage={omega_all.mean():.3f}")

    # 4) Sequential incremental reconstruction
    m_values = list(range(args.step, args.M + 1, args.step))
    if not m_values or m_values[-1] != args.M:
        m_values.append(args.M)

    nmse_list = []
    prev = 0
    print(f"\n{'M':>6s}  {'NMSE':>12s}  {'Time(s)':>8s}")
    print("-" * 32)

    for m_cur in m_values:
        new_locs = locs_all[prev:m_cur]
        new_gamma = gamma_all[prev:m_cur, :]
        new_omega = omega_all[prev:m_cur, :]
        new_grid_idx = grid_path[prev:m_cur]

        t1 = time.time()
        if args.use_interface_sampling:
            # Optional mode: test RealTensorDecomposition.add_samples/reconstruct path.
            samples = build_spectrum_samples_from_grid(
                grid_indices=new_grid_idx,
                gamma_rows=new_gamma,
                omega_rows=new_omega,
                t0=prev,
            )
            td_model.add_samples(samples)
            state = td_model.reconstruct()
            nmse = state.nmse
        else:
            # Default mode: align exactly with iibtdOptRTTest.py update path.
            td_model.btd.add_measurements(
                new_locs,
                new_gamma,
                new_omega,
                n_outer_iter=3,
                max_svt_iter=10,
                debugFlag=False,
            )
            nmse = compute_nmse(td_model.btd.H_hat, sionna.ground_truth)

        dt = time.time() - t1
        nmse_list.append(float(nmse))
        print(f"{m_cur:6d}  {nmse:12.6f}  {dt:8.2f}")
        prev = m_cur

    # 5) Save curve
    save_path = os.path.join(
        SAVE_DIR,
        f"nmse_uncertainty_guide_{args.scene}_planner_{'uncertainty' if args.use_uncertainty_planner else 'fixed'}_"
        f"mode_{'interface' if args.use_interface_sampling else 'direct'}.png",
    )
    save_nmse_plot(
        m_values,
        nmse_list,
        save_path,
        title=(
            f"NMSE vs Sensors | scene={args.scene}, R={args.R}, K={args.K}, "
            f"start=({start_x},{start_y}), planner={'uncertainty' if args.use_uncertainty_planner else 'fixed'}, "
            f"omega={args.omega}, "
            f"mode={'interface' if args.use_interface_sampling else 'direct'}"
        ),
    )
    return m_values, nmse_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RT + II-BTD test using MAPPO real_models (without BNN).")
    parser.add_argument("--scene", type=str, default="simple_street_canyon")
    parser.add_argument("--R", type=int, default=1)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--M", type=int, default=130)
    parser.add_argument("--N1", type=int, default=51)
    parser.add_argument("--N2", type=int, default=51)
    parser.add_argument("--SNR", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_x", type=int, default=None, help="Start grid x index. Default: N1//2")
    parser.add_argument("--start_y", type=int, default=None, help="Start grid y index. Default: N2//2")
    parser.add_argument(
        "--disable_uncertainty_planner",
        action="store_false",
        dest="use_uncertainty_planner",
        help="Disable uncertainty-aware destination planning and use fixed U/D/L/R path.",
    )
    parser.set_defaults(use_uncertainty_planner=True)
    parser.add_argument("--replan_interval", type=int, default=3, help="Re-plan destination every N samples.")
    parser.add_argument("--dist_weight", type=float, default=0.35, help="Distance penalty in uncertainty target score.")
    parser.add_argument("--revisit_penalty", type=float, default=0.15, help="Penalty for repeatedly visiting same cell.")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--omega", type=str, default="cyclic", choices=["full", "dual_center", "random", "cyclic"])
    parser.add_argument("--ratio", type=float, default=0.6)
    parser.add_argument("--use_pathsolver", action="store_true", help="Use PathSolver for sensor links when generating RT data.")
    parser.add_argument(
        "--use_interface_sampling",
        action="store_true",
        help="Use RealTensorDecomposition.add_samples/reconstruct path. "
             "Default is direct add_measurements to match iibtdOptRTTest.py exactly.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_uncertainty_guide_test(args)
