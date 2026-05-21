"""
Multi-Agent RL Environment for UAV-UGV cooperative active sensing.

Key behavior:
- UGV reconstructs radio map from delivered spectrum samples.
- UGV aggregates ShareMem ensemble outputs and provides one most-informative grid target.
- UAV action jointly controls one of 5 movement directions (stay + 4 cardinals) and sensing-communication bandwidth split.
- UAV and UGV both move only on integer grid points.
- UAV may traverse building cells but can only sample on non-building cells.
- UAV can transmit queued data from any in-bounds cell according to channel capacity.
- Delivered data are buffered until the next ensemble interval, then fused and committed for reconstruction.
- UGV keeps 5-way movement actions (stay/east/north/west/south) on non-building cells only.
"""

from __future__ import annotations

from collections import deque
import multiprocessing as mp
import random
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from active_sampling import (
    build_acquisition_space,
    build_observe_mask,
    ensemble_reconstruct_maps,
    fuse_observations_by_grid,
    select_top_k_grid_candidates,
)
from config import Config
from du_iibtd_based.scene_suite import scene_config_from_shared_data, select_shared_data
from sim_models import (
    ChannelInfo,
    GridScene,
    IIBTD_opt,
    RadioMapState,
    SimDataGen,
    SpectrumSample,
    UncertaintyMap,
)


DIRECTION_OFFSETS = {
    0: np.array([0, 0]),    # Stay
    1: np.array([1, 0]),    # East
    2: np.array([0, 1]),    # North
    3: np.array([-1, 0]),   # West
    4: np.array([0, -1]),   # South
}

UAV_DIRECTION_IDS = [0, 1, 2, 3, 4]
UGV_DIRECTION_IDS = [0, 1, 2, 3, 4]

_SUBPROC_READY = "ready"
_SUBPROC_RESULT = "result"
_SUBPROC_ERROR = "error"
_SUBPROC_POLL_INTERVAL_SECONDS = 0.1
_SUBPROC_RESPONSE_TIMEOUT_SECONDS = 300.0

_MINIMAL_STEP_INFO_KEYS = (
    "r_nmse",
    "r_unc",
    "r_new_freq",
    "r_new_spatial",
    "r_tx",
    "r_queue",
    "r_progress",
    "r_revisit",
    "r_ugv_building_clearance",
    "r_terminal",
    "r_uav_progress",
    "r_ugv_progress",
    "r_goal_arrival",
    "ensemble_triggered",
    "team_reward",
    "uav_reward",
    "ugv_reward",
    "data_produced_bits",
    "data_delivered_bits",
    "novel_data_delivered_bits",
    "data_transmitted_bits",
    "uav_move_dist",
    "ugv_move_dist",
    "target_gap_penalty_diag",
    "ugv_building_clearance_norm",
    "ugv_building_clearance_deficit",
    "newly_sampled_freqs",
    "newly_visited_spatial",
    "nmse",
    "target_grid_x",
    "target_grid_y",
    "target_center_freq",
    "target_source",
    "executed_target_grid_x",
    "executed_target_grid_y",
    "executed_target_center_freq",
    "executed_target_source",
    "ensemble_target_grid_x",
    "ensemble_target_grid_y",
    "ensemble_target_center_freq",
    "target_reached",
    "target_exact_reached",
    "target_proximity_reached",
    "target_arrival_distance",
    "target_retargeted",
    "target_retarget_reason",
    "active_target_no_sample_steps",
    "bootstrap_target_reached",
    "planner_submode",
    "planner_mode_switch",
    "uav_energy",
    "queue_size",
    "target_nmse_reached",
    "energy_depleted",
    "terminal_failure",
    "timed_out",
)

_UGV_BUILDING_OBS_DIRECTIONS = tuple(UGV_DIRECTION_IDS)


def _compute_vec_reset_seed(base_seed: int, env_idx: int, reset_count: int, num_envs: int) -> int:
    max_seed = (2 ** 32) - 1
    seed = (int(base_seed) + int(env_idx) + int(reset_count) * int(num_envs)) % max_seed
    return int(seed)

def _seed_subproc_worker_rngs(seed: int) -> None:
    """Seed process-global RNGs used by solver fallbacks inside subproc workers."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _subproc_send(remote, kind: str, payload) -> None:
    try:
        remote.send((kind, payload))
    except (BrokenPipeError, EOFError, OSError):
        pass


def _serialize_subproc_exception(stage: str, exc: BaseException) -> dict:
    return {
        "stage": str(stage),
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _build_subproc_env(
    config: Config,
    shared_data: Dict,
    env_idx: int,
    minimal_info: bool = False,
) -> "UAVUGVEnvironment":
    env_shared_data = select_shared_data(shared_data, env_idx)
    env_config = scene_config_from_shared_data(config, env_shared_data)
    sim_data = SimDataGen(
        config=env_config,
        seed=int(env_config.mappo.seed + env_idx),
        precomputed_data=env_shared_data,
    )
    td = IIBTD_opt(
        config=env_config,
        grid_coords=sim_data.grid_coords,
        bounds=sim_data.bounds,
        i_mask=sim_data.I_mask,
        n_sources=1,
    )
    return UAVUGVEnvironment(
        config=env_config,
        tensor_decomp=td,
        sim_data=sim_data,
        scene_map=GridScene(
            env_config,
            occupancy_grid=sim_data.get_building_mask(),
            building_heights=sim_data.get_building_heights(),
        ),
        minimal_info=minimal_info,
    )


def _subproc_env_worker(
    remote,
    config: Config,
    shared_data: Dict,
    env_idx: int,
    minimal_info: bool = False,
) -> None:
    """Worker loop for one rollout environment."""
    env: Optional[UAVUGVEnvironment] = None
    try:
        _seed_subproc_worker_rngs(int(config.mappo.seed + env_idx))
        env = _build_subproc_env(
            config=config,
            shared_data=shared_data,
            env_idx=env_idx,
            minimal_info=minimal_info,
        )
        _subproc_send(
            remote,
            _SUBPROC_READY,
            {
                "env_idx": int(env_idx),
                "obs_dims": env.get_obs_dims(),
                "action_dims": env.get_action_dims(),
            },
        )
        while True:
            try:
                cmd, payload = remote.recv()
            except EOFError:
                break

            try:
                if cmd == "reset":
                    reset_seed = int(payload)
                    _seed_subproc_worker_rngs(reset_seed)
                    obs, _ = env.reset(seed=reset_seed)
                    _subproc_send(remote, _SUBPROC_RESULT, obs)
                elif cmd == "step":
                    uav_action, ugv_action, reset_seed = payload
                    obs, rew, term, trunc, info = env.step(int(uav_action), int(ugv_action))
                    if term or trunc:
                        info["terminal_obs"] = obs
                        next_reset_seed = int(reset_seed)
                        _seed_subproc_worker_rngs(next_reset_seed)
                        obs, _ = env.reset(seed=next_reset_seed)
                    _subproc_send(remote, _SUBPROC_RESULT, (obs, rew, term, trunc, info))
                elif cmd == "close":
                    break
                else:
                    raise ValueError(f"Unsupported worker command: {cmd}")
            except Exception as exc:
                _subproc_send(remote, _SUBPROC_ERROR, _serialize_subproc_exception(cmd, exc))
                break
    except EOFError:
        pass
    except Exception as exc:
        _subproc_send(remote, _SUBPROC_ERROR, _serialize_subproc_exception("init", exc))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        try:
            remote.close()
        except OSError:
            pass


def _stack_vec_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = obs_list[0].keys()
    return {key: np.stack([obs[key] for obs in obs_list]) for key in keys}


def _pack_vec_step_results(
    results: List[Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, dict]]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict]]:
    obs_list = [r[0] for r in results]
    rewards = {
        "team_reward": np.array(
            [r[1]["team_reward"] for r in results],
            dtype=np.float32,
        ),
        "uav_reward": np.array([r[1]["uav_reward"] for r in results], dtype=np.float32),
        "ugv_reward": np.array([r[1]["ugv_reward"] for r in results], dtype=np.float32),
    }
    terminateds = np.array([r[2] for r in results], dtype=bool)
    truncateds = np.array([r[3] for r in results], dtype=bool)
    infos = [r[4] for r in results]
    return _stack_vec_obs(obs_list), rewards, terminateds, truncateds, infos


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Grid-aligned distance for target-progress shaping."""
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())


def relative_distance_improvement(
    prev_dist: float,
    curr_dist: float,
    floor: float = 1.0,
) -> float:
    """Scale one-step distance change by the pre-move distance with a step-sized floor."""
    prev = float(prev_dist)
    curr = float(curr_dist)
    if not np.isfinite(prev) or not np.isfinite(curr):
        return 0.0
    denom = max(prev, float(max(floor, 1.0)))
    return float(np.clip((prev - curr) / denom, -1.0, 1.0))


@dataclass
class DataPacket:
    """Data packet in the transmission queue."""
    sample: SpectrumSample
    size_bits: float
    created_step: int = 0
    transmitted_bits: float = 0.0
    novelty_ratio: float = 1.0

    @property
    def is_complete(self) -> bool:
        return self.transmitted_bits >= self.size_bits


@dataclass
class PlannerTarget:
    """UGV-planned grid target with center frequency suggestion."""
    gx: int
    gy: int
    x: float
    y: float
    center_freq: int
    score: float


class UAVUGVEnvironment:
    """
    Multi-agent environment for UAV-UGV cooperative radio-map construction.
    """

    def __init__(
        self,
        config: Config,
        tensor_decomp: IIBTD_opt,
        sim_data: SimDataGen,
        scene_map: GridScene,
        minimal_info: bool = False,
    ):
        self.config = config
        self.td = tensor_decomp
        self.sim_data = sim_data
        self.scene = scene_map
        self.minimal_info = bool(minimal_info)

        self.Nx, self.Ny = config.scene.grid_size
        self.K = config.scene.total_freq_bands_nums

        self.target_count = int(config.planner.target_count)
        # Expose all actionable planner targets in observation.
        self.obs_target_slots = max(int(config.planner.obs_target_slots), self.target_count)
        self.bandwidth_ratios = np.asarray(config.uav.bandwidth_ratios, dtype=float)
        if self.bandwidth_ratios.size == 0:
            raise ValueError("config.uav.bandwidth_ratios must not be empty")
        self.num_bw_choices = int(self.bandwidth_ratios.size)
        self.uav_direction_choices = int(config.uav.num_directions)
        max_uav_direction_choices = len(UAV_DIRECTION_IDS)
        if self.uav_direction_choices < 1 or self.uav_direction_choices > max_uav_direction_choices:
            raise ValueError(
                f"config.uav.num_directions must be in [1, {max_uav_direction_choices}], "
                f"got {self.uav_direction_choices}"
            )
        self.uav_direction_ids = UAV_DIRECTION_IDS[:self.uav_direction_choices]
        self.ugv_action_size = int(config.ugv.num_directions)
        max_ugv_direction_choices = len(UGV_DIRECTION_IDS)
        if self.ugv_action_size < 1 or self.ugv_action_size > max_ugv_direction_choices:
            raise ValueError(
                f"config.ugv.num_directions must be in [1, {max_ugv_direction_choices}], "
                f"got {self.ugv_action_size}"
            )
        self.queue_ref = float(config.reward.q_ref)
        self.ugv_building_safe_clearance = max(
            1,
            int(config.reward.ugv_building_safe_clearance),
        )
        self.queue_capacity_packets = int(config.uav.queue_capacity_packets)
        self.ensemble_refresh_interval = max(1, int(config.planner.ensemble_refresh_interval))
        self.local_planner_radius = max(1, int(config.planner.local_planner_radius))
        self.target_mode = str(config.planner.target_mode).strip().lower()
        self.hybrid_enabled = self.target_mode == "hybrid"
        self.hybrid_nmse_stall_steps = max(1, int(config.planner.hybrid_nmse_stall_steps))
        self.hybrid_nmse_stall_threshold = float(config.planner.hybrid_nmse_stall_threshold)
        self.hybrid_global_hold_intervals = max(
            1,
            int(config.planner.hybrid_global_hold_intervals),
        )
        self.hybrid_local_reentry_min_targets = int(
            config.planner.hybrid_local_reentry_min_targets
        )
        self.initial_observation_mode = (
            str(config.planner.initial_observation_mode).strip().lower()
        )
        self.prefill_percent = float(config.planner.prefill_percent)

        # UAV action: choose movement direction + bandwidth split.
        self.uav_action_size = self.uav_direction_choices * self.num_bw_choices
        self._reset_counter = 0

        self._setup_observation_spaces()
        self._load_grid_dataset()
        self._init_cached_constants()

        self.ground_truth_map = self.sim_data.get_full_ground_truth_map()
        if hasattr(self.td, "set_ground_truth"):
            self.td.set_ground_truth(self.ground_truth_map)

    def _finalize_step_info(self, info: dict) -> dict:
        if not self.minimal_info:
            return info
        return {key: info[key] for key in _MINIMAL_STEP_INFO_KEYS if key in info}

    def _reset_episode_nmse_tracking(self) -> None:
        self.prev_nmse = float(self.radio_map_state.nmse)
        self.episode_nmse_start = float(self.radio_map_state.nmse)

    def _get_active_planner_mode(self) -> str:
        return str(self.planner_submode) if self.hybrid_enabled else str(self.target_mode)

    def _should_preserve_global_target(self) -> bool:
        return (
            bool(getattr(self, "planner_initialized", False))
            and self._get_active_planner_mode() == "global"
            and getattr(self, "active_plan_grid", None) is not None
        )

    def _get_hybrid_local_reentry_min_targets(self) -> int:
        return max(
            int(self.hybrid_local_reentry_min_targets),
            int(self.target_count),
        )

    def _get_hybrid_global_hold_steps(self) -> int:
        return int(self.hybrid_global_hold_intervals) * int(self.ensemble_refresh_interval)

    def _clear_global_switch_snapshot(self) -> None:
        self.global_switch_top_targets = []

    @staticmethod
    def _planner_target_from_dict(target: Dict[str, object]) -> PlannerTarget:
        return PlannerTarget(
            gx=int(target["gx"]),
            gy=int(target["gy"]),
            x=float(target["x"]),
            y=float(target["y"]),
            center_freq=int(target["center_freq"]),
            score=float(target["score"]),
        )

    def _clear_suppressed_planner_targets(self) -> None:
        self.suppressed_planner_targets = set()

    def _suppress_planner_target(
        self,
        target_grid: Optional[Tuple[int, int]],
    ) -> bool:
        if target_grid is None:
            return False
        gx, gy = int(target_grid[0]), int(target_grid[1])
        if not (0 <= gx < self.Nx and 0 <= gy < self.Ny):
            return False
        self.suppressed_planner_targets.add((gx, gy))
        return True

    def _build_target_radius_mask(
        self,
        center_cells: set[Tuple[int, int]],
        radius: float,
    ) -> np.ndarray:
        mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        if not center_cells:
            return mask

        radius = max(0.0, float(radius))
        radius_sq = radius * radius
        cell_radius = int(np.ceil(radius))
        for gx, gy in center_cells:
            gx = int(gx)
            gy = int(gy)
            x0 = max(0, gx - cell_radius)
            x1 = min(self.Nx, gx + cell_radius + 1)
            y0 = max(0, gy - cell_radius)
            y1 = min(self.Ny, gy + cell_radius + 1)
            dx = self.grid_x_coords[x0:x1, y0:y1] - float(gx)
            dy = self.grid_y_coords[x0:x1, y0:y1] - float(gy)
            mask[x0:x1, y0:y1] |= (dx * dx + dy * dy) <= radius_sq + 1e-9
        return mask

    def _build_suppressed_target_mask(self) -> np.ndarray:
        return self._build_target_radius_mask(
            self.suppressed_planner_targets,
            self.target_suppression_radius,
        )

    def _planner_target_is_suppressed(self, gx: int, gy: int) -> bool:
        if not self.suppressed_planner_targets:
            return False
        gx = int(gx)
        gy = int(gy)
        if self.target_suppression_radius <= 0.0:
            return (gx, gy) in self.suppressed_planner_targets
        radius_sq = float(self.target_suppression_radius) ** 2
        for sx, sy in self.suppressed_planner_targets:
            dx = float(gx - int(sx))
            dy = float(gy - int(sy))
            if dx * dx + dy * dy <= radius_sq + 1e-9:
                return True
        return False

    def _planner_reachability_horizon(self, planner_mode: Optional[str]) -> int:
        active_mode = (
            self._get_active_planner_mode()
            if planner_mode is None
            else str(planner_mode).strip().lower()
        )
        if active_mode == "global":
            distance_budget = self.global_manhattan_den
        else:
            distance_budget = float(self.local_planner_radius)
        return max(1, int(np.ceil(distance_budget / float(max(self.uav_step_count, 1)))) + 1)

    def _build_reachable_target_mask(self, planner_mode: Optional[str]) -> np.ndarray:
        reachable_endpoints = np.zeros((self.Nx, self.Ny), dtype=bool)
        start_pos = np.rint(
            np.clip(np.asarray(self.uav_pos, dtype=float), [0.0, 0.0], [self.Nx - 1, self.Ny - 1])
        )
        start_cell = self._grid_cell(start_pos)
        reachable_endpoints[start_cell] = True

        action_horizon = self._planner_reachability_horizon(planner_mode)
        frontier = deque([(start_pos.copy(), 0)])
        best_depth: Dict[Tuple[int, int], int] = {start_cell: 0}
        while frontier:
            pos, depth = frontier.popleft()
            if depth >= action_horizon:
                continue
            for direction_idx in self.uav_direction_ids:
                next_pos, moved_steps = self._rollout_direction(
                    position=pos,
                    direction_idx=int(direction_idx),
                    step_count=self.uav_step_count,
                    validator=self.scene.is_uav_position_valid,
                    stop_at_target=False,
                )
                if moved_steps <= 0:
                    continue
                next_cell = self._grid_cell(next_pos)
                next_depth = int(depth) + 1
                prev_best_depth = best_depth.get(next_cell)
                if prev_best_depth is not None and prev_best_depth <= next_depth:
                    continue
                best_depth[next_cell] = next_depth
                reachable_endpoints[next_cell] = True
                frontier.append((next_pos.copy(), next_depth))

        reachable_sampling = reachable_endpoints & self.sampling_valid_mask
        if self.target_arrival_radius <= 0.0:
            return reachable_sampling

        target_mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        endpoint_cells = np.argwhere(reachable_sampling)
        cell_radius = int(np.ceil(self.target_arrival_radius))
        radius_sq = float(self.target_arrival_radius) ** 2
        for gx, gy in endpoint_cells.tolist():
            x0 = max(0, int(gx) - cell_radius)
            x1 = min(self.Nx, int(gx) + cell_radius + 1)
            y0 = max(0, int(gy) - cell_radius)
            y1 = min(self.Ny, int(gy) + cell_radius + 1)
            dx = self.grid_x_coords[x0:x1, y0:y1] - float(gx)
            dy = self.grid_y_coords[x0:x1, y0:y1] - float(gy)
            target_mask[x0:x1, y0:y1] |= (dx * dx + dy * dy) <= radius_sq + 1e-9
        return target_mask

    def _filter_planner_candidate_mask(
        self,
        candidate_mask: np.ndarray,
        planner_mode: Optional[str],
    ) -> np.ndarray:
        base_mask = np.asarray(candidate_mask, dtype=bool) & self.sampling_valid_mask
        if not np.any(base_mask):
            return base_mask

        filtered_mask = base_mask
        reachable_mask = self._build_reachable_target_mask(planner_mode)
        reachable_filtered = filtered_mask & reachable_mask
        if np.any(reachable_filtered):
            filtered_mask = reachable_filtered

        if self.suppressed_planner_targets:
            unsuppressed = filtered_mask & ~self._build_suppressed_target_mask()
            if np.any(unsuppressed):
                filtered_mask = unsuppressed
        return filtered_mask

    def _select_planner_candidates(
        self,
        *,
        planner_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        redundancy_length: float = 0.0,
    ) -> List[PlannerTarget]:
        candidate_k = max(1, int(self.target_count if top_k is None else top_k))
        active_mode = (
            self._get_active_planner_mode()
            if planner_mode is None
            else str(planner_mode).strip().lower()
        )
        acquisition_space, _ = build_acquisition_space(
            var_map=self.latest_var_map,
            lambda_u=self.config.planner.lambda_u,
        )
        planner_candidate_mask = self._build_local_candidate_mask(
            center_pos=self.uav_pos,
            planner_mode=active_mode,
        )
        planner_candidate_mask = self._filter_planner_candidate_mask(
            planner_candidate_mask,
            planner_mode=active_mode,
        )
        target_dicts = select_top_k_grid_candidates(
            acquisition_space=acquisition_space,
            var_map=self.latest_var_map,
            sampled_mask=self.sampled_mask,
            action_visit=self.action_visit,
            top_k=candidate_k,
            beta_f=self.config.planner.beta_f,
            redundancy_length=float(redundancy_length),
            candidate_mask=planner_candidate_mask,
        )
        if target_dicts:
            return [self._planner_target_from_dict(target) for target in target_dicts]
        return self._build_fallback_targets(
            candidate_mask=planner_candidate_mask,
            top_k=candidate_k,
        )

    def _snapshot_global_switch_top_targets(self, top_k: int = 3) -> None:
        if self._get_active_planner_mode() != "global":
            self._clear_global_switch_snapshot()
            return
        self.global_switch_top_targets = self._select_planner_candidates(
            planner_mode="global",
            top_k=max(1, int(top_k)),
            redundancy_length=self.config.planner.redundancy_length,
        )

    def _build_global_switch_top_target_info(self, top_k: int = 3) -> Dict[str, object]:
        info: Dict[str, object] = {
            "global_topk_count": int(len(self.global_switch_top_targets)),
            "global_topk_active": int(bool(self.global_switch_top_targets)),
        }
        for rank in range(max(1, int(top_k))):
            prefix = f"global_top{rank + 1}"
            if rank < len(self.global_switch_top_targets):
                target = self.global_switch_top_targets[rank]
                info[f"{prefix}_x"] = int(target.gx)
                info[f"{prefix}_y"] = int(target.gy)
                info[f"{prefix}_freq"] = int(target.center_freq)
                info[f"{prefix}_score"] = float(target.score)
            else:
                info[f"{prefix}_x"] = -1
                info[f"{prefix}_y"] = -1
                info[f"{prefix}_freq"] = -1
                info[f"{prefix}_score"] = float("nan")
        return info

    def _count_local_reentry_candidates(self) -> int:
        local_mask = self._build_local_candidate_mask(
            center_pos=self.uav_pos,
            planner_mode="local",
        )
        unsampled_cells = np.any(~self.sampled_mask, axis=2)
        return int(np.sum(local_mask & unsampled_cells))

    def _switch_planner_submode(
        self,
        next_mode: str,
        hold_steps: Optional[int] = None,
    ) -> bool:
        if not self.hybrid_enabled:
            return False

        next_mode = str(next_mode).strip().lower()
        if next_mode not in {"local", "global"}:
            raise ValueError(f"Unsupported hybrid planner submode: {next_mode!r}")
        if next_mode == str(self.planner_submode):
            if next_mode == "global" and hold_steps is not None:
                self.global_steps_remaining = max(0, int(hold_steps))
            return False

        self.planner_submode = next_mode
        self.planner_stall_count = 0
        self.global_steps_remaining = (
            max(0, int(hold_steps))
            if next_mode == "global"
            else 0
        )
        self._clear_global_switch_snapshot()
        self.planner_targets = []
        self._clear_active_plan()
        return True

    def _update_hybrid_planner_submode(
        self,
        planner_submode_before_step: str,
        map_updated: bool,
    ) -> None:
        if not self.hybrid_enabled or not self.planner_initialized:
            return

        if planner_submode_before_step == "local" and self.planner_submode == "local":
            if map_updated:
                nmse_improvement = float(self.prev_nmse) - float(self.radio_map_state.nmse)
                if nmse_improvement < self.hybrid_nmse_stall_threshold:
                    self.planner_stall_count += 1
                else:
                    self.planner_stall_count = 0
                if self.planner_stall_count >= self.hybrid_nmse_stall_steps:
                    switched = self._switch_planner_submode(
                        "global",
                        hold_steps=self._get_hybrid_global_hold_steps(),
                    )
                    if switched:
                        self._snapshot_global_switch_top_targets(top_k=3)
                        self._start_new_grid_plan()
                    return

        if planner_submode_before_step == "global" and self.planner_submode == "global":
            self.global_steps_remaining = max(self.global_steps_remaining - 1, 0)
            if self.global_steps_remaining <= 0:
                candidate_count = self._count_local_reentry_candidates()
                if candidate_count >= self._get_hybrid_local_reentry_min_targets():
                    switched = self._switch_planner_submode("local")
                    if switched:
                        self._start_new_grid_plan()
                    return

    def _setup_observation_spaces(self) -> None:
        c = self.config
        self.ugv_building_obs_dim = len(_UGV_BUILDING_OBS_DIRECTIONS)
        # uav: position(2) + energy(1) + queue(1) + bw_ratio(1)
        #    + local_goal(dx, dy, dist, center_freq) + ugv_position(2)
        self.uav_obs_dim = 2 + 1 + 1 + 1 + 4 + 2
        # ugv: self_position(2) + rel_uav(dx, dy) + rel_target(dx, dy)
        #    + target_dist(1) + queue(1) + snr(1) + los_link(1)
        #    + building clearance for stay/east/north/west/south action endpoints
        self.ugv_obs_dim = 2 + 2 + 2 + 1 + 1 + 1 + 1 + self.ugv_building_obs_dim
        # state: uav_position(2) + ugv_position(2) + uav_energy(1) + snr(1)
        #    + queue(1) + nmse(1) + bw_ratio(1) + los_link(1)
        #    + current UGV building clearance(1) + planner_state(5)
        self.critic_state_dim = (
            4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + c.obs.num_planner_features
        )

    def get_obs_dims(self) -> Dict[str, int]:
        return {
            "uav_obs": self.uav_obs_dim,
            "ugv_obs": self.ugv_obs_dim,
            "critic_state": self.critic_state_dim,
        }

    def get_action_dims(self) -> Dict[str, int]:
        return {
            "uav_action": self.uav_action_size,
            "uav_direction": self.uav_direction_choices,
            "uav_bandwidth": self.num_bw_choices,
            "ugv_action": self.ugv_action_size,
        }

    def _load_grid_dataset(self) -> None:
        self.grid_points = np.asarray(self.sim_data.grid_coords, dtype=float)
        self.I_mask = np.asarray(self.sim_data.I_mask, dtype=bool)
        self.bounds = self.sim_data.bounds

        expected_grid_points = self.Nx * self.Ny
        if self.grid_points.shape != (expected_grid_points, 2):
            raise ValueError(
                f"grid_coords shape {self.grid_points.shape} does not match full grid "
                f"({expected_grid_points}, 2)"
            )

        self.grid_index_positions = self.grid_points.astype(int)

    def _init_cached_constants(self) -> None:
        self.grid_norm_den = np.array([self.Nx - 1, self.Ny - 1], dtype=float)
        self.safe_grid_norm_den = np.array(
            [max(self.Nx - 1, 1), max(self.Ny - 1, 1)],
            dtype=float,
        )
        self.max_grid_x_den = float(max(self.Nx - 1, 1))
        self.max_grid_y_den = float(max(self.Ny - 1, 1))
        self.global_manhattan_den = float(max((self.Nx - 1) + (self.Ny - 1), 1))
        self.max_freq_den = float(max(self.K - 1, 1))
        self.max_grid_diag = float(np.hypot(self.max_grid_x_den, self.max_grid_y_den))
        self.uav_step_count = int(self._grid_step_count(float(self.config.uav.step_size)))
        self.ugv_step_count = int(self._grid_step_count(float(self.config.ugv.step_size)))
        self.target_arrival_radius = (
            max(0.0, float(self.config.planner.target_arrival_radius_steps))
            * float(max(self.uav_step_count, 1))
        )
        self.target_suppression_radius = (
            max(0.0, float(self.config.planner.target_suppression_radius_steps))
            * float(max(self.uav_step_count, 1))
        )
        self.target_stuck_no_sample_steps = max(
            1,
            int(self.config.planner.target_stuck_no_sample_steps),
        )
        self.max_bw_ratio = float(np.max(self.bandwidth_ratios))
        total_bw_units = int(self.config.uav.total_bw_num)
        self.max_sensing_units = int(
            np.clip(np.ceil(total_bw_units * self.max_bw_ratio), 1, total_bw_units - 1)
        )
        self.max_q = int(max(1, self.queue_capacity_packets))
        self.max_packet_bits = (
            float(self.max_sensing_units) * float(self.config.comm.data_per_sample)
        )
        self.q_max_bits = float(max(self.queue_ref, 1e-8)) * self.max_packet_bits
        self.uav_energy_den = float(self.config.uav.max_energy)
        self.snr_norm_den = 30.0
        self.walkable_mask = ~self.scene.get_occupancy_grid()
        self.sampling_valid_mask = self.walkable_mask.copy()
        self.grid_x_coords, self.grid_y_coords = np.meshgrid(
            np.arange(self.Nx, dtype=float),
            np.arange(self.Ny, dtype=float),
            indexing="ij",
        )
        self._clearance_mask_cache: Dict[int, np.ndarray] = {0: self.sampling_valid_mask.copy()}
        self._clearance_level_cache: Dict[int, np.ndarray] = {
            0: np.zeros((self.Nx, self.Ny), dtype=np.int16)
        }
        self._ugv_shortest_path_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._uav_action_mask_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        self._ugv_action_mask_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._mask_cache_max_entries = 8192
        self.ugv_progress_uav_weight = float(self.config.reward.ugv_progress_uav_weight)
        self.ugv_progress_target_weight = float(self.config.reward.ugv_progress_target_weight)

    def _init_reset_rng(self, seed: Optional[int]) -> None:
        if seed is not None:
            local_seed = int(seed)
        else:
            local_seed = int(self.config.mappo.seed + self._reset_counter)
            self._reset_counter += 1
        self.rng = np.random.RandomState(local_seed)
        # Keep channel/sensing stochasticity aligned with reset seeds so
        # seeded evaluation episodes are fully reproducible.
        sim_seed = int(local_seed + 7919)
        if hasattr(self.sim_data, "reset_rng"):
            self.sim_data.reset_rng(sim_seed)
        elif hasattr(self.sim_data, "rng"):
            self.sim_data.rng = np.random.RandomState(sim_seed)

    def _is_uav_sampling_position_valid(self, position: np.ndarray) -> bool:
        return self.scene.is_non_building_position_valid(position)

    def _grid_cell(self, position: np.ndarray) -> Tuple[int, int]:
        pos = np.asarray(position, dtype=float).reshape(2)
        gx = int(np.clip(np.rint(pos[0]), 0, self.Nx - 1))
        gy = int(np.clip(np.rint(pos[1]), 0, self.Ny - 1))
        return gx, gy

    def _is_walkable_cell(self, cell: Tuple[int, int]) -> bool:
        gx, gy = int(cell[0]), int(cell[1])
        return bool(self.walkable_mask[gx, gy])

    def _cache_store(self, cache: Dict, key, value) -> None:
        if len(cache) >= self._mask_cache_max_entries:
            cache.clear()
        cache[key] = value

    def _build_clearance_mask(self, clearance: int) -> np.ndarray:
        clearance = max(0, int(clearance))
        if clearance <= 0:
            return self.sampling_valid_mask.copy()

        base = self.sampling_valid_mask
        mask = base.copy()
        for dx in range(-clearance, clearance + 1):
            for dy in range(-clearance, clearance + 1):
                shifted = np.zeros_like(base, dtype=bool)
                src_x0 = max(0, -dx)
                src_x1 = min(self.Nx, self.Nx - dx)
                src_y0 = max(0, -dy)
                src_y1 = min(self.Ny, self.Ny - dy)
                dst_x0 = max(0, dx)
                dst_x1 = min(self.Nx, self.Nx + dx)
                dst_y0 = max(0, dy)
                dst_y1 = min(self.Ny, self.Ny + dy)
                shifted[dst_x0:dst_x1, dst_y0:dst_y1] = base[src_x0:src_x1, src_y0:src_y1]
                mask &= shifted
        return mask

    def _get_clearance_mask(self, clearance: int) -> np.ndarray:
        clearance = max(0, int(clearance))
        cached = self._clearance_mask_cache.get(clearance)
        if cached is not None:
            return cached
        mask = self._build_clearance_mask(clearance)
        self._clearance_mask_cache[clearance] = mask
        return mask

    def _build_clearance_level_map(self, max_clearance: int) -> np.ndarray:
        max_clearance = max(0, int(max_clearance))
        levels = np.zeros((self.Nx, self.Ny), dtype=np.int16)
        for clearance in range(1, max_clearance + 1):
            levels[self._get_clearance_mask(clearance)] = clearance
        return levels

    def _get_clearance_level_map(self, max_clearance: int) -> np.ndarray:
        max_clearance = max(0, int(max_clearance))
        cached = self._clearance_level_cache.get(max_clearance)
        if cached is not None:
            return cached
        levels = self._build_clearance_level_map(max_clearance)
        self._clearance_level_cache[max_clearance] = levels
        return levels

    def _select_high_clearance_mask(
        self,
        base_mask: np.ndarray,
        preferred_clearance: int,
    ) -> Tuple[np.ndarray, int]:
        candidate_mask = np.asarray(base_mask, dtype=bool) & self.sampling_valid_mask
        if not np.any(candidate_mask):
            return candidate_mask, 0
        preferred_clearance = max(0, int(preferred_clearance))
        if preferred_clearance <= 0:
            return candidate_mask, 0
        clearance_levels = self._get_clearance_level_map(preferred_clearance)
        best_clearance = int(np.max(clearance_levels[candidate_mask]))
        return candidate_mask & (clearance_levels == best_clearance), best_clearance

    def _ugv_building_clearance_level(self, position: np.ndarray) -> int:
        gx, gy = self._grid_cell(position)
        levels = self._get_clearance_level_map(self.ugv_building_safe_clearance)
        return int(levels[gx, gy])

    def _ugv_building_clearance_norm(self, position: np.ndarray) -> float:
        level = self._ugv_building_clearance_level(position)
        return float(np.clip(level / float(self.ugv_building_safe_clearance), 0.0, 1.0))

    def _build_ugv_building_obs(self) -> np.ndarray:
        features = []
        for direction_idx in _UGV_BUILDING_OBS_DIRECTIONS:
            next_pos, _ = self._rollout_direction(
                position=self.ugv_pos,
                direction_idx=int(direction_idx),
                step_count=self.ugv_step_count,
                validator=self.scene.is_ugv_position_valid,
            )
            features.append(self._ugv_building_clearance_norm(next_pos))
        return np.asarray(features, dtype=float)

    def _get_ugv_shortest_path_distances(self, target_cell: Tuple[int, int]) -> np.ndarray:
        target = (int(target_cell[0]), int(target_cell[1]))
        cached = self._ugv_shortest_path_cache.get(target)
        if cached is not None:
            return cached

        dist = np.full((self.Nx, self.Ny), np.inf, dtype=float)
        if not self._is_walkable_cell(target):
            self._ugv_shortest_path_cache[target] = dist
            return dist

        dist[target] = 0.0
        frontier = deque([target])
        while frontier:
            gx, gy = frontier.popleft()
            next_dist = float(dist[gx, gy] + 1.0)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = gx + dx
                ny = gy + dy
                if not (0 <= nx < self.Nx and 0 <= ny < self.Ny):
                    continue
                if not self.walkable_mask[nx, ny]:
                    continue
                if next_dist >= dist[nx, ny]:
                    continue
                dist[nx, ny] = next_dist
                frontier.append((nx, ny))

        self._ugv_shortest_path_cache[target] = dist
        return dist

    def _occupancy_shortest_path_distance(
        self,
        source_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> float:
        source_cell = self._grid_cell(source_pos)
        target_cell = self._grid_cell(target_pos)
        if not self._is_walkable_cell(source_cell):
            return float("inf")
        dist_map = self._get_ugv_shortest_path_distances(target_cell)
        return float(dist_map[source_cell])

    def _sample_initial_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        max_sep = float(max(1e-6, self.config.planner.init_pair_max_distance))
        center = np.array([(self.Nx - 1) / 2.0, (self.Ny - 1) / 2.0], dtype=np.float64)
        preferred_clearance = max(0, int(self.config.planner.init_building_clearance))
        clearance_levels = self._get_clearance_level_map(preferred_clearance)

        def _valid_nearby_ugv_positions(anchor: np.ndarray) -> List[np.ndarray]:
            x_min = max(0, int(np.floor(anchor[0] - max_sep)))
            x_max = min(self.Nx - 1, int(np.ceil(anchor[0] + max_sep)))
            y_min = max(0, int(np.floor(anchor[1] - max_sep)))
            y_max = min(self.Ny - 1, int(np.ceil(anchor[1] + max_sep)))
            nearby: List[np.ndarray] = []
            for gx in range(x_min, x_max + 1):
                for gy in range(y_min, y_max + 1):
                    cand = np.array([gx, gy], dtype=np.float64)
                    if np.linalg.norm(cand - anchor) > max_sep + 1e-9:
                        continue
                    if self.scene.is_ugv_position_valid(cand):
                        nearby.append(cand)
            if nearby and preferred_clearance > 0:
                best_clearance = max(
                    int(clearance_levels[int(cand[0]), int(cand[1])]) for cand in nearby
                )
                nearby = [
                    cand
                    for cand in nearby
                    if int(clearance_levels[int(cand[0]), int(cand[1])]) == best_clearance
                ]
            return nearby

        preferred_uav_mask, _ = self._select_high_clearance_mask(
            base_mask=self.sampling_valid_mask,
            preferred_clearance=preferred_clearance,
        )
        preferred_uav_cells = np.argwhere(preferred_uav_mask)
        if preferred_uav_cells.size > 0:
            center_dist = np.linalg.norm(
                preferred_uav_cells.astype(np.float64) - center[np.newaxis, :],
                axis=1,
            )
            sample_weights = 1.0 / (1.0 + center_dist)
            sample_weights /= np.sum(sample_weights)
        else:
            sample_weights = np.array([], dtype=np.float64)

        for _ in range(max(256, 4 * max(1, preferred_uav_cells.shape[0]))):
            if preferred_uav_cells.size == 0:
                break
            sample_idx = int(self.rng.choice(preferred_uav_cells.shape[0], p=sample_weights))
            gx, gy = preferred_uav_cells[sample_idx]
            uav = np.array([gx, gy], dtype=np.float64)
            nearby = _valid_nearby_ugv_positions(uav)
            if not nearby:
                continue
            ugv = nearby[self.rng.randint(0, len(nearby))]
            return uav, ugv.copy()

        # Deterministic fallback independent of any hand-set start position.
        ranked_uav_candidates = sorted(
            (
                np.array([gx, gy], dtype=np.float64)
                for gx in range(self.Nx)
                for gy in range(self.Ny)
                if self._is_uav_sampling_position_valid(np.array([gx, gy], dtype=np.float64))
            ),
            key=lambda pos: (
                -int(clearance_levels[int(pos[0]), int(pos[1])]),
                float(np.linalg.norm(pos - center)),
            ),
        )
        for uav in ranked_uav_candidates:
            nearby = _valid_nearby_ugv_positions(uav)
            if nearby:
                ugv = min(nearby, key=lambda pos: float(np.linalg.norm(pos - uav)))
                return uav.copy(), ugv.copy()

        raise RuntimeError("Failed to find valid fallback initial positions.")

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], dict]:
        self._init_reset_rng(seed)

        self.uav_pos, self.ugv_pos = self._sample_initial_positions()
        self.uav_energy = float(self.config.uav.max_energy)

        self.td.reset()
        if hasattr(self.td, "set_ground_truth"):
            self.td.set_ground_truth(self.ground_truth_map)

        self.current_step = 0
        self._set_bandwidth_info(float(self.config.uav.default_bw_ratio))

        self.uav_data_queue: List[DataPacket] = []
        self.delivered_samples: List[SpectrumSample] = []
        self.pending_ensemble_samples: List[SpectrumSample] = []
        self.pending_ensemble_sample_count = 0
        self.pending_ensemble_effective_sample_count = 0
        self.last_sample_center_freq = -1
        self.map_update_count = 0
        self.planner_initialized = False
        self.planner_submode = "local" if self.hybrid_enabled else str(self.target_mode)
        self.planner_stall_count = 0
        self.global_steps_remaining = 0
        self.active_target_nmse_record: Optional[Dict[str, object]] = None
        self.completed_target_nmse_records: List[Dict[str, object]] = []
        self.last_completed_target_nmse_record: Optional[Dict[str, object]] = None
        self.reconstruction_events: List[Dict[str, object]] = []
        self.ensemble_events: List[Dict[str, object]] = []
        self.bootstrap_events: List[Dict[str, object]] = []
        self.global_switch_top_targets: List[PlannerTarget] = []

        self.sampled_mask = np.zeros((self.Nx, self.Ny, self.K), dtype=bool)
        self.action_visit = np.zeros((self.Nx, self.Ny, self.K), dtype=float)
        # Episode-level revisit counts should persist across planner target switches.
        self.local_spatial_visit = np.zeros((self.Nx, self.Ny), dtype=float)
        self.bootstrap_target_history: set[Tuple[int, int]] = set()
        self.suppressed_planner_targets: set[Tuple[int, int]] = set()
        self.last_target_retarget_reason = ""

        self.planner_targets: List[PlannerTarget] = []
        self._reset_grid_plan_state()

        self.total_collected_samples = 0
        self.latest_var_map = np.ones((self.Nx, self.Ny, self.K), dtype=float)
        self.latest_mean_map = np.zeros((self.Nx, self.Ny, self.K), dtype=float)

        self.radio_map_state = RadioMapState(
            spectrum_map=np.zeros((self.Nx, self.Ny, self.K), dtype=float),
            nmse=1.0,
            last_update_step=0,
        )
        self.uncertainty = UncertaintyMap(
            spatial_uncertainty=np.ones((self.Nx, self.Ny), dtype=float),
            frequency_uncertainty=np.ones(self.K, dtype=float),
            joint_uncertainty=np.ones((self.Nx, self.Ny, self.K), dtype=float),
        )
        self._reset_episode_nmse_tracking()
        self._initialize_planner_warmup_state()
        self._reset_episode_nmse_tracking()
        self.last_sample_center_freq = -1

        self.ugv_channel_info = self._get_channel_info()

        obs = self._build_observations()
        current_target = self._get_current_observation_target()
        info = {
            "nmse": self.radio_map_state.nmse,
            "target_count": len(self.planner_targets),
            "planner_initialized": int(self.planner_initialized),
            "planner_sample_count": int(self._planner_sample_count()),
            "planner_effective_sample_count": int(self._planner_effective_sample_count()),
            "pending_ensemble_sample_count": int(self.pending_ensemble_sample_count),
            "pending_ensemble_effective_sample_count": int(
                self.pending_ensemble_effective_sample_count
            ),
            "planner_submode": str(self._get_active_planner_mode()),
            "planner_mode_switch": "",
            "planner_switched_to_global": 0,
            "planner_stall_count": int(self.planner_stall_count),
            "planner_global_steps_remaining": int(self.global_steps_remaining),
            "initial_observation_mode": str(self.initial_observation_mode),
            "target_grid_x": int(current_target.gx) if current_target is not None else -1,
            "target_grid_y": int(current_target.gy) if current_target is not None else -1,
            "target_center_freq": int(current_target.center_freq) if current_target is not None else -1,
            "target_source": str(self._get_current_target_source()),
            "bootstrap_active": int(self._bootstrap_target_is_active()),
            "prefill_percent": float(self.prefill_percent),
            "prefill_budget_basis": int(self.prefill_budget_basis_count),
            "prefill_sample_count": int(self.prefill_sample_count),
            "prefill_applied": int(self.prefill_applied),
            **self._build_global_switch_top_target_info(top_k=3),
        }
        return obs, info

    def close(self) -> None:
        close_fn = getattr(self.td, "close", None)
        if callable(close_fn):
            close_fn()

    def _reset_grid_plan_state(self) -> None:
        """Reset active single-grid planning state."""
        self.active_plan_grid: Optional[Tuple[int, int]] = None
        self.active_plan_center_freq: Optional[int] = None
        self.active_plan_start_step = int(getattr(self, "current_step", 0))
        self.active_plan_no_sample_steps = 0

    def _resolve_prefill_budget_basis(self) -> int:
        basis = int(self.config.planner.prefill_budget_basis)
        if basis <= 0:
            basis = int(self.config.mappo.episode_max_steps)
        return max(1, basis)

    def _compute_prefill_sample_count(self) -> int:
        candidate_count = int(np.sum(self.sampling_valid_mask))
        if candidate_count <= 0:
            return 0
        budget_basis = min(self._resolve_prefill_budget_basis(), candidate_count)
        percent = float(np.clip(self.prefill_percent, 0.0, 100.0))
        if percent <= 0.0:
            return 0
        return int(
            np.clip(
                round(float(budget_basis) * (percent / 100.0)),
                1,
                budget_basis,
            )
        )

    def _build_uniform_prefill_center_freqs(self, target_count: int) -> np.ndarray:
        """Spread prefill sensing windows across the full frequency axis."""
        count = max(0, int(target_count))
        if count <= 0:
            return np.empty(0, dtype=int)

        width = int(np.clip(self.sensing_band_num, 1, self.K))
        max_start = max(0, int(self.K) - width)
        coverage = np.zeros(self.K, dtype=int)
        centers: List[int] = []
        for _ in range(count):
            best_start = 0
            best_score = None
            best_coverage = None
            for start in range(max_start + 1):
                candidate_coverage = coverage.copy()
                candidate_coverage[start : start + width] += 1
                score = (
                    int(np.max(candidate_coverage) - np.min(candidate_coverage)),
                    float(np.var(candidate_coverage)),
                    int(np.max(candidate_coverage)),
                    -int(np.count_nonzero(candidate_coverage)),
                    int(start),
                )
                if best_score is None or score < best_score:
                    best_start = int(start)
                    best_score = score
                    best_coverage = candidate_coverage
            coverage = best_coverage
            centers.append(int(np.clip(best_start + (width // 2), 0, self.K - 1)))
        return np.asarray(centers, dtype=int)

    def _apply_random_prefill(self) -> None:
        candidate_cells = np.argwhere(self.sampling_valid_mask)
        if candidate_cells.size == 0:
            raise RuntimeError("Prefill candidate set is empty: no non-building sampling cells.")

        target_count = min(int(self.prefill_sample_count), int(candidate_cells.shape[0]))
        if target_count <= 0:
            return

        selected_indices = self.rng.choice(
            candidate_cells.shape[0],
            size=target_count,
            replace=False,
        )
        selected_cells = np.asarray(candidate_cells[selected_indices], dtype=int)
        center_freqs = self._build_uniform_prefill_center_freqs(target_count)
        prefill_samples: List[SpectrumSample] = []

        for (gx, gy), center_freq in zip(selected_cells.tolist(), center_freqs.tolist()):
            sample, _, _, _ = self._collect_grid_sample(
                position=np.array([int(gx), int(gy)], dtype=float),
                center_freq=int(center_freq),
            )
            if sample is not None:
                prefill_samples.append(sample)

        if not prefill_samples:
            return

        self._stage_pending_ensemble_samples(prefill_samples)
        self.prefill_sample_count = int(len(prefill_samples))
        self.prefill_applied = True

        self._run_ensemble_map_update(
            reason="prefill_init",
            seed_offset=0,
            refresh_targets=True,
        )
        self.planner_initialized = bool(self.planner_targets)
        if self.planner_initialized:
            self._clear_active_plan()
            self._start_new_grid_plan()

    def _initialize_planner_warmup_state(self) -> None:
        self.bootstrap_target = None
        self.bootstrap_target_reached_once = False
        self.bootstrap_target_start_step = int(self.current_step)
        self.prefill_budget_basis_count = int(self._resolve_prefill_budget_basis())
        self.prefill_sample_count = 0
        self.prefill_applied = False

        if self.initial_observation_mode == "prefill":
            self.prefill_sample_count = int(self._compute_prefill_sample_count())
            self._apply_random_prefill()
            return
        self._init_bootstrap_target()

    def _build_bootstrap_target(self) -> Optional[PlannerTarget]:
        """Choose a reachable pre-planner target that expands spatial coverage."""
        action_horizon = self._bootstrap_target_refresh_horizon()
        preferred_center_freq = int(np.clip(self.K // 2, 0, self.K - 1))
        center_x = (self.Nx - 1) / 2.0
        center_y = (self.Ny - 1) / 2.0
        min_scene_dim = float(min(self.Nx, self.Ny))
        preferred_edge_margin = float(
            min(
                max(2, int(np.round(0.12 * min_scene_dim))),
                max(int(np.floor((min_scene_dim - 1.0) / 2.0)), 0),
            )
        )
        preferred_min_dist = float(max(2.0, 0.18 * min_scene_dim))
        max_center_dist = float(np.hypot(center_x, center_y))
        preferred_clearance = max(0, int(self.config.planner.bootstrap_building_clearance))
        sampled_spatial_mask = np.any(self.sampled_mask, axis=2)
        unsampled_band_fraction = np.mean(~self.sampled_mask, axis=2)
        dist_uav = np.abs(self.grid_x_coords - float(self.uav_pos[0])) + np.abs(
            self.grid_y_coords - float(self.uav_pos[1])
        )
        reachable_mask, reachable_action_steps = self._enumerate_bootstrap_reachable_cells(
            action_horizon=action_horizon,
        )
        edge_margin = np.minimum.reduce(
            [
                self.grid_x_coords,
                self.grid_y_coords,
                self.max_grid_x_den - self.grid_x_coords,
                self.max_grid_y_den - self.grid_y_coords,
            ]
        )
        center_dist = np.hypot(self.grid_x_coords - center_x, self.grid_y_coords - center_y)
        center_bonus = 1.0 - (center_dist / (max_center_dist + 1e-9))
        edge_bonus = np.minimum(edge_margin / max(preferred_edge_margin, 1.0), 1.0)
        distance_bonus = np.minimum(dist_uav / max(preferred_min_dist, 1.0), 1.0)
        interior_bonus = (edge_margin >= preferred_edge_margin).astype(float)
        base_score = (
            3.0 * interior_bonus
            + 1.8 * center_bonus
            + 0.8 * edge_bonus
            + 0.4 * distance_bonus
        )
        candidate_base_mask = self.sampling_valid_mask & reachable_mask & (dist_uav >= 1.0)
        if not np.any(candidate_base_mask):
            candidate_base_mask = self.sampling_valid_mask & (dist_uav >= 1.0)

        # Bootstrap only needs to collect enough distinct spatial samples to warm up
        # the planner, so prefer cells that have not been spatially visited yet.
        unvisited_spatial_mask = ~sampled_spatial_mask
        if np.any(candidate_base_mask & unvisited_spatial_mask):
            candidate_base_mask &= unvisited_spatial_mask

        history_dist = None
        if self.bootstrap_target_history:
            history_points = np.asarray(
                [
                    (int(gx), int(gy))
                    for gx, gy in self.bootstrap_target_history
                    if 0 <= int(gx) < self.Nx and 0 <= int(gy) < self.Ny
                ],
                dtype=float,
            )
            if history_points.size > 0:
                history_x = history_points[:, 0][:, np.newaxis, np.newaxis]
                history_y = history_points[:, 1][:, np.newaxis, np.newaxis]
                history_dist = np.min(
                    np.abs(self.grid_x_coords[np.newaxis, :, :] - history_x)
                    + np.abs(self.grid_y_coords[np.newaxis, :, :] - history_y),
                    axis=0,
                )
                preferred_history_sep = float(max(1, self.uav_step_count))
                history_far_mask = history_dist >= preferred_history_sep
                if np.any(candidate_base_mask & history_far_mask):
                    candidate_base_mask &= history_far_mask

        candidate_mask, matched_clearance = self._select_high_clearance_mask(
            base_mask=candidate_base_mask,
            preferred_clearance=preferred_clearance,
        )
        if not np.any(candidate_mask):
            return None

        clearance_bonus = (
            float(matched_clearance) / float(max(preferred_clearance, 1))
            if preferred_clearance > 0
            else 0.0
        )
        reachability_bonus = np.zeros((self.Nx, self.Ny), dtype=float)
        finite_reach = np.isfinite(reachable_action_steps) & (reachable_action_steps > 0.0)
        if np.any(finite_reach):
            reach_den = float(max(action_horizon - 1, 1))
            reachability_bonus[finite_reach] = 1.0 - np.clip(
                (reachable_action_steps[finite_reach] - 1.0) / reach_den,
                0.0,
                1.0,
            )
        history_bonus = np.ones((self.Nx, self.Ny), dtype=float)
        if history_dist is not None:
            history_bonus = np.minimum(
                history_dist / float(max(self.uav_step_count, 1)),
                1.0,
            )
        score = (
            base_score
            + 1.5 * clearance_bonus
            + 2.0 * unvisited_spatial_mask.astype(float)
            + 1.2 * unsampled_band_fraction
            + 0.9 * history_bonus
            + 0.8 * reachability_bonus
        )
        masked_score = np.where(candidate_mask, score, -np.inf)
        flat_idx = int(np.argmax(masked_score))
        gx, gy = np.unravel_index(flat_idx, masked_score.shape)
        best_score = float(masked_score[gx, gy])
        if not np.isfinite(best_score):
            return None
        return PlannerTarget(
            gx=int(gx),
            gy=int(gy),
            x=float(gx),
            y=float(gy),
            center_freq=preferred_center_freq,
            score=best_score,
        )

    def _enumerate_bootstrap_reachable_cells(
        self,
        action_horizon: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if action_horizon is None:
            action_horizon = self._bootstrap_target_refresh_horizon()
        action_horizon = max(1, int(action_horizon))

        reachable_mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        reachable_action_steps = np.full((self.Nx, self.Ny), np.inf, dtype=float)
        start_pos = np.rint(
            np.clip(np.asarray(self.uav_pos, dtype=float), [0.0, 0.0], [self.Nx - 1, self.Ny - 1])
        )
        start_cell = self._grid_cell(start_pos)
        reachable_mask[start_cell] = True
        reachable_action_steps[start_cell] = 0.0

        frontier = deque([(start_pos.copy(), 0)])
        best_depth: Dict[Tuple[int, int], int] = {start_cell: 0}
        while frontier:
            pos, depth = frontier.popleft()
            if depth >= action_horizon:
                continue
            for direction_idx in self.uav_direction_ids:
                next_pos, moved_steps = self._rollout_direction(
                    position=pos,
                    direction_idx=int(direction_idx),
                    step_count=self.uav_step_count,
                    validator=self.scene.is_uav_position_valid,
                    stop_at_target=False,
                )
                if moved_steps <= 0:
                    continue
                next_cell = self._grid_cell(next_pos)
                next_depth = int(depth) + 1
                prev_best_depth = best_depth.get(next_cell)
                if prev_best_depth is not None and prev_best_depth <= next_depth:
                    continue
                best_depth[next_cell] = next_depth
                reachable_mask[next_cell] = True
                reachable_action_steps[next_cell] = float(next_depth)
                frontier.append((next_pos.copy(), next_depth))
        return reachable_mask, reachable_action_steps

    def _record_bootstrap_event(
        self,
        event: str,
        reason: str,
        target: Optional[PlannerTarget] = None,
    ) -> Dict[str, object]:
        target = target if target is not None else self.bootstrap_target
        event_record = {
            "event": str(event),
            "reason": str(reason),
            "step": int(self.current_step),
            "nmse": float(self.radio_map_state.nmse),
            "planner_sample_count": int(self._planner_sample_count()),
            "planner_effective_sample_count": int(self._planner_effective_sample_count()),
            "uav_pos": [float(self.uav_pos[0]), float(self.uav_pos[1])],
            "ugv_pos": [float(self.ugv_pos[0]), float(self.ugv_pos[1])],
            "target_grid_x": int(target.gx) if target is not None else -1,
            "target_grid_y": int(target.gy) if target is not None else -1,
            "target_center_freq": int(target.center_freq) if target is not None else -1,
        }
        self.bootstrap_events.append(event_record)
        return event_record

    def _init_bootstrap_target(self) -> None:
        self.bootstrap_target = self._build_bootstrap_target()
        self.bootstrap_target_reached_once = False
        self.bootstrap_target_start_step = int(self.current_step)
        if self.bootstrap_target is not None:
            self.bootstrap_target_history.add((int(self.bootstrap_target.gx), int(self.bootstrap_target.gy)))
            self._record_bootstrap_event(
                event="bootstrap_start",
                reason="reset",
                target=self.bootstrap_target,
            )

    def _bootstrap_target_is_active(self) -> bool:
        return (not self.planner_initialized) and self.bootstrap_target is not None

    def _complete_bootstrap_phase(self, reason: str) -> Optional[Dict[str, object]]:
        if self.bootstrap_target is None:
            return None
        event = self._record_bootstrap_event(
            event="bootstrap_handoff",
            reason=reason,
            target=self.bootstrap_target,
        )
        self.bootstrap_target = None
        self.bootstrap_target_reached_once = False
        self.bootstrap_target_start_step = int(self.current_step)
        return event

    def _retarget_bootstrap_phase(self, reason: str) -> Optional[Dict[str, object]]:
        self.bootstrap_target = self._build_bootstrap_target()
        self.bootstrap_target_reached_once = False
        self.bootstrap_target_start_step = int(self.current_step)
        if self.bootstrap_target is None:
            return None
        self.bootstrap_target_history.add((int(self.bootstrap_target.gx), int(self.bootstrap_target.gy)))
        return self._record_bootstrap_event(
            event="bootstrap_retarget",
            reason=reason,
            target=self.bootstrap_target,
        )

    def _bootstrap_target_refresh_horizon(self) -> int:
        return max(1, int(self.config.planner.ensemble_refresh_interval) + 1)

    def _bootstrap_target_timed_out(self) -> bool:
        if not self._bootstrap_target_is_active():
            return False
        elapsed = int(self.current_step) - int(self.bootstrap_target_start_step)
        return elapsed >= self._bootstrap_target_refresh_horizon()

    def _get_motion_target_grid(self) -> Optional[Tuple[int, int]]:
        if self.active_plan_grid is not None:
            return (int(self.active_plan_grid[0]), int(self.active_plan_grid[1]))
        if self._bootstrap_target_is_active():
            return (int(self.bootstrap_target.gx), int(self.bootstrap_target.gy))
        return None

    def _get_current_target_source(self) -> str:
        if self.active_plan_grid is not None or self.planner_targets:
            return "planner"
        if self._bootstrap_target_is_active():
            return "bootstrap"
        return "none"

    def _target_arrival_distance(
        self,
        target_grid: Optional[Tuple[int, int]],
    ) -> float:
        if target_grid is None:
            return -1.0
        target_pos = np.array([float(target_grid[0]), float(target_grid[1])], dtype=float)
        return float(np.linalg.norm(self.uav_pos - target_pos))

    def _update_active_plan_no_sample_counter(self, sample: Optional[SpectrumSample]) -> None:
        if self.active_plan_grid is None:
            self.active_plan_no_sample_steps = 0
            return
        if sample is None:
            self.active_plan_no_sample_steps += 1
        else:
            self.active_plan_no_sample_steps = 0

    def _active_plan_is_stuck_without_samples(self) -> bool:
        return bool(
            self.active_plan_grid is not None
            and self.active_plan_no_sample_steps >= self.target_stuck_no_sample_steps
        )

    def _retarget_active_plan(self, reason: str) -> bool:
        target_grid = self._get_motion_target_grid()
        suppressed = self._suppress_planner_target(target_grid)
        self.last_target_retarget_reason = str(reason)
        self.planner_targets = []
        self._clear_active_plan()
        self._start_new_grid_plan()
        return bool(suppressed or self.active_plan_grid is not None)

    def _grid_has_unobserved_band(self, gx: int, gy: int) -> bool:
        return not bool(np.all(self.sampled_mask[int(gx), int(gy), :]))

    def _build_local_candidate_mask(
        self,
        center_pos: Optional[np.ndarray] = None,
        radius: Optional[int] = None,
        planner_mode: Optional[str] = None,
    ) -> np.ndarray:
        active_mode = (
            self._get_active_planner_mode()
            if planner_mode is None
            else str(planner_mode).strip().lower()
        )
        if active_mode == "global":
            return np.asarray(self.sampling_valid_mask, dtype=bool).copy()
        if center_pos is None:
            center_pos = self.uav_pos
        center = np.asarray(center_pos, dtype=float).reshape(2)
        center = np.rint(np.clip(center, [0.0, 0.0], [self.Nx - 1, self.Ny - 1])).astype(int)
        radius = self.local_planner_radius if radius is None else max(0, int(radius))
        manhattan = np.abs(self.grid_x_coords - float(center[0])) + np.abs(
            self.grid_y_coords - float(center[1])
        )
        return np.asarray(self.sampling_valid_mask & (manhattan <= float(radius)), dtype=bool)

    def _start_new_grid_plan(self) -> None:
        """Start a new plan from planner output: one target grid + one center frequency."""
        if not self.planner_initialized or self.active_plan_grid is not None:
            return

        refresh_attempted = False
        while True:
            if not self.planner_targets:
                if refresh_attempted:
                    break
                self._refresh_planner_outputs(seed_offset=self.current_step)
                refresh_attempted = True
                continue

            selected = None
            for cand in self.planner_targets:
                if (
                    self._grid_has_unobserved_band(cand.gx, cand.gy)
                    and not self._planner_target_is_suppressed(cand.gx, cand.gy)
                ):
                    selected = cand
                    break

            if selected is not None:
                self.active_plan_grid = (int(selected.gx), int(selected.gy))
                self.active_plan_center_freq = self._select_center_freq_for_grid(
                    int(selected.gx),
                    int(selected.gy),
                    preferred_center_freq=int(selected.center_freq),
                )
                self.active_plan_start_step = int(self.current_step)
                self.active_plan_no_sample_steps = 0
                self._begin_active_target_nmse_record()
                return

            if refresh_attempted:
                break
            self.planner_targets = []
            refresh_attempted = True

    def _clear_active_plan(self) -> None:
        """Clear active plan after update or exhaustion."""
        self._reset_grid_plan_state()

    def _planner_sample_count(self) -> int:
        if hasattr(self.td, "get_num_samples"):
            return int(self.td.get_num_samples())
        return int(len(self.delivered_samples))

    @staticmethod
    def _empty_fusion_meta() -> Dict[str, object]:
        return {
            "raw_count": 0,
            "fused_count": 0,
            "compression_ratio": 1.0,
            "raw_per_fused": np.empty((0,), dtype=int),
        }

    def _samples_to_observation_arrays(
        self,
        samples: List[SpectrumSample],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not samples:
            empty_locs = np.empty((0, 2), dtype=float)
            empty_gamma = np.empty((0, self.K), dtype=float)
            empty_omega = np.empty((0, self.K), dtype=np.int32)
            return empty_locs, empty_gamma, empty_omega

        obs_locs = np.asarray([np.asarray(sample.position, dtype=float) for sample in samples], dtype=float)
        gamma = np.asarray([np.asarray(sample.gamma, dtype=float) for sample in samples], dtype=float)
        omega = np.asarray([np.asarray(sample.omega, dtype=np.int32) for sample in samples], dtype=np.int32)
        return obs_locs, gamma, omega

    def _compute_fusion_meta_for_samples(
        self,
        samples: List[SpectrumSample],
    ) -> Dict[str, object]:
        if not samples:
            return self._empty_fusion_meta()

        obs_locs, gamma, omega = self._samples_to_observation_arrays(samples)
        _, _, _, fusion_meta = fuse_observations_by_grid(
            obs_locs,
            gamma,
            omega,
            self.Nx,
            self.Ny,
        )
        return dict(fusion_meta)

    def _refresh_pending_ensemble_counts(self) -> Dict[str, object]:
        raw_count = int(len(self.pending_ensemble_samples))
        if raw_count <= 0:
            self.pending_ensemble_sample_count = 0
            self.pending_ensemble_effective_sample_count = 0
            return self._empty_fusion_meta()

        fusion_meta = self._compute_fusion_meta_for_samples(self.pending_ensemble_samples)
        self.pending_ensemble_sample_count = raw_count
        self.pending_ensemble_effective_sample_count = int(fusion_meta.get("fused_count", 0))
        return fusion_meta

    def _stage_pending_ensemble_samples(
        self,
        samples: List[SpectrumSample],
    ) -> Dict[str, object]:
        if not samples:
            return self._empty_fusion_meta()

        self.pending_ensemble_samples.extend(list(samples))
        return self._refresh_pending_ensemble_counts()

    def _flush_pending_ensemble_samples(self) -> Dict[str, object]:
        if not self.pending_ensemble_samples:
            self.pending_ensemble_sample_count = 0
            self.pending_ensemble_effective_sample_count = 0
            return self._empty_fusion_meta()

        staged_samples = list(self.pending_ensemble_samples)
        fusion_meta = self._refresh_pending_ensemble_counts()
        self.delivered_samples.extend(staged_samples)
        self.td.add_samples(staged_samples)
        self.pending_ensemble_samples = []
        self.pending_ensemble_sample_count = 0
        self.pending_ensemble_effective_sample_count = 0
        return fusion_meta

    def _planner_effective_sample_count(self) -> int:
        if hasattr(self.td, "get_effective_sample_count"):
            return int(self.td.get_effective_sample_count())
        return int(self._planner_sample_count())

    def _planner_ready(self) -> bool:
        return self._planner_effective_sample_count() >= int(self.config.planner.min_samples_for_ensemble)

    def _begin_active_target_nmse_record(self) -> None:
        if self.active_plan_grid is None:
            self.active_target_nmse_record = None
            return
        self.active_target_nmse_record = {
            "target_grid_x": int(self.active_plan_grid[0]),
            "target_grid_y": int(self.active_plan_grid[1]),
            "target_center_freq": int(self.active_plan_center_freq) if self.active_plan_center_freq is not None else -1,
            "start_step": int(self.current_step),
            "start_uav_pos": [float(self.uav_pos[0]), float(self.uav_pos[1])],
            "start_nmse": float(self.radio_map_state.nmse),
            "reconstruction_steps": [],
            "reconstruction_nmse": [],
            "reconstruction_nmse_delta": [],
        }

    def _record_active_target_nmse_update(self) -> None:
        if self.active_target_nmse_record is None:
            return
        prev_nmse = float(self.active_target_nmse_record["start_nmse"])
        if self.active_target_nmse_record["reconstruction_nmse"]:
            prev_nmse = float(self.active_target_nmse_record["reconstruction_nmse"][-1])
        curr_nmse = float(self.radio_map_state.nmse)
        self.active_target_nmse_record["reconstruction_steps"].append(int(self.current_step))
        self.active_target_nmse_record["reconstruction_nmse"].append(curr_nmse)
        self.active_target_nmse_record["reconstruction_nmse_delta"].append(curr_nmse - prev_nmse)

    def _finalize_active_target_nmse_record(self, target_reached: bool) -> Optional[Dict[str, object]]:
        if self.active_target_nmse_record is None:
            return None

        record = {
            "target_grid_x": int(self.active_target_nmse_record["target_grid_x"]),
            "target_grid_y": int(self.active_target_nmse_record["target_grid_y"]),
            "target_center_freq": int(self.active_target_nmse_record["target_center_freq"]),
            "start_step": int(self.active_target_nmse_record["start_step"]),
            "end_step": int(self.current_step),
            "start_uav_pos": list(self.active_target_nmse_record["start_uav_pos"]),
            "end_uav_pos": [float(self.uav_pos[0]), float(self.uav_pos[1])],
            "start_nmse": float(self.active_target_nmse_record["start_nmse"]),
            "reconstruction_steps": [int(step) for step in self.active_target_nmse_record["reconstruction_steps"]],
            "reconstruction_nmse": [float(value) for value in self.active_target_nmse_record["reconstruction_nmse"]],
            "reconstruction_nmse_delta": [float(value) for value in self.active_target_nmse_record["reconstruction_nmse_delta"]],
            "target_reached": int(target_reached),
        }
        last_nmse = record["start_nmse"]
        if record["reconstruction_nmse"]:
            last_nmse = float(record["reconstruction_nmse"][-1])
        record["nmse_change_total"] = float(last_nmse - record["start_nmse"])
        self.completed_target_nmse_records.append(record)
        self.last_completed_target_nmse_record = record
        self.active_target_nmse_record = None
        return record

    def _record_nmse_event(
        self,
        event_type: str,
        reason: str,
        trigger_sample_count: int,
        extra_fields: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if event_type == "reconstruction":
            history = self.reconstruction_events
        elif event_type == "ensemble":
            history = self.ensemble_events
        else:
            raise ValueError(f"Unsupported event_type: {event_type}")

        prev_nmse = float(self.episode_nmse_start)
        if history:
            prev_nmse = float(history[-1]["nmse"])

        target = self._get_current_observation_target()
        curr_nmse = float(self.radio_map_state.nmse)
        event = {
            "type": str(event_type),
            "step": int(self.current_step),
            "reason": str(reason),
            "nmse": curr_nmse,
            "nmse_delta": float(curr_nmse - prev_nmse),
            "trigger_sample_count": int(trigger_sample_count),
            "planner_sample_count": int(self._planner_sample_count()),
            "planner_effective_sample_count": int(self._planner_effective_sample_count()),
            "map_update_count": int(self.map_update_count),
            "target_grid_x": int(target.gx) if target is not None else -1,
            "target_grid_y": int(target.gy) if target is not None else -1,
            "target_center_freq": int(target.center_freq) if target is not None else -1,
        }
        if extra_fields:
            event.update(dict(extra_fields))
        history.append(event)
        return event

    def _record_ensemble_event(
        self,
        reason: str,
        trigger_sample_count: int,
        extra_fields: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        event = self._record_nmse_event(
            event_type="ensemble",
            reason=reason,
            trigger_sample_count=trigger_sample_count,
            extra_fields=extra_fields,
        )
        return event

    def _sync_cached_ensemble_state(self) -> None:
        sample_count = int(self._planner_sample_count())
        cached_outputs = None
        if hasattr(self.td, "get_latest_ensemble_outputs"):
            cached_outputs = self.td.get_latest_ensemble_outputs(
                expected_sample_count=sample_count,
            )
        if cached_outputs is None:
            mean_map = np.asarray(self.radio_map_state.spectrum_map, dtype=float)
            var_map = np.zeros_like(mean_map)
        else:
            mean_map, var_map = cached_outputs
            mean_map = np.asarray(mean_map, dtype=float)
            var_map = np.asarray(var_map, dtype=float)
        self.latest_mean_map = mean_map.copy()
        self.latest_var_map = np.maximum(var_map, 0.0)
        self.uncertainty = UncertaintyMap(
            spatial_uncertainty=np.mean(self.latest_var_map, axis=2),
            frequency_uncertainty=np.mean(self.latest_var_map, axis=(0, 1)),
            joint_uncertainty=self.latest_var_map.copy(),
        )

    def _run_ensemble_map_update(
        self,
        reason: str,
        seed_offset: int,
        refresh_targets: bool = True,
    ) -> Optional[Dict[str, object]]:
        if self.pending_ensemble_sample_count <= 0 and self._planner_sample_count() <= 0:
            return None

        trigger_sample_count = int(self.pending_ensemble_sample_count)
        trigger_effective_sample_count = int(self.pending_ensemble_effective_sample_count)
        trigger_fusion_meta = (
            self._flush_pending_ensemble_samples()
            if trigger_sample_count > 0
            else self._empty_fusion_meta()
        )
        try:
            self.radio_map_state = self.td.reconstruct()
        except Exception as exc:
            raise RuntimeError(
                f"DU-IIBTD ensemble map update failed during {reason!r}."
            ) from exc
        self.map_update_count += 1
        self._record_active_target_nmse_update()
        ensemble_diag = {}
        if hasattr(self.td, "get_latest_ensemble_diagnostics"):
            ensemble_diag = self.td.get_latest_ensemble_diagnostics()

        if (
            refresh_targets
            and self._planner_ready()
            and not self._should_preserve_global_target()
        ):
            ensemble_event = self._refresh_planner_outputs(
                seed_offset=seed_offset,
                force_ensemble=True,
                trigger_reason=reason,
                trigger_sample_count=trigger_sample_count,
                trigger_effective_sample_count=trigger_effective_sample_count,
                trigger_fusion_meta=trigger_fusion_meta,
            )
            if ensemble_event is not None:
                return ensemble_event

        self._sync_cached_ensemble_state()
        return self._record_ensemble_event(
            reason=reason,
            trigger_sample_count=trigger_sample_count,
            extra_fields={
                "trigger_raw_sample_count": int(trigger_sample_count),
                "trigger_effective_sample_count": int(trigger_effective_sample_count),
                "trigger_fusion_raw_count": int(trigger_fusion_meta.get("raw_count", trigger_sample_count)),
                "trigger_fusion_fused_count": int(
                    trigger_fusion_meta.get("fused_count", trigger_effective_sample_count)
                ),
                "trigger_fusion_compression_ratio": float(
                    trigger_fusion_meta.get("compression_ratio", 1.0)
                ),
                "recon_mode": str(ensemble_diag.get("recon_mode", "")),
                "full_refresh_due": bool(ensemble_diag.get("full_refresh_due", False)),
                "nmse_refresh_triggered": bool(
                    ensemble_diag.get("nmse_refresh_triggered", False)
                ),
                "nmse_refresh_delta": float(ensemble_diag.get("nmse_refresh_delta", 0.0)),
                "nmse_refresh_reference_before": float(
                    ensemble_diag.get("nmse_refresh_reference_before", float("nan"))
                ),
                "nmse_refresh_reference_after": float(
                    ensemble_diag.get("nmse_refresh_reference_after", float("nan"))
                ),
                "nmse_degradation": float(
                    ensemble_diag.get("nmse_degradation", float("nan"))
                ),
                "pre_refresh_nmse": float(
                    ensemble_diag.get("pre_refresh_nmse", float("nan"))
                ),
                "post_refresh_nmse": float(
                    ensemble_diag.get("post_refresh_nmse", float("nan"))
                ),
            },
        )

    def step(
        self,
        uav_action: int,
        ugv_action: int,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, dict]:
        self.current_step += 1
        planner_submode_before_step = str(self._get_active_planner_mode())

        prev_uav_pos = self.uav_pos.copy()
        prev_ugv_pos = self.ugv_pos.copy()

        # 1) Planner keeps one active grid target; UAV controls movement + bandwidth.
        self._start_new_grid_plan()
        step_target = self._get_current_observation_target()
        step_target_source = self._get_current_target_source()
        step_target_grid = (
            (int(step_target.gx), int(step_target.gy))
            if step_target is not None
            else None
        )
        step_target_center_freq = int(step_target.center_freq) if step_target is not None else -1
        reward_target = (
            np.array([float(step_target.gx), float(step_target.gy)], dtype=float)
            if step_target is not None
            else None
        )
        if reward_target is not None:
            prev_uav_target_dist = manhattan_distance(prev_uav_pos, reward_target)
        else:
            prev_uav_target_dist = None

        # 设置UAV的观测带宽和移动 
        self._apply_uav_joint_action(int(uav_action))

        # 2) Movement execution.
        self._move_ugv(int(ugv_action))
        uav_move_dist = self._grid_distance_to_meters(
            float(np.linalg.norm(self.uav_pos - prev_uav_pos))
        )
        ugv_move_dist = self._grid_distance_to_meters(
            float(np.linalg.norm(self.ugv_pos - prev_ugv_pos))
        )
        curr_uav_target_dist = None
        uav_progress_raw = 0.0
        uav_progress_ratio = 0.0
        if reward_target is not None and prev_uav_target_dist is not None:
            curr_uav_target_dist = manhattan_distance(self.uav_pos, reward_target)
            uav_progress_raw = float(prev_uav_target_dist - curr_uav_target_dist)
            uav_progress_ratio = relative_distance_improvement(
                prev_uav_target_dist,
                curr_uav_target_dist,
                floor=self.uav_step_count,
            )

        prev_ugv_uav_dist = manhattan_distance(prev_ugv_pos, self.uav_pos)
        curr_ugv_uav_dist = manhattan_distance(self.ugv_pos, self.uav_pos)
        ugv_progress_to_uav_raw = float(prev_ugv_uav_dist - curr_ugv_uav_dist)
        ugv_progress_to_uav_ratio = relative_distance_improvement(
            prev_ugv_uav_dist,
            curr_ugv_uav_dist,
            floor=self.ugv_step_count,
        )

        if reward_target is not None:
            prev_ugv_target_dist = self._occupancy_shortest_path_distance(
                prev_ugv_pos,
                reward_target,
            )
            curr_ugv_target_dist = self._occupancy_shortest_path_distance(
                self.ugv_pos,
                reward_target,
            )
            if np.isfinite(prev_ugv_target_dist) and np.isfinite(curr_ugv_target_dist):
                ugv_progress_to_target_raw = float(prev_ugv_target_dist - curr_ugv_target_dist)
            else:
                ugv_progress_to_target_raw = 0.0
            ugv_progress_to_target_ratio = relative_distance_improvement(
                prev_ugv_target_dist,
                curr_ugv_target_dist,
                floor=self.ugv_step_count,
            )
        else:
            prev_ugv_target_dist = None
            curr_ugv_target_dist = None
            ugv_progress_to_target_raw = 0.0
            ugv_progress_to_target_ratio = 0.0

        active_ugv_uav_weight = float(self.ugv_progress_uav_weight)
        active_ugv_target_weight = float(self.ugv_progress_target_weight)
        if not (
            reward_target is not None
            and prev_ugv_target_dist is not None
            and curr_ugv_target_dist is not None
            and np.isfinite(prev_ugv_target_dist)
            and np.isfinite(curr_ugv_target_dist)
        ):
            active_ugv_uav_weight = 1.0
            active_ugv_target_weight = 0.0

        ugv_prev_mixed_dist = (
            active_ugv_uav_weight * prev_ugv_uav_dist
            + active_ugv_target_weight * (
                float(prev_ugv_target_dist) if prev_ugv_target_dist is not None else 0.0
            )
        )
        ugv_curr_mixed_dist = (
            active_ugv_uav_weight * curr_ugv_uav_dist
            + active_ugv_target_weight * (
                float(curr_ugv_target_dist) if curr_ugv_target_dist is not None else 0.0
            )
        )
        ugv_progress_raw = float(ugv_prev_mixed_dist - ugv_curr_mixed_dist)
        ugv_progress_ratio = relative_distance_improvement(
            ugv_prev_mixed_dist,
            ugv_curr_mixed_dist,
            floor=self.ugv_step_count,
        )

        # 3) Channel update.
        self.ugv_channel_info = self._get_channel_info()

        # 4) UAV samples the current grid point directly from GT with noise
        # only when the current cell is not a building.
        sample, newly_sampled_freqs, newly_visited_spatial, sampling_stats = self._collect_current_grid_sample()
        data_produced_bits = 0.0
        if sample is not None:
            self._enqueue_sample_packet(
                sample,
                novelty_ratio=float(sampling_stats.get("novelty_ratio", 1.0)),
            )
            data_size = float(self.sensing_band_num) * self.config.comm.data_per_sample
            data_produced_bits = data_size

        # 5) Transmission queue simulation.
        queue_bits_before_tx = self._queue_remaining_bits()
        delivered_packets, data_delivered_bits, novel_data_delivered_bits = self._simulate_transmission()
        processed_samples = self._process_delivered_samples(delivered_packets)
        dropped_packets = self._enforce_queue_capacity()
        queue_bits_after_tx = self._queue_remaining_bits()
        self._update_active_plan_no_sample_counter(sample)

        map_updated = False
        ensemble_event: Optional[Dict[str, object]] = None
        bootstrap_target_reached_event: Optional[Dict[str, object]] = None
        bootstrap_retarget_event: Optional[Dict[str, object]] = None
        bootstrap_handoff_event: Optional[Dict[str, object]] = None
        target_exact_reached = bool(
            step_target_grid is not None
            and int(np.rint(self.uav_pos[0])) == int(step_target_grid[0])
            and int(np.rint(self.uav_pos[1])) == int(step_target_grid[1])
        )
        target_arrival_distance = self._target_arrival_distance(step_target_grid)
        target_proximity_reached = bool(
            step_target_grid is not None
            and sample is not None
            and (not target_exact_reached)
            and self.target_arrival_radius > 0.0
            and target_arrival_distance <= self.target_arrival_radius + 1e-9
        )
        target_reached = bool(target_exact_reached or target_proximity_reached)
        target_retargeted = False
        target_retarget_reason = ""
        bootstrap_target_reached = bool(step_target_source == "bootstrap" and target_reached)
        if bootstrap_target_reached and not self.bootstrap_target_reached_once:
            self.bootstrap_target_reached_once = True
            bootstrap_target_reached_event = self._record_bootstrap_event(
                event="bootstrap_target_reached",
                reason="target_reached",
                target=self.bootstrap_target,
            )
        completed_target_nmse_record: Optional[Dict[str, object]] = None

        # During planner warmup, commit buffered observations only at ensemble intervals.
        if not self.planner_initialized:
            if self.pending_ensemble_sample_count >= self.ensemble_refresh_interval:
                ensemble_event = self._run_ensemble_map_update(
                    reason="warmup_interval",
                    seed_offset=self.current_step,
                    refresh_targets=True,
                )
                map_updated = ensemble_event is not None

            if self._planner_ready():
                if not self.planner_targets:
                    if self.pending_ensemble_sample_count > 0 or (
                        self._planner_sample_count() > 0 and self.radio_map_state.last_update_step <= 0
                    ):
                        ensemble_event = self._run_ensemble_map_update(
                            reason="planner_warmup",
                            seed_offset=self.current_step,
                            refresh_targets=True,
                        )
                        map_updated = ensemble_event is not None
                    else:
                        ensemble_event = self._refresh_planner_outputs(
                            seed_offset=self.current_step,
                            force_ensemble=True,
                            trigger_reason="planner_warmup",
                        )
                self.planner_initialized = bool(self.planner_targets)
                if self.planner_initialized:
                    bootstrap_handoff_event = self._complete_bootstrap_phase(reason="planner_warmup")
                self._clear_active_plan()
                self._start_new_grid_plan()
            elif bootstrap_target_reached and self._bootstrap_target_is_active():
                bootstrap_retarget_event = self._retarget_bootstrap_phase(reason="target_reached")
            elif self._bootstrap_target_timed_out():
                bootstrap_retarget_event = self._retarget_bootstrap_phase(reason="step_timeout")
        elif self.planner_initialized:
            preserve_global_target = self._should_preserve_global_target()
            if target_reached:
                if self.pending_ensemble_sample_count >= self.ensemble_refresh_interval:
                    ensemble_event = self._run_ensemble_map_update(
                        reason="target_reached",
                        seed_offset=self.current_step,
                        refresh_targets=not preserve_global_target,
                    )
                    map_updated = ensemble_event is not None
                completed_target_nmse_record = self._finalize_active_target_nmse_record(
                    target_reached=True
                )
                if preserve_global_target:
                    target_retargeted = self._retarget_active_plan(
                        reason="global_target_reached",
                    )
                    target_retarget_reason = "global_target_reached"
                else:
                    if map_updated:
                        self._clear_active_plan()
                        self._start_new_grid_plan()
                    else:
                        target_retargeted = self._retarget_active_plan(
                            reason="target_reached_deferred_ensemble",
                        )
                        target_retarget_reason = "target_reached_deferred_ensemble"
            else:
                if self.pending_ensemble_sample_count >= self.ensemble_refresh_interval:
                    ensemble_event = self._run_ensemble_map_update(
                        reason="ensemble_interval",
                        seed_offset=self.current_step,
                        refresh_targets=not preserve_global_target,
                    )
                    map_updated = ensemble_event is not None
                    if ensemble_event is not None and not preserve_global_target:
                        self._clear_active_plan()
                        self._start_new_grid_plan()
                elif self._active_plan_is_stuck_without_samples():
                    completed_target_nmse_record = self._finalize_active_target_nmse_record(
                        target_reached=False
                    )
                    target_retargeted = self._retarget_active_plan(reason="stuck_no_sample")
                    target_retarget_reason = "stuck_no_sample"

        should_force_flush = (
            bool(self.config.planner.flush_reconstruction_on_episode_end)
            and (not map_updated)
            and self.pending_ensemble_sample_count > 0
            and (
                self.uav_energy <= 0
                or self.current_step >= self.config.mappo.episode_max_steps
            )
        )
        if should_force_flush:
            ensemble_event = self._run_ensemble_map_update(
                reason="terminal_flush",
                seed_offset=self.current_step,
                refresh_targets=False,
            )
            map_updated = ensemble_event is not None
        if self.planner_initialized and self.active_plan_grid is None and self.planner_targets:
            self._start_new_grid_plan()
        self._update_hybrid_planner_submode(
            planner_submode_before_step=planner_submode_before_step,
            map_updated=map_updated,
        )
        planner_submode_after_step = str(self._get_active_planner_mode())
        planner_mode_switch = (
            f"{planner_submode_before_step}->{planner_submode_after_step}"
            if planner_submode_after_step != planner_submode_before_step
            else ""
        )
        planner_switched_to_global = int(planner_mode_switch == "local->global")

        # 7) Rewards.
        rewards, reward_info = self._compute_reward(
            uav_move_dist=uav_move_dist,
            ugv_move_dist=ugv_move_dist,
            uav_progress_ratio=uav_progress_ratio,
            ugv_progress_ratio=ugv_progress_ratio,
            ugv_progress_to_uav_ratio=ugv_progress_to_uav_ratio,
            ugv_progress_to_target_ratio=ugv_progress_to_target_ratio,
            target_source=step_target_source,
            map_updated=map_updated,
            reward_target_grid=step_target_grid,
            target_reached=target_reached,
            queue_bits_before_tx=queue_bits_before_tx,
            queue_bits_after_tx=queue_bits_after_tx,
            data_produced_bits=data_produced_bits,
            data_delivered_bits=data_delivered_bits,
            novel_data_delivered_bits=novel_data_delivered_bits,
            dropped_packets=dropped_packets,
            newly_sampled_freqs=newly_sampled_freqs,
            newly_visited_spatial=newly_visited_spatial,
            sampling_stats=sampling_stats,
        )
        reward_info.update(
            {
                "goal_normalization_den": 1.0,
                "uav_target_dist_prev": (
                    float(prev_uav_target_dist) if prev_uav_target_dist is not None else -1.0
                ),
                "uav_target_dist_curr": (
                    float(curr_uav_target_dist) if curr_uav_target_dist is not None else -1.0
                ),
                "ugv_target_dist_prev": (
                    float(prev_ugv_target_dist)
                    if prev_ugv_target_dist is not None and np.isfinite(prev_ugv_target_dist)
                    else -1.0
                ),
                "ugv_target_dist_curr": (
                    float(curr_ugv_target_dist)
                    if curr_ugv_target_dist is not None and np.isfinite(curr_ugv_target_dist)
                    else -1.0
                ),
                "ugv_uav_dist_prev": float(prev_ugv_uav_dist),
                "ugv_uav_dist_curr": float(curr_ugv_uav_dist),
                "ugv_mixed_dist_prev": float(ugv_prev_mixed_dist),
                "ugv_mixed_dist_curr": float(ugv_curr_mixed_dist),
                "uav_progress": float(uav_progress_raw),
                "uav_progress_ratio": float(uav_progress_ratio),
                "uav_progress_norm": float(uav_progress_ratio),
                "ugv_progress": float(ugv_progress_raw),
                "ugv_progress_ratio": float(ugv_progress_ratio),
                "ugv_progress_norm": float(ugv_progress_ratio),
                "ugv_progress_to_uav": float(ugv_progress_to_uav_raw),
                "ugv_progress_to_uav_ratio": float(ugv_progress_to_uav_ratio),
                "ugv_progress_to_uav_norm": float(ugv_progress_to_uav_ratio),
                "ugv_progress_to_target": float(ugv_progress_to_target_raw),
                "ugv_progress_to_target_ratio": float(ugv_progress_to_target_ratio),
                "ugv_progress_to_target_norm": float(ugv_progress_to_target_ratio),
                "ugv_progress_uav_weight": float(active_ugv_uav_weight),
                "ugv_progress_target_weight": float(active_ugv_target_weight),
                "ugv_progress_uav_weight_config": float(self.ugv_progress_uav_weight),
                "ugv_progress_target_weight_config": float(self.ugv_progress_target_weight),
                "progress_metric": (
                    "relative_improvement__delta_over_prev_distance__"
                    "uav_to_goal__ugv_to_weighted(grid_uav,target_shortest_path)"
                ),
            }
        )

        # 8) Termination.
        rc = self.config.reward
        target_nmse_reached = bool(
            self.radio_map_state.last_update_step > 0
            and self.radio_map_state.nmse <= rc.accuracy_target_nmse
        )
        energy_depleted = bool(self.uav_energy <= 0)
        reached_step_limit = bool(self.current_step >= self.config.mappo.episode_max_steps)

        # Evaluation uses a fixed horizon. NMSE is diagnostic only; the sole
        # failure mode is exhausting UAV energy before the horizon.
        terminated = bool(energy_depleted)
        truncated = bool((not terminated) and reached_step_limit)
        timed_out = bool(truncated)
        terminal_failure = bool(energy_depleted)

        terminal_reward = (
            -abs(float(rc.terminal_failure_penalty)) if terminal_failure else 0.0
        )
        if abs(terminal_reward) > 0.0:
            rewards["team_reward"] = float(rewards["team_reward"] + terminal_reward)
            rewards["uav_reward"] = float(rewards["uav_reward"] + terminal_reward)
            rewards["ugv_reward"] = float(rewards["ugv_reward"] + terminal_reward)
        reward_info["r_terminal"] = float(terminal_reward)
        reward_info["target_nmse_reached"] = int(target_nmse_reached)
        reward_info["energy_depleted"] = int(energy_depleted)
        reward_info["terminal_failure"] = int(terminal_failure)
        reward_info["timed_out"] = int(timed_out)
        reward_info["team_reward"] = float(rewards["team_reward"])
        reward_info["uav_reward"] = float(rewards["uav_reward"])
        reward_info["ugv_reward"] = float(rewards["ugv_reward"])
        bootstrap_event_labels: List[str] = []
        bootstrap_event_reasons: List[str] = []
        if bootstrap_target_reached_event is not None:
            bootstrap_event_labels.append(str(bootstrap_target_reached_event["event"]))
            bootstrap_event_reasons.append(str(bootstrap_target_reached_event["reason"]))
        if bootstrap_retarget_event is not None:
            bootstrap_event_labels.append(str(bootstrap_retarget_event["event"]))
            bootstrap_event_reasons.append(str(bootstrap_retarget_event["reason"]))
        if bootstrap_handoff_event is not None:
            bootstrap_event_labels.append(str(bootstrap_handoff_event["event"]))
            bootstrap_event_reasons.append(str(bootstrap_handoff_event["reason"]))

        # 9) Observation.
        obs = self._build_observations()
        info_current_target = self._get_current_observation_target()
        info_target_grid = (
            (int(info_current_target.gx), int(info_current_target.gy))
            if info_current_target is not None
            else None
        )
        info_target_center_freq = (
            int(info_current_target.center_freq) if info_current_target is not None else -1
        )
        info_target_source = str(self._get_current_target_source())
        info = {
            **reward_info,
            "nmse": self.radio_map_state.nmse,
            "channel_capacity": self.ugv_channel_info.capacity_bps,
            "channel_los": self.ugv_channel_info.los,
            "snr_db": self.ugv_channel_info.snr_db,
            "bw_ratio": self.current_bw_ratio,
            "sample_center_freq": int(self.last_sample_center_freq),
            "uav_energy": self.uav_energy,
            "queue_size": len(self.uav_data_queue),
            "queue_capacity_packets": int(self.max_q),
            "total_samples": self.total_collected_samples,
            "step": self.current_step,
            "target_grid_x": int(info_target_grid[0]) if info_target_grid is not None else -1,
            "target_grid_y": int(info_target_grid[1]) if info_target_grid is not None else -1,
            "target_center_freq": int(info_target_center_freq),
            "target_source": info_target_source,
            "executed_target_grid_x": (
                int(step_target_grid[0]) if step_target_grid is not None else -1
            ),
            "executed_target_grid_y": (
                int(step_target_grid[1]) if step_target_grid is not None else -1
            ),
            "executed_target_center_freq": int(step_target_center_freq),
            "executed_target_source": str(step_target_source),
            "target_count": len(self.planner_targets),
            "map_updated": int(map_updated),
            "planner_initialized": int(self.planner_initialized),
            "planner_sample_count": int(self._planner_sample_count()),
            "planner_effective_sample_count": int(self._planner_effective_sample_count()),
            "planner_submode": planner_submode_after_step,
            "planner_mode_switch": planner_mode_switch,
            "planner_switched_to_global": planner_switched_to_global,
            "planner_stall_count": int(self.planner_stall_count),
            "planner_global_steps_remaining": int(self.global_steps_remaining),
            "initial_observation_mode": str(self.initial_observation_mode),
            "target_reached": int(target_reached),
            "target_exact_reached": int(target_exact_reached),
            "target_proximity_reached": int(target_proximity_reached),
            "target_arrival_distance": float(target_arrival_distance),
            "target_arrival_radius": float(self.target_arrival_radius),
            "target_retargeted": int(target_retargeted),
            "target_retarget_reason": str(target_retarget_reason),
            "active_target_no_sample_steps": int(self.active_plan_no_sample_steps),
            "suppressed_target_count": int(len(self.suppressed_planner_targets)),
            "bootstrap_active": int(info_target_source == "bootstrap"),
            "bootstrap_target_reached": int(bootstrap_target_reached),
            "bootstrap_handoff": int(bootstrap_handoff_event is not None),
            "bootstrap_event": "|".join(bootstrap_event_labels),
            "bootstrap_event_reason": "|".join(bootstrap_event_reasons),
            "prefill_percent": float(self.prefill_percent),
            "prefill_budget_basis": int(self.prefill_budget_basis_count),
            "prefill_sample_count": int(self.prefill_sample_count),
            "prefill_applied": int(self.prefill_applied),
            "ensemble_triggered": int(ensemble_event is not None),
            "ensemble_reason": (
                str(ensemble_event["reason"])
                if ensemble_event is not None
                else ""
            ),
            "ensemble_event_nmse": (
                float(ensemble_event["nmse"])
                if ensemble_event is not None
                else float(self.radio_map_state.nmse)
            ),
            "ensemble_event_nmse_delta": (
                float(ensemble_event["nmse_delta"])
                if ensemble_event is not None
                else 0.0
            ),
            "ensemble_target_grid_x": (
                int(ensemble_event.get("target_grid_x", -1))
                if ensemble_event is not None
                else -1
            ),
            "ensemble_target_grid_y": (
                int(ensemble_event.get("target_grid_y", -1))
                if ensemble_event is not None
                else -1
            ),
            "ensemble_target_center_freq": (
                int(ensemble_event.get("target_center_freq", -1))
                if ensemble_event is not None
                else -1
            ),
            "ensemble_recon_mode": (
                str(ensemble_event.get("recon_mode", ""))
                if ensemble_event is not None
                else ""
            ),
            "ensemble_full_refresh_due": (
                int(bool(ensemble_event.get("full_refresh_due", False)))
                if ensemble_event is not None
                else 0
            ),
            "ensemble_nmse_refresh_triggered": int(
                bool(ensemble_event.get("nmse_refresh_triggered", False))
                if ensemble_event is not None
                else False
            ),
            "ensemble_nmse_refresh_delta": (
                float(ensemble_event.get("nmse_refresh_delta", 0.0))
                if ensemble_event is not None
                else 0.0
            ),
            "ensemble_nmse_refresh_reference_before": (
                float(ensemble_event.get("nmse_refresh_reference_before", float("nan")))
                if ensemble_event is not None
                else float("nan")
            ),
            "ensemble_nmse_refresh_reference_after": (
                float(ensemble_event.get("nmse_refresh_reference_after", float("nan")))
                if ensemble_event is not None
                else float("nan")
            ),
            "ensemble_nmse_degradation": (
                float(ensemble_event.get("nmse_degradation", float("nan")))
                if ensemble_event is not None
                else float("nan")
            ),
            "ensemble_trigger_sample_count": (
                int(ensemble_event.get("trigger_raw_sample_count", ensemble_event.get("trigger_sample_count", 0)))
                if ensemble_event is not None
                else 0
            ),
            "ensemble_trigger_effective_sample_count": (
                int(ensemble_event.get("trigger_effective_sample_count", 0))
                if ensemble_event is not None
                else 0
            ),
            "ensemble_trigger_fusion_compression_ratio": (
                float(ensemble_event.get("trigger_fusion_compression_ratio", 1.0))
                if ensemble_event is not None
                else 1.0
            ),
            "pending_ensemble_sample_count": int(self.pending_ensemble_sample_count),
            "pending_ensemble_effective_sample_count": int(
                self.pending_ensemble_effective_sample_count
            ),
            "sensing_band_num": int(self.sensing_band_num),
            "sensing_bw_units": int(self.current_sensing_units),
            "comm_bw_units": int(self.current_comm_units),
            "processed_samples": int(processed_samples),
            "target_path_nmse_trace": (
                [float(completed_target_nmse_record["start_nmse"]), *completed_target_nmse_record["reconstruction_nmse"]]
                if completed_target_nmse_record is not None
                else []
            ),
            "target_path_nmse_delta": (
                list(completed_target_nmse_record["reconstruction_nmse_delta"])
                if completed_target_nmse_record is not None
                else []
            ),
            "target_path_reconstruction_steps": (
                list(completed_target_nmse_record["reconstruction_steps"])
                if completed_target_nmse_record is not None
                else []
            ),
            "target_path_nmse_change_total": (
                float(completed_target_nmse_record["nmse_change_total"])
                if completed_target_nmse_record is not None
                else 0.0
            ),
            **self._build_global_switch_top_target_info(top_k=3),
        }
        return obs, rewards, terminated, truncated, self._finalize_step_info(info)

    def _set_bandwidth_info(self, ratio: float) -> None:
        total_units = max(1, int(self.config.uav.total_bw_num))
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if total_units == 1:
            sensing_units = 1
            comm_units = 0
        else:
            sensing_units = int(np.clip(np.ceil(total_units * ratio), 1, total_units - 1))
            comm_units = total_units - sensing_units

        self.current_sensing_units = int(sensing_units)
        self.current_comm_units = int(comm_units)
        # Keep the exposed ratio aligned with the realized discrete allocation.
        self.current_bw_ratio = float(self.current_sensing_units) / float(total_units)
        self.sensing_band_num = int(self.current_sensing_units)

    def _decode_uav_action(self, action: int) -> Tuple[int, int]:
        action = int(np.clip(action, 0, self.uav_action_size - 1))
        direction_choice_idx = action // self.num_bw_choices
        bw_choice_idx = action % self.num_bw_choices
        direction_idx = int(self.uav_direction_ids[direction_choice_idx])
        return direction_idx, bw_choice_idx

    def _apply_uav_joint_action(self, action: int) -> None:
        direction_idx, bw_choice_idx = self._decode_uav_action(action)
        self._set_bandwidth_info(float(self.bandwidth_ratios[bw_choice_idx]))
        self._move_uav(direction_idx=direction_idx)

    def _grid_step_count(self, step_size: float) -> int:
        return max(1, int(round(float(step_size))))

    def _grid_distance_to_meters(self, distance_in_grid: float) -> float:
        return float(distance_in_grid) * float(self.config.scene.grid_spacing)

    def _uav_action_mask_cache_key(self) -> Tuple[int, int, int, int]:
        ux, uy = self._grid_cell(self.uav_pos)
        target_grid = self._get_motion_target_grid()
        if target_grid is None:
            return (ux, uy, -1, -1)
        return (ux, uy, int(target_grid[0]), int(target_grid[1]))

    def _ugv_action_mask_cache_key(self) -> Tuple[int, int]:
        return self._grid_cell(self.ugv_pos)

    def _rollout_direction(
        self,
        position: np.ndarray,
        direction_idx: int,
        step_count: int,
        validator,
        stop_at_target: bool = False,
    ) -> Tuple[np.ndarray, int]:
        offset = DIRECTION_OFFSETS.get(int(direction_idx), np.array([0, 0], dtype=float)).astype(float)
        if np.allclose(offset, 0.0):
            clipped = np.rint(np.clip(np.asarray(position, dtype=float), [0.0, 0.0], [self.Nx - 1, self.Ny - 1]))
            return clipped, 0

        new_pos = np.asarray(position, dtype=float).copy()
        target_grid = self._get_motion_target_grid() if stop_at_target else None
        moved_steps = 0
        for _ in range(step_count):
            proposed = new_pos + offset
            if not validator(proposed):
                break
            new_pos = proposed
            moved_steps += 1
            if (
                target_grid is not None
                and int(np.rint(new_pos[0])) == int(target_grid[0])
                and int(np.rint(new_pos[1])) == int(target_grid[1])
            ):
                break

        clipped = np.rint(np.clip(new_pos, [0.0, 0.0], [self.Nx - 1, self.Ny - 1]))
        return clipped, int(moved_steps)

    def _can_follow_direction(
        self,
        position: np.ndarray,
        direction_idx: int,
        step_count: int,
        validator,
        stop_at_target: bool = False,
    ) -> bool:
        """Return whether a direction would produce any valid movement from position."""
        _, moved_steps = self._rollout_direction(
            position=position,
            direction_idx=direction_idx,
            step_count=step_count,
            validator=validator,
            stop_at_target=stop_at_target,
        )
        return bool(moved_steps > 0 or np.allclose(DIRECTION_OFFSETS.get(int(direction_idx), np.array([0, 0], dtype=float)), 0.0))

    def _move_uav(self, direction_idx: int) -> float:
        old_pos = self.uav_pos.copy()
        self.uav_pos, moved_steps = self._rollout_direction(
            position=old_pos,
            direction_idx=direction_idx,
            step_count=self.uav_step_count,
            validator=self.scene.is_uav_position_valid,
            stop_at_target=True,
        )
        if moved_steps > 0:
            # Flight power is modeled per successful grid hop, not per meter traveled.
            flight_duration = float(moved_steps) * float(self.config.uav.step_duration)
            energy = self.config.uav.flight_power * flight_duration
        else:
            energy = self.config.uav.hover_power * self.config.uav.step_duration
        energy += self.config.uav.sensing_power * self.config.uav.step_duration
        self.uav_energy -= energy
        return float(energy)

    def _move_ugv(self, direction: int) -> None:
        offset = DIRECTION_OFFSETS.get(direction, np.array([0, 0]))
        step_count = self.ugv_step_count
        for _ in range(step_count):
            new_pos = self.ugv_pos + offset
            if self.scene.is_ugv_position_valid(new_pos):
                self.ugv_pos = new_pos
            else:
                break
        self.ugv_pos = np.rint(np.clip(self.ugv_pos, [0, 0], [self.Nx - 1, self.Ny - 1]))

    def _get_channel_info(self) -> ChannelInfo:
        # 考虑有效带宽
        comm_bw = self.current_comm_units * self.config.uav.unit_bandwidth_hz * 0.8
        is_los = self.scene.has_line_of_sight(
            uav_position=self.uav_pos,
            ugv_position=self.ugv_pos,
        )
        info = self.sim_data.get_channel_info(
            uav_position=self.uav_pos,
            ugv_position=self.ugv_pos,
            bandwidth_comm=comm_bw,
            tx_power_dbm=self.config.comm.tx_power_dbm,
            los=is_los,
        )

        adjusted_snr_db = float(info.snr_db)
        adjusted_snr_linear = 10 ** (adjusted_snr_db / 10.0)
        adjusted_capacity = comm_bw * np.log2(1.0 + max(adjusted_snr_linear, 0.0))
        adjusted_gain = float(info.channel_gain)
        return ChannelInfo(
            path_loss_db=float(info.path_loss_db),
            channel_gain=adjusted_gain,
            los=bool(info.los),
            capacity_bps=float(adjusted_capacity),
            snr_db=adjusted_snr_db,
        )

    def _queue_remaining_bits(self) -> float:
        return float(sum(max(p.size_bits - p.transmitted_bits, 0.0) for p in self.uav_data_queue))

    def _drain_completed_packets(self) -> List[DataPacket]:
        """Flush any fully transmitted packets left in the queue by stale state."""
        if not self.uav_data_queue:
            return []

        delivered: List[DataPacket] = []
        pending_packets: List[DataPacket] = []
        for packet in self.uav_data_queue:
            remaining = packet.size_bits - packet.transmitted_bits
            if remaining <= 1e-9:
                packet.transmitted_bits = packet.size_bits
                delivered.append(packet)
            else:
                pending_packets.append(packet)
        self.uav_data_queue = pending_packets
        return delivered

    def _enqueue_sample_packet(self, sample: SpectrumSample, novelty_ratio: float = 1.0) -> None:
        data_size = float(self.sensing_band_num) * self.config.comm.data_per_sample

        self.uav_data_queue.append(
            DataPacket(
                sample=sample,
                size_bits=data_size,
                created_step=int(self.current_step),
                novelty_ratio=float(np.clip(novelty_ratio, 0.0, 1.0)),
            )
        )

    def _enforce_queue_capacity(self) -> int:
        max_q = int(self.max_q)
        curr_q = len(self.uav_data_queue)
        if curr_q <= max_q:
            return 0

        dropped = curr_q - max_q
        drop_order = sorted(
            range(len(self.uav_data_queue)),
            key=lambda idx: int(self.uav_data_queue[idx].created_step),
        )
        drop_set = set(drop_order[:dropped])
        self.uav_data_queue = [
            packet for idx, packet in enumerate(self.uav_data_queue)
            if idx not in drop_set
        ]
        return int(dropped)

    def _select_next_packet_index(self) -> int:
        if not self.uav_data_queue:
            return -1

        return min(
            range(len(self.uav_data_queue)),
            key=lambda idx: (
                int(self.uav_data_queue[idx].created_step),
                -float(self.uav_data_queue[idx].transmitted_bits),
            ),
        )

    def _simulate_transmission(self) -> Tuple[List[DataPacket], float, float]:
        delivered = self._drain_completed_packets()
        if not self.uav_data_queue:
            return delivered, 0.0, 0.0

        available_bits = self.ugv_channel_info.capacity_bps * self.config.uav.step_duration
        transmitted_bits = 0.0
        novel_transmitted_bits = 0.0
        max_iterations = len(self.uav_data_queue) + 1
        iteration_count = 0
        while self.uav_data_queue and available_bits > 0:
            iteration_count += 1
            if iteration_count > max_iterations:
                raise RuntimeError(
                    "Transmission loop exceeded expected iterations without draining the queue."
                )
            best_idx = self._select_next_packet_index()
            if best_idx < 0:
                break
            packet = self.uav_data_queue[best_idx]

            # 检查数据包中还有多少数据没有传输
            remaining = max(packet.size_bits - packet.transmitted_bits, 0.0)
            
            # 如果都传输完成，则将原来的数据包弹出uav_data_queue
            if remaining <= 0:
                packet.transmitted_bits = packet.size_bits
                self.uav_data_queue.pop(best_idx)
                delivered.append(packet)
                continue
            
            # 如果没有传输完成 则查看应该发送多少数据
            sent_bits = min(available_bits, remaining)
            if sent_bits <= 0.0:
                raise RuntimeError(
                    "Transmission loop selected a packet but could not make forward progress."
                )
            packet.transmitted_bits += sent_bits
            available_bits -= sent_bits
            transmitted_bits += sent_bits
            novel_transmitted_bits += sent_bits * float(np.clip(packet.novelty_ratio, 0.0, 1.0))

            # 如果本次信道容量可以完成传输数据
            if packet.is_complete:
                packet.transmitted_bits = packet.size_bits
                self.uav_data_queue.pop(best_idx)
                delivered.append(packet)
        return delivered, float(transmitted_bits), float(novel_transmitted_bits)

    def _process_delivered_samples(self, delivered_packets: List[DataPacket]) -> int:
        if not delivered_packets:
            return 0

        processed = [packet.sample for packet in delivered_packets]
        self._stage_pending_ensemble_samples(processed)
        return int(len(processed))

    def _is_in_designated_region(self, gx: int, gy: int) -> bool:
        target_grid = self._get_motion_target_grid()
        return target_grid is not None and int(gx) == int(target_grid[0]) and int(gy) == int(target_grid[1])

    def _select_center_freq_for_grid(
        self,
        gx: int,
        gy: int,
        preferred_center_freq: Optional[int] = None,
    ) -> int:
        gx = int(gx)
        gy = int(gy)
        freq_score = self.latest_var_map[gx, gy, :] - (
            self.config.planner.beta_f * self.action_visit[gx, gy, :]
        )
        unsampled_bands = ~self.sampled_mask[gx, gy, :]
        if np.any(unsampled_bands):
            masked_score = freq_score.copy()
            masked_score[~unsampled_bands] = -np.inf
            if preferred_center_freq is not None:
                pref = int(np.clip(preferred_center_freq, 0, self.K - 1))
                if unsampled_bands[pref]:
                    masked_score[pref] += 1e-6
            return int(np.argmax(masked_score))

        if preferred_center_freq is not None:
            return int(np.clip(preferred_center_freq, 0, self.K - 1))
        return int(np.argmax(freq_score))

    def _pick_center_freq(self, gx: int, gy: int) -> int:
        if not self.planner_initialized:
            return int(self.rng.randint(0, self.K))
        preferred_center_freq = None
        if self._is_in_designated_region(gx, gy) and self.active_plan_center_freq is not None:
            preferred_center_freq = int(self.active_plan_center_freq)
        return self._select_center_freq_for_grid(gx, gy, preferred_center_freq=preferred_center_freq)

    def _collect_current_grid_sample(
        self,
    ) -> Tuple[Optional[SpectrumSample], int, bool, Dict[str, float]]:
        ux = int(np.clip(np.round(self.uav_pos[0]), 0, self.Nx - 1))
        uy = int(np.clip(np.round(self.uav_pos[1]), 0, self.Ny - 1))

        if not self._is_uav_sampling_position_valid(np.array([ux, uy], dtype=float)):
            self.last_sample_center_freq = -1
            return None, 0, False, {
                "spatial_revisit_count": 0.0,
                "observed_band_count": 0.0,
                "novelty_ratio": 0.0,
                "repeat_ratio": 0.0,
                "sampling_blocked": 1.0,
                "sampling_blocked_reason": "building",
            }

        center_freq = self._pick_center_freq(ux, uy)
        self.last_sample_center_freq = int(center_freq)
        if self.active_plan_grid is not None and self._is_in_designated_region(ux, uy):
            self.active_plan_center_freq = int(center_freq)

        sample, newly_sampled_freqs, newly_visited_spatial, sampling_stats = self._collect_grid_sample(
            position=np.array([ux, uy], dtype=float),
            center_freq=center_freq,
        )
        return sample, newly_sampled_freqs, newly_visited_spatial, sampling_stats

    def _collect_grid_sample(
        self,
        position: np.ndarray,
        center_freq: int,
    ) -> Tuple[Optional[SpectrumSample], int, bool, Dict[str, float]]:
        loc = np.asarray(position, dtype=float).reshape(2).copy()
        gx = int(np.clip(np.round(loc[0]), 0, self.Nx - 1))
        gy = int(np.clip(np.round(loc[1]), 0, self.Ny - 1))
        loc = np.array([gx, gy], dtype=float)

        omega, observed_bands = build_observe_mask(
            num_bands=self.K,
            center_freq=center_freq,
            band_width=self.sensing_band_num,
        )
        prior_spatial_visits = float(self.local_spatial_visit[gx, gy])
        was_visited = bool(np.any(self.sampled_mask[gx, gy, :]))
        newly_sampled_freqs = int(np.sum(~self.sampled_mask[gx, gy, observed_bands]))
        observed_band_count = int(observed_bands.size)
        novelty_ratio = float(newly_sampled_freqs) / float(max(1, observed_band_count))
        sampling_stats = {
            "spatial_revisit_count": prior_spatial_visits,
            "observed_band_count": float(observed_band_count),
            "novelty_ratio": novelty_ratio,
            "repeat_ratio": 1.0 - novelty_ratio,
            "sampling_blocked": 0.0,
            "sampling_blocked_reason": "",
        }

        gamma_full = self.sim_data.get_data_at_newpos(
            position=loc,
            add_noise=True,
            quantized=False,
        ).astype(float)
        gamma_sparse = gamma_full * omega.astype(float)
        measurements = gamma_full[observed_bands]
        self.sampled_mask[gx, gy, observed_bands] = True
        newly_visited_spatial = (not was_visited) and (newly_sampled_freqs > 0)

        self.action_visit[gx, gy, observed_bands] += 1.0
        self.local_spatial_visit[gx, gy] += 1.0
        self.total_collected_samples += 1

        # Keep repeated noisy observations even when this whole sensing window
        # was already sampled, so data delivery / planner refresh does not stall.
        sample = SpectrumSample(
            position=loc.astype(float).copy(),
            freq_group_idx=int(center_freq),
            freq_band_indices=observed_bands.astype(np.int32),
            measurements=measurements.astype(float),
            gamma=gamma_sparse.astype(float),
            omega=omega.astype(np.int32),
            timestamp=self.current_step,
        )
        return sample, newly_sampled_freqs, newly_visited_spatial, sampling_stats

    def _refresh_planner_outputs(
        self,
        seed_offset: int,
        force_ensemble: bool = False,
        trigger_reason: str = "",
        trigger_sample_count: Optional[int] = None,
        trigger_effective_sample_count: Optional[int] = None,
        trigger_fusion_meta: Optional[Dict[str, object]] = None,
    ) -> Optional[Dict[str, object]]:
        min_samples = int(self.config.planner.min_samples_for_ensemble)
        planner_sample_count = self._planner_sample_count()
        effective_sample_count = self._planner_effective_sample_count()
        ensemble_event: Optional[Dict[str, object]] = None

        if effective_sample_count < min_samples:
            self.planner_targets = []
            self.latest_mean_map = self.radio_map_state.spectrum_map.copy()
            self.latest_var_map = np.ones((self.Nx, self.Ny, self.K), dtype=float)
            self.latest_var_map[self.sampled_mask] = 0.0
            self.uncertainty = UncertaintyMap(
                spatial_uncertainty=np.mean(self.latest_var_map, axis=2),
                frequency_uncertainty=np.mean(self.latest_var_map, axis=(0, 1)),
                joint_uncertainty=self.latest_var_map.copy(),
            )
            return None

        if force_ensemble:
            cached_outputs = None
            if hasattr(self.td, "get_latest_ensemble_outputs"):
                cached_outputs = self.td.get_latest_ensemble_outputs(
                    expected_sample_count=planner_sample_count,
                )
            if cached_outputs is not None:
                mean_map, var_map = cached_outputs
            else:
                obs_locs = np.asarray([s.position for s in self.delivered_samples], dtype=float)
                gamma = np.asarray([s.gamma for s in self.delivered_samples], dtype=float)
                omega = np.asarray([s.omega for s in self.delivered_samples], dtype=np.int32)

                try:
                    mean_map, var_map, _ = ensemble_reconstruct_maps(
                        obs_locs=obs_locs,
                        gamma=gamma,
                        omega=omega,
                        n_sources=1,
                        grid_size=(self.Nx, self.Ny),
                        grid_points=self.grid_points,
                        bounds=self.bounds,
                        i_mask=self.I_mask,
                        m_ens=self.config.planner.ensemble_size,
                        seed=self.config.mappo.seed + int(seed_offset),
                        quality_weighted=bool(self.config.planner.ensemble_quality_weighted),
                        mu=self.config.planner.iibtd_mu,
                        nu=self.config.planner.iibtd_nu,
                        kernel_bandwidth=self.config.planner.iibtd_kernel_bandwidth,
                        solver_backend=self.config.planner.iibtd_backend,
                        solver_device=(
                            self.config.mappo.device
                            if str(self.config.planner.iibtd_device).strip().lower() == "auto"
                            else self.config.planner.iibtd_device
                        ),
                        du_iibtd_checkpoints=self.config.planner.du_iibtd_checkpoints,
                        du_iibtd_min_sensors_for_update=self.config.planner.du_iibtd_min_sensors_for_update,
                        du_iibtd_update_batch_size=self.config.planner.du_iibtd_update_batch_size,
                        member_init_jitter_scale=self.config.planner.ensemble_init_jitter_scale,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "DU-IIBTD planner refresh failed "
                        f"during {trigger_reason or 'forced_refresh'!r}."
                    ) from exc
            self.latest_mean_map = mean_map
            self.latest_var_map = np.maximum(var_map, 0.0)

            self.uncertainty = UncertaintyMap(
                spatial_uncertainty=np.mean(self.latest_var_map, axis=2),
                frequency_uncertainty=np.mean(self.latest_var_map, axis=(0, 1)),
                joint_uncertainty=self.latest_var_map.copy(),
            )
            effective_trigger_count = (
                int(self.pending_ensemble_effective_sample_count)
                if trigger_effective_sample_count is None
                else int(trigger_effective_sample_count)
            )
            raw_trigger_count = (
                int(self.pending_ensemble_sample_count)
                if trigger_sample_count is None
                else int(trigger_sample_count)
            )
            fusion_meta = (
                dict(self._empty_fusion_meta())
                if trigger_fusion_meta is None
                else dict(trigger_fusion_meta)
            )
            ensemble_event = self._record_ensemble_event(
                reason=trigger_reason or "forced_refresh",
                trigger_sample_count=raw_trigger_count,
                extra_fields={
                    "trigger_raw_sample_count": int(raw_trigger_count),
                    "trigger_effective_sample_count": int(effective_trigger_count),
                    "trigger_fusion_raw_count": int(fusion_meta.get("raw_count", raw_trigger_count)),
                    "trigger_fusion_fused_count": int(
                        fusion_meta.get("fused_count", effective_trigger_count)
                    ),
                    "trigger_fusion_compression_ratio": float(
                        fusion_meta.get("compression_ratio", 1.0)
                    ),
                },
            )

        if force_ensemble:
            self._clear_suppressed_planner_targets()
        self.planner_targets = self._select_planner_candidates(top_k=max(1, self.target_count))
        return ensemble_event

    def _build_fallback_targets(
        self,
        candidate_mask: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
    ) -> List[PlannerTarget]:
        if candidate_mask is None:
            candidate_mask = np.ones((self.Nx, self.Ny), dtype=bool)
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        candidate_mask &= ~self.scene.get_occupancy_grid()

        remaining_grid = np.argwhere(candidate_mask & ~np.any(self.sampled_mask, axis=2))
        if remaining_grid.size == 0:
            remaining_grid = np.argwhere(candidate_mask & ~np.all(self.sampled_mask, axis=2))
        if remaining_grid.size == 0:
            remaining_grid = np.argwhere(candidate_mask)
        if remaining_grid.size == 0:
            remaining_grid = np.argwhere(~self.scene.get_occupancy_grid())
        if remaining_grid.size == 0:
            raise RuntimeError("No non-building grid cells are available for planner fallback targets.")

        dists = np.linalg.norm(remaining_grid.astype(float) - self.uav_pos[np.newaxis, :], axis=1)
        candidate_k = min(
            max(1, int(self.target_count if top_k is None else top_k)),
            remaining_grid.shape[0],
        )
        top_indices = np.argsort(dists)[:candidate_k]
        targets: List[PlannerTarget] = []
        for idx in top_indices.tolist():
            gx, gy = remaining_grid[int(idx)]
            center_freq = self._select_center_freq_for_grid(int(gx), int(gy))
            targets.append(PlannerTarget(
                gx=int(gx),
                gy=int(gy),
                x=float(gx),
                y=float(gy),
                center_freq=center_freq,
                score=float(self.latest_var_map[int(gx), int(gy), center_freq]),
            ))
        return targets

    def _get_current_observation_target(self) -> Optional[PlannerTarget]:
        """Return the active target used consistently across agent observations."""
        if self.active_plan_grid is not None:
            gx, gy = int(self.active_plan_grid[0]), int(self.active_plan_grid[1])
            center_freq = int(self.active_plan_center_freq) if self.active_plan_center_freq is not None else 0
            score = 0.0
            for cand in self.planner_targets:
                if int(cand.gx) == gx and int(cand.gy) == gy:
                    if self.active_plan_center_freq is None:
                        center_freq = int(cand.center_freq)
                    score = float(cand.score)
                    break
            return PlannerTarget(
                gx=gx,
                gy=gy,
                x=float(gx),
                y=float(gy),
                center_freq=center_freq,
                score=score,
            )

        if self.planner_targets:
            cand = self.planner_targets[0]
            return PlannerTarget(
                gx=int(cand.gx),
                gy=int(cand.gy),
                x=float(cand.x),
                y=float(cand.y),
                center_freq=int(cand.center_freq),
                score=float(cand.score),
            )
        if self._bootstrap_target_is_active():
            return PlannerTarget(
                gx=int(self.bootstrap_target.gx),
                gy=int(self.bootstrap_target.gy),
                x=float(self.bootstrap_target.x),
                y=float(self.bootstrap_target.y),
                center_freq=int(self.bootstrap_target.center_freq),
                score=float(self.bootstrap_target.score),
            )
        return None

    def _compute_reward(
        self,
        uav_move_dist: float,
        ugv_move_dist: float,
        uav_progress_ratio: float,
        ugv_progress_ratio: float,
        ugv_progress_to_uav_ratio: float,
        ugv_progress_to_target_ratio: float,
        target_source: str,
        map_updated: bool,
        reward_target_grid: Optional[Tuple[int, int]],
        target_reached: bool,
        queue_bits_before_tx: float,
        queue_bits_after_tx: float,
        data_produced_bits: float,
        data_delivered_bits: float,
        novel_data_delivered_bits: float,
        dropped_packets: int,
        newly_sampled_freqs: int,
        newly_visited_spatial: bool,
        sampling_stats: Dict[str, float],
    ) -> Tuple[Dict[str, float], dict]:
        rc = self.config.reward

        curr_global_unc = float(np.mean(self.uncertainty.spatial_uncertainty))
        if map_updated:
            prev_nmse = max(float(self.prev_nmse), 1e-8)
            curr_nmse = float(self.radio_map_state.nmse)
            delta_nmse = self.prev_nmse - curr_nmse
            delta_nmse_norm = delta_nmse / prev_nmse
            self.prev_nmse = curr_nmse
        else:
            curr_nmse = float(self.radio_map_state.nmse)
            delta_nmse = 0.0
            delta_nmse_norm = 0.0

        curr_goal_local_unc = 0.0
        delta_unc = 0.0
        delta_unc_norm = 0.0
        # Map-quality terms: normalized NMSE / uncertainty improvement.
        nmse_signed_clip = float(rc.nmse_signed_clip)
        if nmse_signed_clip > 0.0:
            delta_nmse_norm_clipped = float(
                np.clip(delta_nmse_norm, -nmse_signed_clip, nmse_signed_clip)
            )
        else:
            delta_nmse_norm_clipped = float(delta_nmse_norm)
        r_nmse = rc.alpha_nmse * delta_nmse_norm_clipped
        target_nmse = max(float(rc.accuracy_target_nmse), 1e-8)
        nmse_target_gap = max(curr_nmse - target_nmse, 0.0)
        nmse_target_gap_norm = nmse_target_gap / target_nmse
        target_gap_penalty_diag = -abs(float(rc.target_gap_penalty_coef)) * float(
            nmse_target_gap_norm
        )
        r_unc = 0.0

        # Track absolute delivered payload for diagnostics.
        if queue_bits_before_tx > 0:
            tx_throughput = data_delivered_bits / (queue_bits_before_tx + 1e-8)
            tx_novelty_throughput = novel_data_delivered_bits / (queue_bits_before_tx + 1e-8)
        else:
            tx_throughput = 0.0
            tx_novelty_throughput = 0.0
        # Local-goal shaping intentionally disables throughput reward to keep the
        # policy focused on reaching the planner-selected local target.
        r_tx = 0.0

        # Penalize only the backlog that remains after transmission.
        queue_bits_norm = queue_bits_after_tx / (self.q_max_bits + 1e-8)

        curr_q = len(self.uav_data_queue)
        dropped_norm = float(dropped_packets) / float(self.max_q)
        # Penalize both residual backlog and packets dropped by queue overflow.
        r_queue = -rc.gamma_queue * float(queue_bits_norm + dropped_norm)

        observed_band_count = max(float(sampling_stats.get("observed_band_count", 0.0)), 0.0)
        r_new_freq = rc.lambda_new_freq * float(newly_sampled_freqs)
        r_new_spatial = rc.lambda_new_spatial * float(int(newly_visited_spatial))

        uav_progress_norm = float(uav_progress_ratio)
        ugv_progress_norm = float(ugv_progress_ratio)
        ugv_progress_to_uav_norm = float(ugv_progress_to_uav_ratio)
        ugv_progress_to_target_norm = float(ugv_progress_to_target_ratio)
        progress_scale = (
            float(rc.bootstrap_progress_scale)
            if str(target_source) == "bootstrap"
            else 1.0
        )
        uav_forward = max(float(uav_progress_norm), 0.0)
        uav_backward = max(float(-uav_progress_norm), 0.0)
        ugv_forward = max(float(ugv_progress_norm), 0.0)
        ugv_backward = max(float(-ugv_progress_norm), 0.0)
        r_uav_progress = progress_scale * (
            (rc.lambda_uav_progress * uav_forward)
            - (rc.lambda_uav_backtrack * uav_backward)
        )
        r_ugv_progress = progress_scale * (
            (rc.lambda_ugv_progress * ugv_forward)
            - (rc.lambda_ugv_backtrack * ugv_backward)
        )
        r_goal_arrival = progress_scale * float(rc.local_goal_arrival_bonus) * float(
            int(target_reached and reward_target_grid is not None)
        )
        r_progress = r_uav_progress + r_ugv_progress + r_goal_arrival
        spatial_revisit_count = max(float(sampling_stats.get("spatial_revisit_count", 0.0)), 0.0)
        repeat_ratio = float(sampling_stats.get("repeat_ratio", 0.0))
        novelty_ratio = float(sampling_stats.get("novelty_ratio", 0.0))
        r_revisit = -float(rc.lambda_spatial_revisit) * np.log1p(spatial_revisit_count) * repeat_ratio
        ugv_building_clearance_level = self._ugv_building_clearance_level(self.ugv_pos)
        ugv_building_clearance_norm = self._ugv_building_clearance_norm(self.ugv_pos)
        ugv_building_clearance_deficit = max(0.0, 1.0 - ugv_building_clearance_norm)
        r_ugv_building_clearance = (
            -max(0.0, float(rc.lambda_ugv_building_clearance)) * ugv_building_clearance_deficit
        )

        shared_reward = (
            r_nmse
            + r_unc
            + r_new_freq
            + r_new_spatial
            + r_queue
            + r_progress
            + r_revisit
            + r_ugv_building_clearance
        )

        queue_norm = min(curr_q / float(max(self.queue_ref, 1e-8)), 1.0)
        queue_occupancy = curr_q / float(self.max_q)

        rewards = {
            "team_reward": float(shared_reward),
            "uav_reward": float(shared_reward),
            "ugv_reward": float(shared_reward),
        }
        info = {
            "r_nmse": float(r_nmse),
            "r_unc": float(r_unc),
            "r_new_freq": float(r_new_freq),
            "r_new_spatial": float(r_new_spatial),
            "r_tx": float(r_tx),
            "r_queue": float(r_queue),
            "r_progress": float(r_progress),
            "r_revisit": float(r_revisit),
            "r_uav_progress": float(r_uav_progress),
            "r_ugv_progress": float(r_ugv_progress),
            "r_goal_arrival": float(r_goal_arrival),
            "r_ugv_building_clearance": float(r_ugv_building_clearance),
            "tx_throughput": float(tx_throughput),
            "tx_novelty_throughput": float(tx_novelty_throughput),
            "queue_bits_before_tx": float(queue_bits_before_tx),
            "queue_bits_after_tx": float(queue_bits_after_tx),
            "queue_bits_norm": float(queue_bits_norm),
            "data_produced_bits": float(data_produced_bits),
            "data_delivered_bits": float(data_delivered_bits),
            "novel_data_delivered_bits": float(novel_data_delivered_bits),
            "data_transmitted_bits": float(data_delivered_bits),
            "team_reward": float(shared_reward),
            "uav_reward": float(shared_reward),
            "ugv_reward": float(shared_reward),
            "delta_nmse": float(delta_nmse),
            "delta_nmse_norm": float(delta_nmse_norm),
            "delta_nmse_norm_clipped": float(delta_nmse_norm_clipped),
            "delta_unc": float(delta_unc),
            "delta_unc_norm": float(delta_unc_norm),
            "target_nmse": float(target_nmse),
            "nmse_target_gap": float(nmse_target_gap),
            "nmse_target_gap_norm": float(nmse_target_gap_norm),
            "target_gap_penalty_diag": float(target_gap_penalty_diag),
            "goal_local_unc_prev": 0.0,
            "goal_local_unc_curr": float(curr_goal_local_unc),
            "global_unc_mean": float(curr_global_unc),
            "local_goal_radius": int(round(max(self.local_planner_radius, 1))),
            "goal_normalization_den": 1.0,
            "queue_norm": float(queue_norm),
            "queue_occupancy": float(queue_occupancy),
            "dropped": int(dropped_packets),
            "dropped_norm": float(dropped_norm),
            "uav_move_dist": float(uav_move_dist),
            "ugv_move_dist": float(ugv_move_dist),
            "uav_progress": float(uav_progress_norm),
            "uav_progress_norm": float(uav_progress_norm),
            "ugv_progress": float(ugv_progress_norm),
            "ugv_progress_norm": float(ugv_progress_norm),
            "ugv_progress_to_uav": float(ugv_progress_to_uav_norm),
            "ugv_progress_to_uav_norm": float(ugv_progress_to_uav_norm),
            "ugv_progress_to_target": float(ugv_progress_to_target_norm),
            "ugv_progress_to_target_norm": float(ugv_progress_to_target_norm),
            "ugv_progress_uav_weight": float(self.ugv_progress_uav_weight),
            "ugv_progress_target_weight": float(self.ugv_progress_target_weight),
            "target_source": str(target_source),
            "progress_scale": float(progress_scale),
            "progress_metric": (
                "relative_improvement__delta_over_prev_distance__"
                "uav_to_goal__ugv_to_weighted(grid_uav,target_shortest_path)"
            ),
            "observed_band_count": float(observed_band_count),
            "spatial_revisit_count": float(spatial_revisit_count),
            "sample_novelty_ratio": float(novelty_ratio),
            "sample_repeat_ratio": float(repeat_ratio),
            "ugv_building_clearance_level": int(ugv_building_clearance_level),
            "ugv_building_safe_clearance": int(self.ugv_building_safe_clearance),
            "ugv_building_clearance_norm": float(ugv_building_clearance_norm),
            "ugv_building_clearance_deficit": float(ugv_building_clearance_deficit),
            "sampling_blocked": int(float(sampling_stats.get("sampling_blocked", 0.0)) > 0.5),
            "sampling_blocked_reason": str(sampling_stats.get("sampling_blocked_reason", "")),
            "newly_sampled_freqs": int(newly_sampled_freqs),
            "newly_visited_spatial": int(newly_visited_spatial),
            "r_terminal": 0.0,
            "target_nmse_reached": 0,
            "energy_depleted": 0,
            "terminal_failure": 0,
        }
        return rewards, info

    def _build_observations(self) -> Dict[str, np.ndarray]:
        current_target = self._get_current_observation_target()
        snr_norm = float(np.tanh(self.ugv_channel_info.snr_db / self.snr_norm_den))
        queue_norm = min(len(self.uav_data_queue) / float(self.queue_ref), 1.0)
        uav_energy_norm = float(self.uav_energy / self.uav_energy_den)
        los_link_obs = self._build_uav_ugv_los_obs()
        return {
            "uav_obs": self._build_uav_obs(
                current_target=current_target,
                queue_norm=queue_norm,
                uav_energy_norm=uav_energy_norm,
            ),
            "uav_action_mask": self._build_uav_action_mask(),
            "ugv_obs": self._build_ugv_obs(
                current_target=current_target,
                queue_norm=queue_norm,
                snr_norm=snr_norm,
                los_link_obs=los_link_obs,
            ),
            "ugv_action_mask": self._build_ugv_action_mask(),
            "critic_state": self._build_critic_state(
                current_target=current_target,
                snr_norm=snr_norm,
                queue_norm=queue_norm,
                uav_energy_norm=uav_energy_norm,
                los_link_obs=los_link_obs,
            ),
        }

    def _build_uav_ugv_los_obs(self) -> np.ndarray:
        return np.array([float(self.ugv_channel_info.los)], dtype=float)

    def _build_uav_obs(
        self,
        current_target: Optional[PlannerTarget] = None,
        queue_norm: Optional[float] = None,
        uav_energy_norm: Optional[float] = None,
    ) -> np.ndarray:
        parts = []
        if queue_norm is None:
            queue_norm = min(len(self.uav_data_queue) / float(self.queue_ref), 1.0)
        if uav_energy_norm is None:
            uav_energy_norm = float(self.uav_energy / self.uav_energy_den)

        parts.append(self._normalize_grid_position(self.uav_pos))
        parts.append(np.array([uav_energy_norm], dtype=float))
        parts.append(np.array([queue_norm], dtype=float))
        parts.append(np.array([self.current_bw_ratio], dtype=float))
        parts.append(self._encode_local_goal_for_uav_obs(current_target=current_target))
        parts.append(self._normalize_grid_position(self.ugv_pos))

        obs = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _build_uav_action_mask(self) -> np.ndarray:
        cache_key = self._uav_action_mask_cache_key()
        cached_mask = self._uav_action_mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask.copy()

        valid_dirs = np.zeros(self.uav_direction_choices, dtype=bool)
        for direction_choice_idx, direction_idx in enumerate(self.uav_direction_ids):
            valid_dirs[direction_choice_idx] = self._can_follow_direction(
                position=self.uav_pos,
                direction_idx=direction_idx,
                step_count=self.uav_step_count,
                validator=self.scene.is_uav_position_valid,
                stop_at_target=True,
            )
        mask = np.repeat(valid_dirs, self.num_bw_choices)
        self._cache_store(self._uav_action_mask_cache, cache_key, mask.copy())
        return mask

    def _build_ugv_action_mask(self) -> np.ndarray:
        cache_key = self._ugv_action_mask_cache_key()
        cached_mask = self._ugv_action_mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask.copy()

        mask = np.zeros(self.ugv_action_size, dtype=bool)
        for direction_idx in range(self.ugv_action_size):
            mask[direction_idx] = self._can_follow_direction(
                position=self.ugv_pos,
                direction_idx=direction_idx,
                step_count=self.ugv_step_count,
                validator=self.scene.is_ugv_position_valid,
            )
        self._cache_store(self._ugv_action_mask_cache, cache_key, mask.copy())
        return mask

    def _encode_local_goal_for_uav_obs(
        self,
        current_target: Optional[PlannerTarget] = None,
    ) -> np.ndarray:
        if current_target is None:
            current_target = self._get_current_observation_target()
        if current_target is not None:
            goal_pos = np.array([float(current_target.gx), float(current_target.gy)], dtype=float)
            goal_dist = manhattan_distance(self.uav_pos, goal_pos)
            goal_direction = self._normalize_delta_by_target_distance(
                source_pos=self.uav_pos,
                target_pos=goal_pos,
            )
            return np.array(
                [
                    float(goal_direction[0]),
                    float(goal_direction[1]),
                    np.clip(goal_dist / self.global_manhattan_den, 0.0, 1.0),
                    float(current_target.center_freq) / self.max_freq_den,
                ],
                dtype=float,
            )
        return np.zeros(4, dtype=float)

    def _build_ugv_obs(
        self,
        current_target: Optional[PlannerTarget] = None,
        queue_norm: Optional[float] = None,
        snr_norm: Optional[float] = None,
        los_link_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if queue_norm is None:
            queue_norm = min(len(self.uav_data_queue) / float(self.queue_ref), 1.0)
        if snr_norm is None:
            snr_norm = float(np.tanh(self.ugv_channel_info.snr_db / self.snr_norm_den))
        if los_link_obs is None:
            los_link_obs = self._build_uav_ugv_los_obs()

        ugv_pos_norm = self._normalize_grid_position(self.ugv_pos)
        rel_uav = np.clip(
            (self.uav_pos - self.ugv_pos) / self.safe_grid_norm_den,
            -1.0,
            1.0,
        ).astype(float)

        if current_target is None:
            current_target = self._get_current_observation_target()
        if current_target is not None:
            target_pos = np.array([float(current_target.gx), float(current_target.gy)], dtype=float)
            rel_target = self._normalize_delta_by_target_distance(
                source_pos=self.ugv_pos,
                target_pos=target_pos,
            )
            target_dist = self._occupancy_shortest_path_distance(
                self.ugv_pos,
                target_pos,
            )
            if np.isfinite(target_dist):
                target_dist_norm = float(np.clip(target_dist / self.global_manhattan_den, 0.0, 1.0))
            else:
                target_dist_norm = 1.0
        else:
            rel_target = np.zeros(2, dtype=float)
            target_dist_norm = 1.0

        parts = []
        parts.append(ugv_pos_norm)
        parts.append(rel_uav)
        parts.append(rel_target)
        parts.append(np.array([target_dist_norm], dtype=float))
        parts.append(np.array([queue_norm], dtype=float))
        parts.append(np.array([snr_norm], dtype=float))
        parts.append(np.asarray(los_link_obs, dtype=float).reshape(1))
        parts.append(self._build_ugv_building_obs())
        obs = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _build_critic_state(
        self,
        current_target: Optional[PlannerTarget] = None,
        snr_norm: Optional[float] = None,
        queue_norm: Optional[float] = None,
        uav_energy_norm: Optional[float] = None,
        los_link_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if snr_norm is None:
            snr_norm = float(np.tanh(self.ugv_channel_info.snr_db / self.snr_norm_den))
        if queue_norm is None:
            queue_norm = min(len(self.uav_data_queue) / float(self.queue_ref), 1.0)
        if uav_energy_norm is None:
            uav_energy_norm = float(self.uav_energy / self.uav_energy_den)
        if los_link_obs is None:
            los_link_obs = self._build_uav_ugv_los_obs()

        parts = []
        parts.append(self._normalize_grid_position(self.uav_pos))
        parts.append(self._normalize_grid_position(self.ugv_pos))
        parts.append(np.array([uav_energy_norm], dtype=float))
        parts.append(np.array([snr_norm], dtype=float))
        parts.append(np.array([queue_norm], dtype=float))
        parts.append(np.array([self.radio_map_state.nmse], dtype=float))
        parts.append(np.array([self.current_bw_ratio], dtype=float))
        parts.append(np.asarray(los_link_obs, dtype=float).reshape(1))
        parts.append(np.array([self._ugv_building_clearance_norm(self.ugv_pos)], dtype=float))
        parts.append(self._extract_planner_state_features(current_target=current_target))
        state = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    def _normalize_grid_position(self, position: np.ndarray) -> np.ndarray:
        return (np.asarray(position, dtype=float) / self.safe_grid_norm_den).astype(float)

    def _normalize_delta_by_target_distance(
        self,
        source_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        delta = np.asarray(target_pos, dtype=float) - np.asarray(source_pos, dtype=float)
        dist = max(manhattan_distance(source_pos, target_pos), 1.0)
        return np.clip(delta / dist, -1.0, 1.0).astype(float)

    def _extract_planner_state_features(
        self,
        current_target: Optional[PlannerTarget] = None,
    ) -> np.ndarray:
        n_features = self.config.obs.num_planner_features
        features = np.zeros(n_features, dtype=float)
        if current_target is None:
            current_target = self._get_current_observation_target()
        if current_target is None:
            return features

        target_grid_pos = np.array([float(current_target.gx), float(current_target.gy)], dtype=float)
        idx = 0
        if idx + 1 < n_features:
            features[idx] = target_grid_pos[0] / self.max_grid_x_den
            features[idx + 1] = target_grid_pos[1] / self.max_grid_y_den
            idx += 2

        if idx < n_features:
            features[idx] = float(current_target.center_freq) / self.max_freq_den
            idx += 1

        if idx < n_features:
            features[idx] = float(np.linalg.norm(self.uav_pos - target_grid_pos) / (self.max_grid_diag + 1e-9))
            idx += 1
        if idx < n_features:
            features[idx] = float(np.linalg.norm(self.ugv_pos - target_grid_pos) / (self.max_grid_diag + 1e-9))
        return features


class _VecResetSeedMixin:
    def _peek_reset_seed(self, env_idx: int) -> int:
        count = int(self._env_reset_counts[env_idx])
        return _compute_vec_reset_seed(self.base_seed, env_idx, count, self.num_envs)


class VecUAVUGVEnvironment(_VecResetSeedMixin):
    """Simple synchronous vectorized wrapper."""

    def __init__(
        self,
        num_envs: int,
        config: Config,
        env_factory=None,
        minimal_info: bool = False,
    ):
        self.num_envs = num_envs
        self.config = config
        self.minimal_info = bool(minimal_info)
        if env_factory is not None:
            self.envs = [env_factory(env_idx) for env_idx in range(num_envs)]
        else:
            self.envs = [self._make_default_env(i) for i in range(num_envs)]
        self.base_seed = int(config.mappo.seed)
        self._env_reset_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.obs_dims = self.envs[0].get_obs_dims()
        self.action_dims = self.envs[0].get_action_dims()

    def _make_default_env(self, idx: int) -> UAVUGVEnvironment:
        sim_data = SimDataGen(self.config, seed=42 + idx)
        td = IIBTD_opt(
            config=self.config,
            grid_coords=sim_data.grid_coords,
            bounds=sim_data.bounds,
            i_mask=sim_data.I_mask,
            n_sources=1,
        )
        return UAVUGVEnvironment(
            config=self.config,
            tensor_decomp=td,
            sim_data=sim_data,
            scene_map=GridScene(
                self.config,
                occupancy_grid=sim_data.get_building_mask(),
                building_heights=sim_data.get_building_heights(),
            ),
            minimal_info=self.minimal_info,
        )


    def _next_reset_seed(self, env_idx: int) -> int:
        seed = self._peek_reset_seed(env_idx)
        self._env_reset_counts[env_idx] = int(self._env_reset_counts[env_idx]) + 1
        return seed

    def reset(self) -> Dict[str, np.ndarray]:
        all_obs = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self._next_reset_seed(i))
            all_obs.append(obs)
        return _stack_vec_obs(all_obs)

    def step(
        self,
        uav_actions: np.ndarray,
        ugv_actions: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict]]:
        results = []
        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(int(uav_actions[i]), int(ugv_actions[i]))
            if term or trunc:
                info["terminal_obs"] = obs
                obs, _ = env.reset(seed=self._next_reset_seed(i))
            results.append((obs, rew, term, trunc, info))

        return _pack_vec_step_results(results)

    def close(self) -> None:
        for env in getattr(self, "envs", []):
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()


class SubprocVecUAVUGVEnvironment(_VecResetSeedMixin):
    """Spawn-based vectorized wrapper that instantiates each env inside its worker."""

    def __init__(
        self,
        num_envs: int,
        config: Config,
        shared_data: Dict,
        minimal_info: bool = False,
    ):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if shared_data is None:
            raise ValueError("SubprocVecUAVUGVEnvironment requires non-null shared_data")

        available_methods = mp.get_all_start_methods()
        if "spawn" not in available_methods:
            raise RuntimeError(
                "SubprocVecUAVUGVEnvironment requires the 'spawn' start method."
            )

        self.num_envs = int(num_envs)
        self.config = config
        self.base_seed = int(config.mappo.seed)
        self.shared_data = shared_data
        self.minimal_info = bool(minimal_info)
        self._env_reset_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.closed = False
        self.ctx = mp.get_context("spawn")
        self.remote_poll_interval_s = _SUBPROC_POLL_INTERVAL_SECONDS
        self.worker_response_timeout_s = _SUBPROC_RESPONSE_TIMEOUT_SECONDS
        self.remotes = []
        self.processes = []
        self.obs_dims = None
        self.action_dims = None

        try:
            for env_idx in range(self.num_envs):
                parent_remote, worker_remote = self.ctx.Pipe()
                process = self.ctx.Process(
                    target=_subproc_env_worker,
                    args=(
                        worker_remote,
                        self.config,
                        self.shared_data,
                        env_idx,
                        self.minimal_info,
                    ),
                    daemon=True,
                )
                process.start()
                worker_remote.close()

                self.remotes.append(parent_remote)
                self.processes.append(process)

                ready_payload = self._recv_worker_payload(env_idx, expected_kind=_SUBPROC_READY)
                obs_dims = ready_payload["obs_dims"]
                action_dims = ready_payload["action_dims"]
                if env_idx == 0:
                    self.obs_dims = obs_dims
                    self.action_dims = action_dims
                elif obs_dims != self.obs_dims or action_dims != self.action_dims:
                    self._abort(
                        "Subproc worker dimensions do not match across environments: "
                        f"env 0 has obs={self.obs_dims}, action={self.action_dims}, "
                        f"but env {env_idx} has obs={obs_dims}, action={action_dims}."
                    )
        except Exception:
            self.close()
            raise


    def _mark_env_reset(self, env_idx: int) -> None:
        self._env_reset_counts[env_idx] = int(self._env_reset_counts[env_idx]) + 1

    def _worker_label(self, env_idx: int) -> str:
        process = self.processes[env_idx]
        return f"worker {env_idx} (pid={process.pid}, exitcode={process.exitcode})"

    def _abort(self, message: str, error_type=RuntimeError) -> None:
        try:
            self.close()
        except Exception:
            pass
        raise error_type(message)

    def _recv_worker_payload(self, env_idx: int, expected_kind: str):
        remote = self.remotes[env_idx]
        process = self.processes[env_idx]
        waited_s = 0.0

        while True:
            try:
                if remote.poll(self.remote_poll_interval_s):
                    break
            except (EOFError, OSError) as exc:
                self._abort(
                    f"Failed while polling {self._worker_label(env_idx)} for '{expected_kind}': {exc}"
                )

            if not process.is_alive():
                self._abort(
                    f"{self._worker_label(env_idx)} exited unexpectedly while waiting for "
                    f"'{expected_kind}'."
                )

            waited_s += self.remote_poll_interval_s
            if waited_s >= self.worker_response_timeout_s:
                self._abort(
                    f"Timed out after {self.worker_response_timeout_s:.1f}s waiting for "
                    f"{self._worker_label(env_idx)} to send '{expected_kind}'.",
                    error_type=TimeoutError,
                )

        try:
            message = remote.recv()
        except (EOFError, OSError) as exc:
            self._abort(
                f"Failed while receiving '{expected_kind}' from {self._worker_label(env_idx)}: {exc}"
            )

        if not isinstance(message, tuple) or len(message) != 2:
            self._abort(
                f"{self._worker_label(env_idx)} sent an invalid message: {message!r}"
            )

        kind, payload = message
        if kind == _SUBPROC_ERROR:
            stage = payload.get("stage", "unknown")
            error_type_name = payload.get("type", "UnknownError")
            error_message = payload.get("message", "")
            error_traceback = payload.get("traceback", "<worker traceback unavailable>")
            self._abort(
                f"{self._worker_label(env_idx)} failed during '{stage}' with "
                f"{error_type_name}: {error_message}\n"
                f"Worker traceback:\n{error_traceback}"
            )
        if kind != expected_kind:
            self._abort(
                f"{self._worker_label(env_idx)} sent unexpected message kind "
                f"{kind!r}; expected {expected_kind!r}."
            )
        return payload

    def _send_command(self, env_idx: int, command: str, payload) -> None:
        try:
            self.remotes[env_idx].send((command, payload))
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._abort(
                f"Failed to send '{command}' to {self._worker_label(env_idx)}: {exc}"
            )

    def reset(self) -> Dict[str, np.ndarray]:
        for env_idx, remote in enumerate(self.remotes):
            self._send_command(env_idx, "reset", self._peek_reset_seed(env_idx))
        obs_list = [
            self._recv_worker_payload(env_idx, expected_kind=_SUBPROC_RESULT)
            for env_idx in range(self.num_envs)
        ]
        for env_idx in range(self.num_envs):
            self._mark_env_reset(env_idx)
        return _stack_vec_obs(obs_list)

    def step(
        self,
        uav_actions: np.ndarray,
        ugv_actions: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict]]:
        for env_idx in range(self.num_envs):
            self._send_command(
                env_idx,
                "step",
                (
                    int(uav_actions[env_idx]),
                    int(ugv_actions[env_idx]),
                    self._peek_reset_seed(env_idx),
                ),
            )

        results = [
            self._recv_worker_payload(env_idx, expected_kind=_SUBPROC_RESULT)
            for env_idx in range(self.num_envs)
        ]
        for env_idx, result in enumerate(results):
            if bool(result[2]) or bool(result[3]):
                self._mark_env_reset(env_idx)
        return _pack_vec_step_results(results)

    def close(self) -> None:
        if self.closed:
            return

        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass

        for remote in self.remotes:
            try:
                remote.close()
            except OSError:
                pass

        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

        self.closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
