"""
Multi-Agent RL Environment for UAV-UGV cooperative active sensing.

Key behavior:
- UGV reconstructs radio map from delivered spectrum samples.
- UGV runs ensemble resampling and provides one most-informative grid target.
- UAV policy samples movement and sensing-communication split from two independent heads.
- UAV and UGV both move only on integer grid points.
- UAV samples the current grid point directly from GT with observation noise.
- Delivered data are processed by priority: designated-region samples every step, non-designated samples at intervals.
- UGV keeps 5-way movement actions (stay/east/north/west/south).
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from active_sampling import (
    build_acquisition_space,
    build_observe_mask,
    ensemble_reconstruct_maps,
    adaptive_keep_ratio,
    normalize_score_map,
    select_top_k_grid_candidates,
    select_quantization_bits_from_uncertainty,
)
from config import Config
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


def _subproc_env_worker(remote, env_factory, env_idx: int) -> None:
    """Worker loop for one rollout environment."""
    env: Optional[UAVUGVEnvironment] = None
    try:
        env = env_factory(env_idx)
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
                    obs, _ = env.reset(seed=int(payload))
                    _subproc_send(remote, _SUBPROC_RESULT, obs)
                elif cmd == "step":
                    uav_move_action, uav_bw_action, ugv_action, reset_seed = payload
                    obs, rew, term, trunc, info = env.step(
                        int(uav_move_action),
                        int(uav_bw_action),
                        int(ugv_action),
                    )
                    if term or trunc:
                        info["terminal_obs"] = obs
                        info["timeout_bootstrap_critic_state"] = np.asarray(
                            obs["critic_state"],
                            dtype=np.float32,
                        ).copy()
                        obs, _ = env.reset(seed=int(reset_seed))
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
            [r[1].get("team_reward", r[1]["uav_reward"]) for r in results],
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


@dataclass
class DataPacket:
    """Data packet in the transmission queue."""
    sample: SpectrumSample
    size_bits: float
    priority: int = 1
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
    quantization_bits: int = 8


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
    ):
        self.config = config
        self.td = tensor_decomp
        self.sim_data = sim_data
        self.scene = scene_map

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
        self.queue_ref = max(1, int(round(float(config.reward.q_ref))))
        self.low_priority_process_interval = max(1, int(config.planner.low_priority_process_interval))
        self.ensemble_refresh_interval = max(1, int(config.planner.ensemble_refresh_interval))
        self.local_planner_radius = max(1, int(config.planner.local_planner_radius))

        # UAV action heads: movement and bandwidth split are sampled independently.
        self.uav_move_action_size = self.uav_direction_choices
        self.uav_bw_action_size = self.num_bw_choices
        self._reset_counter = 0

        self._setup_observation_spaces()
        self._load_grid_dataset()
        self._init_cached_constants()

        self.ground_truth_map = self.sim_data.get_full_ground_truth_map()
        if hasattr(self.td, "set_ground_truth"):
            self.td.set_ground_truth(self.ground_truth_map)

    def _setup_observation_spaces(self) -> None:
        c = self.config
        # UAV: base(7) + planner target(5) + link state(3) + remaining budgets(2)
        self.uav_obs_dim = 17
        # UGV: base(6) + planner relative target(3) + UAV heading(2) + queue bits(1)
        self.ugv_obs_dim = 12
        # Critic: original 16 + recent deltas(3) + global summary(3)
        self.critic_state_dim = 16 + 3 + 3

    def get_obs_dims(self) -> Dict[str, int]:
        return {
            "uav_obs": self.uav_obs_dim,
            "ugv_obs": self.ugv_obs_dim,
            "critic_state": self.critic_state_dim,
        }

    def get_action_dims(self) -> Dict[str, int]:
        return {
            "uav_move_action": self.uav_move_action_size,
            "uav_bw_action": self.uav_bw_action_size,
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
        self.max_freq_den = float(max(self.K - 1, 1))
        self.max_grid_diag = float(np.hypot(self.max_grid_x_den, self.max_grid_y_den))
        self.uav_step_count = int(self._grid_step_count(float(self.config.uav.step_size)))
        self.ugv_step_count = int(self._grid_step_count(float(self.config.ugv.step_size)))
        self.max_bw_ratio = float(np.max(self.bandwidth_ratios))
        self.max_q = int(max(1, self.queue_ref))
        self.default_quantization_bits = int(self.config.planner.default_quantization_bits)
        self.high_quantization_bits = int(self.config.planner.high_quantization_bits)
        self.low_quantization_bits = int(self.config.planner.low_quantization_bits)
        self.max_raw_step_bits = float(self.K) * float(self.high_quantization_bits)
        self.raw_episode_bit_budget = (
            float(self.config.mappo.episode_max_steps) * self.max_raw_step_bits
        )
        self.max_packet_bits = float(self.K) * float(self.config.comm.data_per_sample)
        self.q_max_bits = float(self.max_q) * self.max_packet_bits
        self.uav_energy_den = float(self.config.uav.max_energy)
        self.uav_info_level_den = 2.0
        self.snr_norm_den = 30.0
        self.max_comm_capacity_bps = self._estimate_reference_comm_capacity()

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

    def _sample_initial_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        max_sep = float(max(1e-6, self.config.planner.init_pair_max_distance))
        center = np.array([(self.Nx - 1) / 2.0, (self.Ny - 1) / 2.0], dtype=np.float64)
        for _ in range(256):
            uav = np.array(
                [
                    self.rng.randint(0, self.Nx),
                    self.rng.randint(0, self.Ny),
                ],
                dtype=np.float64,
            )
            # Keep random starts, but nudge them slightly away from map edges.
            uav = np.rint(0.8 * uav + 0.3 * center)
            x_min = max(0, int(np.floor(uav[0] - max_sep)))
            x_max = min(self.Nx - 1, int(np.ceil(uav[0] + max_sep)))
            y_min = max(0, int(np.floor(uav[1] - max_sep)))
            y_max = min(self.Ny - 1, int(np.ceil(uav[1] + max_sep)))

            nearby = [
                (gx, gy)
                for gx in range(x_min, x_max + 1)
                for gy in range(y_min, y_max + 1)
                if np.linalg.norm(np.array([gx, gy], dtype=float) - uav) <= max_sep + 1e-9
            ]
            if not nearby:
                continue
            ugv = np.array(nearby[self.rng.randint(0, len(nearby))], dtype=np.float64)
            if self.scene.is_uav_position_valid(uav) and self.scene.is_ugv_position_valid(ugv):
                return uav, ugv

        # Deterministic fallback independent of any hand-set start position.
        uav = np.array([self.Nx // 2, self.Ny // 2], dtype=np.float64)
        if not self.scene.is_uav_position_valid(uav):
            found_uav = False
            for gx in range(self.Nx):
                for gy in range(self.Ny):
                    cand = np.array([gx, gy], dtype=np.float64)
                    if self.scene.is_uav_position_valid(cand):
                        uav = cand
                        found_uav = True
                        break
                if found_uav:
                    break
            if not found_uav:
                raise RuntimeError("Failed to find a valid fallback UAV start position.")

        nearby = [
            np.array([gx, gy], dtype=np.float64)
            for gx in range(self.Nx)
            for gy in range(self.Ny)
            if self.scene.is_ugv_position_valid(np.array([gx, gy], dtype=np.float64))
            and np.linalg.norm(np.array([gx, gy], dtype=np.float64) - uav) <= max_sep + 1e-9
        ]
        if nearby:
            ugv = min(nearby, key=lambda pos: float(np.linalg.norm(pos - uav)))
            return uav, ugv

        if self.scene.is_ugv_position_valid(uav):
            return uav, uav.copy()

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
        self.pending_high_priority_samples: List[SpectrumSample] = []
        self.pending_low_priority_samples: List[SpectrumSample] = []
        self.pending_ensemble_sample_count = 0
        self.uav_info_level = 1
        self.last_preempted_low_packets = 0
        self.last_dropped_pending_low = 0
        self.last_sample_center_freq = -1
        self.map_update_count = 0
        self.planner_initialized = False
        self.active_target_nmse_record: Optional[Dict[str, object]] = None
        self.completed_target_nmse_records: List[Dict[str, object]] = []
        self.last_completed_target_nmse_record: Optional[Dict[str, object]] = None
        self.reconstruction_events: List[Dict[str, object]] = []
        self.ensemble_events: List[Dict[str, object]] = []
        self.bootstrap_events: List[Dict[str, object]] = []

        self.sampled_mask = np.zeros((self.Nx, self.Ny, self.K), dtype=bool)
        self.action_visit = np.zeros((self.Nx, self.Ny, self.K), dtype=float)
        self.local_spatial_visit = np.zeros((self.Nx, self.Ny), dtype=float)

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
        self.prev_nmse = float(self.radio_map_state.nmse)
        self.episode_nmse_start = float(self.radio_map_state.nmse)
        self._init_bootstrap_target()

        self.ugv_channel_info = self._get_channel_info()
        self.last_uav_heading = np.zeros(2, dtype=float)
        self.last_step_quantization_bits = int(self.high_quantization_bits)
        self.last_step_raw_bits = 0.0
        self.total_raw_sample_bits = 0.0
        self.recent_delta_nmse_norm = 0.0
        self.recent_delta_queue_norm = 0.0
        self.recent_delta_snr_norm = 0.0

        self._update_agent_info_levels()

        obs = self._build_observations()
        current_target = self._get_current_observation_target()
        info = {
            "nmse": self.radio_map_state.nmse,
            "target_count": len(self.planner_targets),
            "planner_initialized": 0,
            "target_grid_x": int(current_target.gx) if current_target is not None else -1,
            "target_grid_y": int(current_target.gy) if current_target is not None else -1,
            "target_center_freq": int(current_target.center_freq) if current_target is not None else -1,
            "target_quantization_bits": (
                int(current_target.quantization_bits) if current_target is not None else int(self.high_quantization_bits)
            ),
            "target_source": str(self._get_current_target_source()),
            "bootstrap_active": int(self._bootstrap_target_is_active()),
        }
        return obs, info

    def _reset_grid_plan_state(self) -> None:
        """Reset active single-grid planning state."""
        self.active_plan_grid: Optional[Tuple[int, int]] = None
        self.active_plan_center_freq: Optional[int] = None
        self.active_plan_quantization_bits: Optional[int] = None
        if hasattr(self, "local_spatial_visit"):
            self.local_spatial_visit.fill(0.0)

    def _build_bootstrap_target(self) -> Optional[PlannerTarget]:
        """Choose a valid pre-planner target over the full UAV-feasible map."""
        preferred_center_freq = int(np.clip(self.K // 2, 0, self.K - 1))
        best_target: Optional[PlannerTarget] = None
        best_score = -np.inf

        for gx, gy in self.grid_index_positions:
            pos = np.array([gx, gy], dtype=float)
            if not self.scene.is_uav_position_valid(pos):
                continue
            dist_uav = manhattan_distance(pos, self.uav_pos)
            if dist_uav < 1.0:
                continue
            edge_margin = float(min(gx, gy, self.Nx - 1 - gx, self.Ny - 1 - gy))
            score = dist_uav + (0.2 * edge_margin)
            if score > best_score + 1e-9:
                best_score = score
                best_target = PlannerTarget(
                    gx=int(gx),
                    gy=int(gy),
                    x=float(gx),
                    y=float(gy),
                    center_freq=preferred_center_freq,
                    score=float(score),
                    quantization_bits=int(self.high_quantization_bits),
                )
        return best_target

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
        if self.bootstrap_target is not None:
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
        return event

    def _retarget_bootstrap_phase(self, reason: str) -> Optional[Dict[str, object]]:
        self.bootstrap_target = self._build_bootstrap_target()
        self.bootstrap_target_reached_once = False
        if self.bootstrap_target is None:
            return None
        return self._record_bootstrap_event(
            event="bootstrap_retarget",
            reason=reason,
            target=self.bootstrap_target,
        )

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

    def _grid_has_unobserved_band(self, gx: int, gy: int) -> bool:
        return not bool(np.all(self.sampled_mask[int(gx), int(gy), :]))

    def _build_local_candidate_mask(
        self,
        center_pos: Optional[np.ndarray] = None,
        radius: Optional[int] = None,
    ) -> np.ndarray:
        if center_pos is None:
            center_pos = self.uav_pos
        center = np.asarray(center_pos, dtype=float).reshape(2)
        center = np.rint(np.clip(center, [0.0, 0.0], [self.Nx - 1, self.Ny - 1])).astype(int)
        radius = self.local_planner_radius if radius is None else max(0, int(radius))

        mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        x_min = max(0, int(center[0] - radius))
        x_max = min(self.Nx - 1, int(center[0] + radius))
        for gx in range(x_min, x_max + 1):
            remaining = radius - abs(gx - int(center[0]))
            y_min = max(0, int(center[1] - remaining))
            y_max = min(self.Ny - 1, int(center[1] + remaining))
            for gy in range(y_min, y_max + 1):
                pos = np.array([gx, gy], dtype=float)
                if self.scene.is_uav_position_valid(pos):
                    mask[gx, gy] = True
        return mask

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
                if self._grid_has_unobserved_band(cand.gx, cand.gy):
                    selected = cand
                    break

            if selected is not None:
                self.active_plan_grid = (int(selected.gx), int(selected.gy))
                self.active_plan_center_freq = self._select_center_freq_for_grid(
                    int(selected.gx),
                    int(selected.gy),
                    preferred_center_freq=int(selected.center_freq),
                )
                self.active_plan_quantization_bits = int(selected.quantization_bits)
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

    def _planner_ready(self) -> bool:
        return self._planner_sample_count() >= int(self.config.planner.min_samples_for_ensemble)

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
            "map_update_count": int(self.map_update_count),
            "target_grid_x": int(target.gx) if target is not None else -1,
            "target_grid_y": int(target.gy) if target is not None else -1,
            "target_center_freq": int(target.center_freq) if target is not None else -1,
        }
        history.append(event)
        return event

    def _record_reconstruction_event(
        self,
        reason: str,
        trigger_sample_count: int,
    ) -> Dict[str, object]:
        return self._record_nmse_event(
            event_type="reconstruction",
            reason=reason,
            trigger_sample_count=trigger_sample_count,
        )

    def _record_ensemble_event(
        self,
        reason: str,
        trigger_sample_count: int,
    ) -> Dict[str, object]:
        event = self._record_nmse_event(
            event_type="ensemble",
            reason=reason,
            trigger_sample_count=trigger_sample_count,
        )
        self.pending_ensemble_sample_count = 0
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
        self.radio_map_state = self.td.reconstruct()
        self.pending_ensemble_sample_count = 0
        self.map_update_count += 1
        self._record_active_target_nmse_update()

        if refresh_targets and self._planner_ready():
            ensemble_event = self._refresh_planner_outputs(
                seed_offset=seed_offset,
                force_ensemble=True,
                trigger_reason=reason,
            )
            if ensemble_event is not None:
                return ensemble_event

        self._sync_cached_ensemble_state()
        return self._record_ensemble_event(
            reason=reason,
            trigger_sample_count=trigger_sample_count,
        )

    def step(
        self,
        uav_move_action: int,
        uav_bw_action: int,
        ugv_action: int,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, dict]:
        self.current_step += 1

        prev_uav_pos = self.uav_pos.copy()
        prev_ugv_pos = self.ugv_pos.copy()
        prev_snr_norm = self._get_snr_norm()

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
            prev_ugv_target_dist = manhattan_distance(prev_ugv_pos, reward_target)
        else:
            prev_uav_target_dist = None
            prev_ugv_target_dist = None
        
        # Apply UAV movement and sensing/communication split from independent heads.
        self._apply_uav_actions(
            move_action=int(uav_move_action),
            bw_action=int(uav_bw_action),
        )

        # 2) Movement execution.
        self._move_ugv(int(ugv_action))
        self._update_agent_info_levels()
        uav_move_dist = float(np.linalg.norm(self.uav_pos - prev_uav_pos))
        ugv_move_dist = float(np.linalg.norm(self.ugv_pos - prev_ugv_pos))
        if self.uav_step_count > 0:
            self.last_uav_heading = (self.uav_pos - prev_uav_pos) / float(self.uav_step_count)
        else:
            self.last_uav_heading = self.uav_pos - prev_uav_pos
        if reward_target is not None and prev_uav_target_dist is not None:
            curr_uav_target_dist = manhattan_distance(self.uav_pos, reward_target)
            uav_progress = prev_uav_target_dist - curr_uav_target_dist
        else:
            uav_progress = 0.0
        if reward_target is not None and prev_ugv_target_dist is not None:
            curr_ugv_target_dist = manhattan_distance(self.ugv_pos, reward_target)
            ugv_progress = prev_ugv_target_dist - curr_ugv_target_dist
        else:
            ugv_progress = 0.0

        # 3) Channel update.
        self.ugv_channel_info = self._get_channel_info()
        curr_snr_norm = self._get_snr_norm()
        self.recent_delta_snr_norm = float(curr_snr_norm - prev_snr_norm)

        # 4) UAV samples the current grid point directly from GT with noise.
        sample, newly_sampled_freqs, newly_visited_spatial, sampling_stats = self._collect_current_grid_sample()
        data_produced_bits = 0.0
        self.last_preempted_low_packets = 0
        if sample is not None:
            self.last_preempted_low_packets = self._enqueue_sample_packet(
                sample,
                novelty_ratio=float(sampling_stats.get("novelty_ratio", 1.0)),
            )
            data_size = float(sample.payload_bits)
            data_produced_bits = data_size

        # 5) Transmission queue simulation.
        queue_bits_before_tx = self._queue_remaining_bits()
        delivered_packets, data_delivered_bits, novel_data_delivered_bits = self._simulate_transmission()
        self._schedule_delivered_samples(delivered_packets)
        dropped_packets = self._enforce_queue_capacity()
        queue_bits_after_tx = self._queue_remaining_bits()

        # 6) UGV ensemble-driven map/planner refresh.
        processed_high, processed_low, dropped_pending_low = self._process_pending_samples()

        map_updated = False
        reconstruction_event: Optional[Dict[str, object]] = None
        ensemble_event: Optional[Dict[str, object]] = None
        bootstrap_target_reached_event: Optional[Dict[str, object]] = None
        bootstrap_handoff_event: Optional[Dict[str, object]] = None
        target_reached = bool(
            step_target_grid is not None
            and int(np.rint(self.uav_pos[0])) == int(step_target_grid[0])
            and int(np.rint(self.uav_pos[1])) == int(step_target_grid[1])
        )
        bootstrap_target_reached = bool(step_target_source == "bootstrap" and target_reached)
        if bootstrap_target_reached and not self.bootstrap_target_reached_once:
            self.bootstrap_target_reached_once = True
            bootstrap_target_reached_event = self._record_bootstrap_event(
                event="bootstrap_target_reached",
                reason="target_reached",
                target=self.bootstrap_target,
            )
        completed_target_nmse_record: Optional[Dict[str, object]] = None

        # Warm up planner once enough packets have been delivered.
        if (not self.planner_initialized) and self._planner_ready():
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
            self._retarget_bootstrap_phase(reason="target_reached")
        elif self.planner_initialized:
            if target_reached:
                completed_target_nmse_record = self._finalize_active_target_nmse_record(target_reached=True)
                if self.pending_ensemble_sample_count > 0:
                    ensemble_event = self._run_ensemble_map_update(
                        reason="target_reached",
                        seed_offset=self.current_step,
                        refresh_targets=True,
                    )
                    map_updated = ensemble_event is not None
                else:
                    ensemble_event = self._refresh_planner_outputs(
                        seed_offset=self.current_step,
                        force_ensemble=True,
                        trigger_reason="target_reached",
                    )
                self._clear_active_plan()
                self._start_new_grid_plan()
            else:
                if self.pending_ensemble_sample_count >= self.ensemble_refresh_interval:
                    ensemble_event = self._run_ensemble_map_update(
                        reason="ensemble_interval",
                        seed_offset=self.current_step,
                        refresh_targets=True,
                    )
                    map_updated = ensemble_event is not None
                    if ensemble_event is not None:
                        self._clear_active_plan()
                        self._start_new_grid_plan()

        should_force_flush = (
            (not map_updated)
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
        reconstruction_event = ensemble_event if map_updated else None
        self.last_dropped_pending_low = int(dropped_pending_low)
        if self.planner_initialized and self.active_plan_grid is None and self.planner_targets:
            self._start_new_grid_plan()
        # 7) Rewards.
        rewards, reward_info = self._compute_reward(
            uav_move_dist=uav_move_dist,
            ugv_move_dist=ugv_move_dist,
            uav_progress=uav_progress,
            ugv_progress=ugv_progress,
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

        # 8) Termination.
        rc = self.config.reward
        terminated = False
        truncated = False
        terminal_success = bool(
            self.radio_map_state.last_update_step > 0
            and self.radio_map_state.nmse <= rc.accuracy_target_nmse
        )
        energy_depleted = bool(self.uav_energy <= 0)
        reached_step_limit = bool(self.current_step >= self.config.mappo.episode_max_steps)

        if terminal_success:
            terminated = True
        elif energy_depleted:
            terminated = True
        timed_out = bool((not terminated) and reached_step_limit)
        if timed_out:
            truncated = True

        terminal_reward = 0.0
        terminal_failure = False
        timeout_failure = False
        if terminal_success:
            terminal_reward = float(rc.terminal_success_bonus)
        elif energy_depleted:
            terminal_reward = -abs(float(rc.terminal_failure_penalty))
            terminal_failure = True
        elif timed_out:
            terminal_reward = -abs(float(rc.terminal_failure_penalty))
            terminal_failure = True
            timeout_failure = True

        if abs(terminal_reward) > 0.0:
            rewards["team_reward"] = float(rewards["team_reward"] + terminal_reward)
            rewards["uav_reward"] = float(rewards["uav_reward"] + terminal_reward)
            rewards["ugv_reward"] = float(rewards["ugv_reward"] + terminal_reward)
        reward_info["r_terminal"] = float(terminal_reward)
        reward_info["terminal_success"] = int(terminal_success)
        reward_info["terminal_failure"] = int(terminal_failure)
        reward_info["timeout_failure"] = int(timeout_failure)
        reward_info["timed_out"] = int(timed_out)
        reward_info["team_reward"] = float(rewards["team_reward"])
        reward_info["uav_reward"] = float(rewards["uav_reward"])
        reward_info["ugv_reward"] = float(rewards["ugv_reward"])
        bootstrap_event_labels: List[str] = []
        if bootstrap_target_reached_event is not None:
            bootstrap_event_labels.append(str(bootstrap_target_reached_event["event"]))
        if bootstrap_handoff_event is not None:
            bootstrap_event_labels.append(str(bootstrap_handoff_event["event"]))

        # 9) Observation.
        obs = self._build_observations()
        info = {
            **reward_info,
            "nmse": self.radio_map_state.nmse,
            "channel_capacity": self.ugv_channel_info.capacity_bps,
            "channel_los": self.ugv_channel_info.los,
            "snr_db": self.ugv_channel_info.snr_db,
            "snr_norm": self._get_snr_norm(),
            "bw_ratio": self.current_bw_ratio,
            "sensing_ind": int(self.last_sample_center_freq),
            "sample_center_freq": int(self.last_sample_center_freq),
            "uav_energy": self.uav_energy,
            "queue_size": len(self.uav_data_queue),
            "queue_bits_norm_obs": self._get_queue_bits_norm(),
            "total_samples": self.total_collected_samples,
            "step": self.current_step,
            "target_grid_x": int(step_target_grid[0]) if step_target_grid is not None else -1,
            "target_grid_y": int(step_target_grid[1]) if step_target_grid is not None else -1,
            "target_center_freq": int(step_target_center_freq),
            "target_quantization_bits": (
                int(step_target.quantization_bits) if step_target is not None else int(self.high_quantization_bits)
            ),
            "target_freq": int(step_target_center_freq),
            "target_source": str(step_target_source),
            "target_count": len(self.planner_targets),
            "map_updated": int(map_updated),
            "planner_initialized": int(self.planner_initialized),
            "planner_sample_count": int(self._planner_sample_count()),
            "target_reached": int(target_reached),
            "bootstrap_active": int(step_target_source == "bootstrap"),
            "bootstrap_target_reached": int(bootstrap_target_reached),
            "bootstrap_handoff": int(bootstrap_handoff_event is not None),
            "bootstrap_event": "|".join(bootstrap_event_labels),
            "reconstruction_triggered": int(reconstruction_event is not None),
            "reconstruction_reason": (
                str(reconstruction_event["reason"])
                if reconstruction_event is not None
                else ""
            ),
            "reconstruction_event_nmse": (
                float(reconstruction_event["nmse"])
                if reconstruction_event is not None
                else float(self.radio_map_state.nmse)
            ),
            "reconstruction_event_nmse_delta": (
                float(reconstruction_event["nmse_delta"])
                if reconstruction_event is not None
                else 0.0
            ),
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
            # Compatibility alias: map updates are now ensemble-driven.
            "pending_reconstruct_sample_count": int(self.pending_ensemble_sample_count),
            "pending_ensemble_sample_count": int(self.pending_ensemble_sample_count),
            "sensing_band_num": int(self.sensing_band_num),
            "sensing_bw_units": int(self.current_sensing_units),
            "comm_bw_units": int(self.current_comm_units),
            "step_quantization_bits": int(self.last_step_quantization_bits),
            "total_bits_norm": float(self._get_total_bits_norm()),
            "bits_budget_remaining_norm": float(self._get_bits_budget_remaining_norm()),
            "comm_capacity_norm": float(self._get_comm_capacity_norm()),
            "info_level": int(self.uav_info_level),
            "uav_info_level": int(self.uav_info_level),
            "processed_high": int(processed_high),
            "processed_low": int(processed_low),
            "preempted_low_packets": int(self.last_preempted_low_packets),
            "dropped_pending_low": int(self.last_dropped_pending_low),
            "pending_high_queue": int(len(self.pending_high_priority_samples)),
            "pending_low_queue": int(len(self.pending_low_priority_samples)),
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
        }
        return obs, rewards, terminated, truncated, info

    def _set_bandwidth_info(self, ratio: float) -> None:
        total_units = max(1, int(self.config.uav.total_bw_num))
        if total_units == 1:
            sensing_units = 1
            comm_units = 0
        else:
            sensing_units = int(np.clip(np.round(total_units * float(ratio)), 1, total_units - 1))
            comm_units = total_units - sensing_units

        self.current_sensing_units = int(sensing_units)
        self.current_comm_units = int(comm_units)
        self.current_bw_ratio = ratio
        self.sensing_band_num = int(np.clip(np.round(self.K * self.current_bw_ratio), 1, self.K))

    def _decode_uav_move_action(self, action: int) -> int:
        action = int(np.clip(action, 0, self.uav_move_action_size - 1))
        return int(self.uav_direction_ids[action])

    def _decode_uav_bw_action(self, action: int) -> int:
        action = int(np.clip(action, 0, self.uav_bw_action_size - 1))
        return action

    def _encode_uav_action(self, move_action: int, bw_action: int) -> int:
        move_idx = int(np.clip(move_action, 0, self.uav_move_action_size - 1))
        bw_idx = int(np.clip(bw_action, 0, self.uav_bw_action_size - 1))
        return int(move_idx * self.uav_bw_action_size + bw_idx)

    def _decode_uav_action(self, action: int) -> Tuple[int, int]:
        action = int(np.clip(action, 0, (self.uav_move_action_size * self.uav_bw_action_size) - 1))
        direction_choice_idx = action // self.uav_bw_action_size
        bw_choice_idx = action % self.uav_bw_action_size
        direction_idx = int(self.uav_direction_ids[direction_choice_idx])
        return direction_idx, bw_choice_idx

    def _apply_uav_actions(self, move_action: int, bw_action: int) -> None:
        direction_idx = self._decode_uav_move_action(move_action)
        bw_choice_idx = self._decode_uav_bw_action(bw_action)
        self._set_bandwidth_info(float(self.bandwidth_ratios[bw_choice_idx]))
        self._move_uav(direction_idx=direction_idx)

    def _grid_step_count(self, step_size: float) -> int:
        grid_spacing = max(float(self.config.scene.grid_spacing), 1e-9)
        return max(1, int(round(float(step_size) / grid_spacing)))

    def _can_follow_direction(
        self,
        position: np.ndarray,
        direction_idx: int,
        step_count: int,
        validator,
        stop_at_target: bool = False,
    ) -> bool:
        """Return whether a direction would produce any valid movement from position."""
        offset = DIRECTION_OFFSETS.get(int(direction_idx), np.array([0, 0], dtype=float)).astype(float)
        if np.allclose(offset, 0.0):
            return True

        new_pos = np.asarray(position, dtype=float).copy()
        target_grid = self._get_motion_target_grid() if stop_at_target else None
        moved = False
        for _ in range(step_count):
            proposed = new_pos + offset
            if not validator(proposed):
                break
            new_pos = proposed
            moved = True
            if (
                target_grid is not None
                and int(np.rint(new_pos[0])) == int(target_grid[0])
                and int(np.rint(new_pos[1])) == int(target_grid[1])
            ):
                break
        return moved

    def _move_uav(self, direction_idx: int) -> float:
        old_pos = self.uav_pos.copy()
        offset = DIRECTION_OFFSETS.get(int(direction_idx), np.array([0, 0], dtype=float)).astype(float)
        new_pos = old_pos.copy()
        step_count = self.uav_step_count
        target_grid = self._get_motion_target_grid()

        for _ in range(step_count):
            proposed = new_pos + offset
            if not self.scene.is_uav_position_valid(proposed):
                break
            new_pos = proposed

            # If the UAV passes through the active uncertainty target, stop there
            # so the subsequent sensing step samples that target grid immediately.
            if (
                target_grid is not None
                and int(np.rint(new_pos[0])) == int(target_grid[0])
                and int(np.rint(new_pos[1])) == int(target_grid[1])
            ):
                break

        self.uav_pos = np.rint(np.clip(new_pos, [0.0, 0.0], [self.Nx - 1, self.Ny - 1]))
        move_dist = float(np.linalg.norm(self.uav_pos - old_pos))
        if move_dist > 1e-9:
            energy = self.config.uav.flight_power * self.config.uav.step_duration * move_dist
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
        info = self.sim_data.get_channel_info(
            uav_position=self.uav_pos,
            ugv_position=self.ugv_pos,
            bandwidth_comm=comm_bw,
            tx_power_dbm=self.config.comm.tx_power_dbm,
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

    def _estimate_reference_comm_capacity(self) -> float:
        """Best-case link capacity used to normalize communication quality."""
        max_comm_bw = float(self.config.uav.total_bandwidth) * 0.8
        if max_comm_bw <= 0.0:
            return 1.0

        d_3d = max(float(self.config.scene.uav_height), 1.0)
        fspl = 20 * np.log10(d_3d) + 20 * np.log10(float(self.config.comm.carrier_freq)) - 147.55
        noise_power_dbm = (
            -174
            + 10 * np.log10(max_comm_bw + 1e-3)
            + float(self.config.comm.noise_figure_db)
        )
        snr_linear = 10 ** ((float(self.config.comm.tx_power_dbm) - fspl - noise_power_dbm) / 10.0)
        capacity = max_comm_bw * np.log2(1.0 + max(snr_linear, 0.0))
        return float(max(capacity, 1.0))

    def _get_snr_norm(self) -> float:
        return float(np.tanh(self.ugv_channel_info.snr_db / self.snr_norm_den))

    def _get_comm_capacity_norm(self) -> float:
        return float(
            np.clip(
                self.ugv_channel_info.capacity_bps / max(self.max_comm_capacity_bps, 1e-8),
                0.0,
                1.0,
            )
        )

    def _get_queue_bits_norm(self) -> float:
        return float(np.clip(self._queue_remaining_bits() / (self.q_max_bits + 1e-8), 0.0, 1.0))

    def _get_steps_remaining_norm(self) -> float:
        remaining = max(0, int(self.config.mappo.episode_max_steps) - int(self.current_step))
        return float(remaining / max(1, int(self.config.mappo.episode_max_steps)))

    def _get_total_bits_norm(self) -> float:
        return float(
            np.clip(self.total_raw_sample_bits / max(self.raw_episode_bit_budget, 1e-8), 0.0, 1.0)
        )

    def _get_bits_budget_remaining_norm(self) -> float:
        return float(np.clip(1.0 - self._get_total_bits_norm(), 0.0, 1.0))

    def _compute_quadrant_features(
        self,
        center_pos: np.ndarray,
    ) -> np.ndarray:
        x_idx = int(np.clip(np.round(center_pos[0]), 0, self.Nx - 1))
        y_idx = int(np.clip(np.round(center_pos[1]), 0, self.Ny - 1))
        spatial_unc = normalize_score_map(self.uncertainty.spatial_uncertainty)
        visited_map = np.any(self.sampled_mask, axis=2).astype(float)

        quadrants = {
            "NE": (slice(x_idx, self.Nx), slice(y_idx, self.Ny)),
            "NW": (slice(0, x_idx), slice(y_idx, self.Ny)),
            "SE": (slice(x_idx, self.Nx), slice(0, y_idx)),
            "SW": (slice(0, x_idx), slice(0, y_idx)),
        }

        unc_feats: List[float] = []
        vis_feats: List[float] = []
        for key in ("NE", "NW", "SE", "SW"):
            sl_x, sl_y = quadrants[key]
            patch_unc = spatial_unc[sl_x, sl_y]
            patch_vis = visited_map[sl_x, sl_y]
            if patch_unc.size == 0:
                unc_feats.append(0.0)
                vis_feats.append(1.0)
            else:
                unc_feats.append(float(np.mean(patch_unc)))
                vis_feats.append(float(np.mean(patch_vis > 0.0)))
        return np.asarray([*unc_feats, *vis_feats], dtype=float)

    def _select_target_quantization_bits(
        self,
        gx: int,
        gy: int,
        phase: str = "planned_path",
    ) -> int:
        spatial_unc = normalize_score_map(self.uncertainty.spatial_uncertainty)
        uncertainty_norm = float(spatial_unc[int(gx), int(gy)])
        q_bits, _ = select_quantization_bits_from_uncertainty(
            uncertainty_norm,
            adaptive_quantization_bits=bool(self.config.planner.adaptive_quantization_bits),
            quantization_bits=int(self.default_quantization_bits),
            high_quantization_bits=int(self.high_quantization_bits),
            low_quantization_bits=int(self.low_quantization_bits),
            uncertainty_quantization_threshold=float(
                self.config.planner.uncertainty_quantization_threshold
            ),
            phase=str(phase),
        )
        return int(q_bits)

    def _queue_remaining_bits(self) -> float:
        return float(sum(max(p.size_bits - p.transmitted_bits, 0.0) for p in self.uav_data_queue))

    def _enqueue_sample_packet(self, sample: SpectrumSample, novelty_ratio: float = 1.0) -> int:
        data_size = float(max(sample.payload_bits, 0.0))
        priority = int(self.uav_info_level)

        preempted_low = 0
        if priority >= 2:
            kept_packets: List[DataPacket] = []
            for packet in self.uav_data_queue:
                if int(packet.priority) < 2 and packet.transmitted_bits <= 0.0:
                    preempted_low += 1
                    continue
                kept_packets.append(packet)
            self.uav_data_queue = kept_packets

        self.uav_data_queue.append(
            DataPacket(
                sample=sample,
                size_bits=data_size,
                priority=priority,
                created_step=int(self.current_step),
                novelty_ratio=float(np.clip(novelty_ratio, 0.0, 1.0)),
            )
        )
        return int(preempted_low)

    def _enforce_queue_capacity(self) -> int:
        max_q = int(max(1, self.queue_ref))
        curr_q = len(self.uav_data_queue)
        if curr_q <= max_q:
            return 0

        dropped = curr_q - max_q
        drop_order = sorted(
            range(len(self.uav_data_queue)),
            key=lambda idx: (
                int(self.uav_data_queue[idx].priority),
                int(self.uav_data_queue[idx].created_step),
            ),
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

        high_indices = [
            idx for idx, packet in enumerate(self.uav_data_queue)
            if int(packet.priority) >= 2
        ]
        if high_indices:
            return min(
                high_indices,
                key=lambda idx: (
                    int(self.uav_data_queue[idx].created_step),
                    -float(self.uav_data_queue[idx].transmitted_bits),
                ),
            )

        return min(
            range(len(self.uav_data_queue)),
            key=lambda idx: (
                int(self.uav_data_queue[idx].created_step),
                float(self.uav_data_queue[idx].transmitted_bits),
            ),
        )

    def _simulate_transmission(self) -> Tuple[List[DataPacket], float, float]:
        if not self.uav_data_queue:
            return [], 0.0, 0.0

        available_bits = self.ugv_channel_info.capacity_bps * self.config.uav.step_duration
        delivered: List[DataPacket] = []
        transmitted_bits = 0.0
        novel_transmitted_bits = 0.0
        while self.uav_data_queue and available_bits > 0:
            best_idx = self._select_next_packet_index()
            if best_idx < 0:
                break
            packet = self.uav_data_queue[best_idx]

            # 检查数据包中还有多少数据没有传输
            remaining = max(packet.size_bits - packet.transmitted_bits, 0.0)
            
            # 如果都传输完成，则将原来的数据包弹出uav_data_queue
            if remaining <= 0:
                self.uav_data_queue.pop(best_idx)
                delivered.append(packet)
                continue
            
            # 如果没有传输完成 则查看应该发送多少数据
            sent_bits = min(available_bits, remaining)
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

    def _schedule_delivered_samples(self, delivered_packets: List[DataPacket]) -> None:
        for packet in delivered_packets:
            if int(packet.priority) >= 2:
                self.pending_high_priority_samples.append(packet.sample)
            else:
                self.pending_low_priority_samples.append(packet.sample)

    def _pending_processing_pressure(self) -> float:
        total_pending = len(self.pending_high_priority_samples) + len(self.pending_low_priority_samples)
        return float(total_pending / max(1.0, float(self.queue_ref)))

    def _trim_pending_low_priority(self, max_keep: int) -> int:
        max_keep = max(0, int(max_keep))
        low_count = len(self.pending_low_priority_samples)
        if low_count <= max_keep:
            return 0

        dropped = low_count - max_keep
        if max_keep > 0:
            self.pending_low_priority_samples = self.pending_low_priority_samples[-max_keep:]
        else:
            self.pending_low_priority_samples = []
        return int(dropped)

    def _process_pending_samples(self) -> Tuple[int, int, int]:
        processed: List[SpectrumSample] = []
        dropped_low = 0

        processed_high = len(self.pending_high_priority_samples)
        if processed_high > 0:
            pending_pressure = self._pending_processing_pressure()
            if pending_pressure > 1.0 and self.pending_low_priority_samples:
                keep_low = int(np.ceil(0.5 * max(1, self.queue_ref)))
                dropped_low = self._trim_pending_low_priority(keep_low)
            processed.extend(self.pending_high_priority_samples)
            self.pending_high_priority_samples = []

        processed_low = 0
        low_count = len(self.pending_low_priority_samples)
        pending_pressure = self._pending_processing_pressure()
        low_backlog_heavy = low_count >= max(2, 2 * self.low_priority_process_interval)
        low_pressure_small = pending_pressure <= 0.5

        if processed_high == 0 and low_count > 0:
            if not (low_backlog_heavy and low_pressure_small):
                if pending_pressure > 1.0:
                    batch_size = min(low_count, max(1, int(np.ceil(0.5 * self.queue_ref))))
                elif self.current_step % self.low_priority_process_interval == 0:
                    batch_size = min(low_count, max(1, self.low_priority_process_interval))
                else:
                    batch_size = 0

                if batch_size > 0:
                    processed.extend(self.pending_low_priority_samples[:batch_size])
                    self.pending_low_priority_samples = self.pending_low_priority_samples[batch_size:]
                    processed_low = int(batch_size)

        if not processed:
            return 0, 0, int(dropped_low)

        processed_count = len(processed)
        self.delivered_samples.extend(processed)
        self.td.add_samples(processed)
        self.pending_ensemble_sample_count += processed_count
        return processed_high, processed_low, int(dropped_low)

    def _is_in_designated_region(self, gx: int, gy: int) -> bool:
        target_grid = self._get_motion_target_grid()
        return target_grid is not None and int(gx) == int(target_grid[0]) and int(gy) == int(target_grid[1])

    def _get_position_info_level(self, position: np.ndarray) -> int:
        gx = int(np.clip(np.round(position[0]), 0, self.Nx - 1))
        gy = int(np.clip(np.round(position[1]), 0, self.Ny - 1))
        return 2 if self._is_in_designated_region(gx, gy) else 1

    def _update_agent_info_levels(self) -> None:
        self.uav_info_level = int(self._get_position_info_level(self.uav_pos))

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
        current_target = self._get_current_observation_target()

        center_freq = self._pick_center_freq(ux, uy)
        self.last_sample_center_freq = int(center_freq)
        if self.active_plan_grid is not None and self._is_in_designated_region(ux, uy):
            self.active_plan_center_freq = int(center_freq)
        if current_target is not None:
            quantization_bits = int(current_target.quantization_bits)
        else:
            quantization_bits = int(self.high_quantization_bits)
        self.last_step_quantization_bits = int(quantization_bits)

        sample, newly_sampled_freqs, newly_visited_spatial, sampling_stats = self._collect_grid_sample(
            position=np.array([ux, uy], dtype=float),
            center_freq=center_freq,
            quantization_bits=int(quantization_bits),
        )
        return sample, newly_sampled_freqs, newly_visited_spatial, sampling_stats

    def _collect_grid_sample(
        self,
        position: np.ndarray,
        center_freq: int,
        quantization_bits: int,
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
        raw_step_bits = float(observed_band_count) * float(max(1, int(quantization_bits)))
        payload_bits = (
            float(observed_band_count)
            * (float(max(1, int(quantization_bits))) / float(max(1, self.high_quantization_bits)))
            * float(self.config.comm.data_per_sample)
        )
        sampling_stats = {
            "spatial_revisit_count": prior_spatial_visits,
            "observed_band_count": float(observed_band_count),
            "novelty_ratio": novelty_ratio,
            "repeat_ratio": 1.0 - novelty_ratio,
            "quantization_bits": float(max(1, int(quantization_bits))),
            "raw_step_bits": float(raw_step_bits),
            "payload_bits": float(payload_bits),
        }

        gamma_full = self.sim_data.get_data_at_newpos(
            position=loc,
            add_noise=True,
            quantized=True,
            quantization_bits=int(quantization_bits),
        ).astype(float)
        gamma_sparse = gamma_full * omega.astype(float)
        measurements = gamma_full[observed_bands]
        self.sampled_mask[gx, gy, observed_bands] = True
        newly_visited_spatial = (not was_visited) and (newly_sampled_freqs > 0)

        self.action_visit[gx, gy, observed_bands] += 1.0
        self.local_spatial_visit[gx, gy] += 1.0
        self.total_collected_samples += 1
        self.last_step_raw_bits = float(raw_step_bits)
        self.total_raw_sample_bits += float(raw_step_bits)

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
            quantization_bits=int(max(1, int(quantization_bits))),
            payload_bits=float(payload_bits),
        )
        return sample, newly_sampled_freqs, newly_visited_spatial, sampling_stats

    def _refresh_planner_outputs(
        self,
        seed_offset: int,
        force_ensemble: bool = False,
        trigger_reason: str = "",
    ) -> Optional[Dict[str, object]]:
        min_samples = int(self.config.planner.min_samples_for_ensemble)
        available_samples = self._planner_sample_count()
        ensemble_event: Optional[Dict[str, object]] = None

        if available_samples < min_samples:
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
                    expected_sample_count=available_samples,
                )
            if cached_outputs is not None:
                mean_map, var_map = cached_outputs
            else:
                obs_locs = np.asarray([s.position for s in self.delivered_samples], dtype=float)
                gamma = np.asarray([s.gamma for s in self.delivered_samples], dtype=float)
                omega = np.asarray([s.omega for s in self.delivered_samples], dtype=np.int32)

                keep_ratio_t = adaptive_keep_ratio(
                    available_samples,
                    early_ratio=self.config.planner.ensemble_keep_ratio,
                    late_ratio=self.config.planner.ensemble_keep_ratio - 0.2,
                    switch_M=30,
                )

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
                    keep_ratio=keep_ratio_t,
                    keep_recent=self.config.planner.ensemble_keep_recent,
                    seed=self.config.mappo.seed + int(seed_offset),
                    base_model=self.td.get_btd_model() if hasattr(self.td, "get_btd_model") else None,
                    quality_weighted=True,
                    solver_backend=self.config.planner.iibtd_backend,
                    solver_device=(
                        self.config.mappo.device
                        if str(self.config.planner.iibtd_device).strip().lower() == "auto"
                        else self.config.planner.iibtd_device
                    ),
                    gpu_phi_solver=self.config.planner.iibtd_gpu_phi_solver,
                )
            self.latest_mean_map = mean_map
            self.latest_var_map = np.maximum(var_map, 0.0)

            self.uncertainty = UncertaintyMap(
                spatial_uncertainty=np.mean(self.latest_var_map, axis=2),
                frequency_uncertainty=np.mean(self.latest_var_map, axis=(0, 1)),
                joint_uncertainty=self.latest_var_map.copy(),
            )
            ensemble_event = self._record_ensemble_event(
                reason=trigger_reason or "forced_refresh",
                trigger_sample_count=int(self.pending_ensemble_sample_count),
            )

        acquisition_space, _ = build_acquisition_space(
            var_map=self.latest_var_map,
            lambda_u=self.config.planner.lambda_u,
        )
        local_candidate_mask = self._build_local_candidate_mask(center_pos=self.uav_pos)

        target_dicts = select_top_k_grid_candidates(
            acquisition_space=acquisition_space,
            var_map=self.latest_var_map,
            sampled_mask=self.sampled_mask,
            action_visit=self.action_visit,
            top_k=max(1, self.target_count),
            beta_f=self.config.planner.beta_f,
            candidate_mask=local_candidate_mask,
        )

        if not target_dicts:
            self.planner_targets = self._build_fallback_targets(candidate_mask=local_candidate_mask)
            return ensemble_event

        self.planner_targets = [
            PlannerTarget(
                gx=int(target["gx"]),
                gy=int(target["gy"]),
                x=float(target["x"]),
                y=float(target["y"]),
                center_freq=int(target["center_freq"]),
                score=float(target["score"]),
                quantization_bits=self._select_target_quantization_bits(
                    int(target["gx"]),
                    int(target["gy"]),
                ),
            )
            for target in target_dicts
        ]
        return ensemble_event

    def _build_fallback_targets(
        self,
        candidate_mask: Optional[np.ndarray] = None,
    ) -> List[PlannerTarget]:
        if candidate_mask is None:
            candidate_mask = np.ones((self.Nx, self.Ny), dtype=bool)
        candidate_mask = np.asarray(candidate_mask, dtype=bool)

        remaining_grid = np.argwhere(candidate_mask & ~np.any(self.sampled_mask, axis=2))
        if remaining_grid.size == 0:
            remaining_grid = np.argwhere(candidate_mask & ~np.all(self.sampled_mask, axis=2))
        if remaining_grid.size == 0:
            remaining_grid = np.argwhere(candidate_mask)
        if remaining_grid.size == 0:
            remaining_grid = self.grid_index_positions.copy()

        dists = np.linalg.norm(remaining_grid.astype(float) - self.uav_pos[np.newaxis, :], axis=1)
        top_k = min(max(1, self.target_count), remaining_grid.shape[0])
        top_indices = np.argsort(dists)[:top_k]
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
                quantization_bits=self._select_target_quantization_bits(int(gx), int(gy)),
            ))
        return targets

    def _get_reward_target_position(self) -> Optional[np.ndarray]:
        """Return the current planning target position used by reward shaping."""
        target = self._get_current_observation_target()
        if target is not None:
            return np.array([float(target.gx), float(target.gy)], dtype=float)
        return None

    def _get_current_observation_target(self) -> Optional[PlannerTarget]:
        """Return the active target used consistently across agent observations."""
        if self.active_plan_grid is not None:
            gx, gy = int(self.active_plan_grid[0]), int(self.active_plan_grid[1])
            center_freq = int(self.active_plan_center_freq) if self.active_plan_center_freq is not None else 0
            quantization_bits = (
                int(self.active_plan_quantization_bits)
                if self.active_plan_quantization_bits is not None
                else int(self.high_quantization_bits)
            )
            score = 0.0
            for cand in self.planner_targets:
                if int(cand.gx) == gx and int(cand.gy) == gy:
                    if self.active_plan_center_freq is None:
                        center_freq = int(cand.center_freq)
                    if self.active_plan_quantization_bits is None:
                        quantization_bits = int(cand.quantization_bits)
                    score = float(cand.score)
                    break
            return PlannerTarget(
                gx=gx,
                gy=gy,
                x=float(gx),
                y=float(gy),
                center_freq=center_freq,
                score=score,
                quantization_bits=quantization_bits,
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
                quantization_bits=int(cand.quantization_bits),
            )
        if self._bootstrap_target_is_active():
            return PlannerTarget(
                gx=int(self.bootstrap_target.gx),
                gy=int(self.bootstrap_target.gy),
                x=float(self.bootstrap_target.x),
                y=float(self.bootstrap_target.y),
                center_freq=int(self.bootstrap_target.center_freq),
                score=float(self.bootstrap_target.score),
                quantization_bits=int(self.bootstrap_target.quantization_bits),
            )
        return None

    def _compute_reward(
        self,
        uav_move_dist: float,
        ugv_move_dist: float,
        uav_progress: float,
        ugv_progress: float,
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

        delta_unc = 0.0
        delta_unc_norm = 0.0
        self.recent_delta_nmse_norm = float(delta_nmse_norm)
        observed_band_count = max(float(sampling_stats.get("observed_band_count", 1.0)), 1.0)
        raw_step_bits = float(sampling_stats.get("raw_step_bits", 0.0))
        step_bits_norm = float(np.clip(raw_step_bits / max(self.max_raw_step_bits, 1e-8), 0.0, 1.0))
        r_nmse = float(rc.w_nmse) * max(delta_nmse_norm, 0.0)
        r_tx = -float(rc.w_tx) * step_bits_norm

        queue_bits_before_norm = queue_bits_before_tx / (self.q_max_bits + 1e-8)
        queue_bits_norm = queue_bits_after_tx / (self.q_max_bits + 1e-8)
        curr_q = len(self.uav_data_queue)
        dropped_norm = float(dropped_packets) / float(self.max_q)
        r_queue = -float(rc.w_queue) * float(queue_bits_norm + dropped_norm)
        self.recent_delta_queue_norm = float(queue_bits_before_norm - queue_bits_norm)

        comm_quality_norm = self._get_comm_capacity_norm()
        r_comm = -float(rc.w_comm) * float(1.0 - comm_quality_norm)

        if queue_bits_before_tx > 0:
            tx_throughput = data_delivered_bits / (queue_bits_before_tx + 1e-8)
            tx_novelty_throughput = novel_data_delivered_bits / (queue_bits_before_tx + 1e-8)
        else:
            tx_throughput = 0.0
            tx_novelty_throughput = 0.0

        r_unc = 0.0
        r_new_freq = 0.0
        r_new_spatial = 0.0
        progress_den = max(float(self.local_planner_radius), 1e-8)
        uav_progress_norm = float(uav_progress) / progress_den
        ugv_progress_norm = float(ugv_progress) / progress_den
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
            (float(rc.lambda_uav_progress) * uav_forward)
            - (float(rc.lambda_uav_backtrack) * uav_backward)
        )
        r_ugv_progress = progress_scale * (
            (float(rc.lambda_ugv_progress) * ugv_forward)
            - (float(rc.lambda_ugv_backtrack) * ugv_backward)
        )
        r_goal_arrival = progress_scale * float(rc.local_goal_arrival_bonus) * float(
            int(target_reached and reward_target_grid is not None)
        )
        r_progress = r_uav_progress + r_ugv_progress + r_goal_arrival
        r_revisit = 0.0
        spatial_revisit_count = max(float(sampling_stats.get("spatial_revisit_count", 0.0)), 0.0)
        repeat_ratio = float(sampling_stats.get("repeat_ratio", 0.0))
        novelty_ratio = float(sampling_stats.get("novelty_ratio", 0.0))

        shared_reward = r_nmse + r_tx + r_queue + r_comm + r_progress
        uav_reward = shared_reward
        ugv_reward = shared_reward

        queue_norm = curr_q / float(self.max_q)

        rewards = {
            "team_reward": float(shared_reward),
            "uav_reward": float(uav_reward),
            "ugv_reward": float(ugv_reward),
        }
        info = {
            "r_nmse": float(r_nmse),
            "r_unc": float(r_unc),
            "r_new_freq": float(r_new_freq),
            "r_new_spatial": float(r_new_spatial),
            "r_tx": float(r_tx),
            "r_queue": float(r_queue),
            "r_comm": float(r_comm),
            "r_progress": float(r_progress),
            "r_revisit": float(r_revisit),
            "r_uav_progress": float(r_uav_progress),
            "r_ugv_progress": float(r_ugv_progress),
            "r_goal_arrival": float(r_goal_arrival),
            "tx_throughput": float(tx_throughput),
            "tx_novelty_throughput": float(tx_novelty_throughput),
            "queue_bits_before_tx": float(queue_bits_before_tx),
            "queue_bits_after_tx": float(queue_bits_after_tx),
            "queue_bits_before_norm": float(queue_bits_before_norm),
            "queue_bits_norm": float(queue_bits_norm),
            "data_produced_bits": float(data_produced_bits),
            "data_delivered_bits": float(data_delivered_bits),
            "novel_data_delivered_bits": float(novel_data_delivered_bits),
            "team_reward": float(shared_reward),
            "uav_reward": float(uav_reward),
            "ugv_reward": float(ugv_reward),
            "delta_nmse": float(delta_nmse),
            "delta_nmse_norm": float(delta_nmse_norm),
            "delta_unc": float(delta_unc),
            "delta_unc_norm": float(delta_unc_norm),
            "global_unc_mean": float(curr_global_unc),
            "local_goal_radius": int(self.local_planner_radius),
            "queue_norm": float(queue_norm),
            "dropped": int(dropped_packets),
            "dropped_norm": float(dropped_norm),
            "step_bits": float(raw_step_bits),
            "step_bits_norm": float(step_bits_norm),
            "step_quantization_bits": int(self.last_step_quantization_bits),
            "comm_quality_norm": float(comm_quality_norm),
            "uav_move_dist": float(uav_move_dist),
            "ugv_move_dist": float(ugv_move_dist),
            "uav_progress": float(uav_progress),
            "uav_progress_norm": 0.0,
            "ugv_progress": float(ugv_progress),
            "ugv_progress_norm": 0.0,
            "target_source": str(target_source),
            "progress_scale": 0.0,
            "progress_metric": "disabled",
            "observed_band_count": float(observed_band_count),
            "spatial_revisit_count": float(spatial_revisit_count),
            "sample_novelty_ratio": float(novelty_ratio),
            "sample_repeat_ratio": float(repeat_ratio),
            "newly_sampled_freqs": int(newly_sampled_freqs),
            "newly_visited_spatial": int(newly_visited_spatial),
            "r_terminal": 0.0,
            "terminal_success": 0,
            "terminal_failure": 0,
        }
        return rewards, info

    def _build_observations(self) -> Dict[str, np.ndarray]:
        current_target = self._get_current_observation_target()
        snr_norm = self._get_snr_norm()
        queue_norm = min(len(self.uav_data_queue) / float(self.queue_ref), 1.0)
        queue_bits_norm = self._get_queue_bits_norm()
        uav_energy_norm = float(self.uav_energy / self.uav_energy_den)
        uav_info_level_norm = float(self.uav_info_level / self.uav_info_level_den)
        comm_capacity_norm = self._get_comm_capacity_norm()
        steps_remaining_norm = self._get_steps_remaining_norm()
        bits_budget_remaining_norm = self._get_bits_budget_remaining_norm()
        return {
            "uav_obs": self._build_uav_obs(
                current_target=current_target,
                queue_norm=queue_norm,
                uav_energy_norm=uav_energy_norm,
                uav_info_level_norm=uav_info_level_norm,
                snr_norm=snr_norm,
                comm_capacity_norm=comm_capacity_norm,
                steps_remaining_norm=steps_remaining_norm,
                bits_budget_remaining_norm=bits_budget_remaining_norm,
            ),
            "uav_move_action_mask": self._build_uav_move_action_mask(),
            "uav_bw_action_mask": self._build_uav_bw_action_mask(),
            "ugv_obs": self._build_ugv_obs(
                current_target=current_target,
                snr_norm=snr_norm,
                uav_info_level_norm=uav_info_level_norm,
                queue_bits_norm=queue_bits_norm,
            ),
            "ugv_action_mask": self._build_ugv_action_mask(),
            "critic_state": self._build_critic_state(
                current_target=current_target,
                snr_norm=snr_norm,
                queue_norm=queue_norm,
                uav_energy_norm=uav_energy_norm,
                uav_info_level_norm=uav_info_level_norm,
            ),
        }

    def _build_uav_obs(
        self,
        current_target: Optional[PlannerTarget] = None,
        queue_norm: Optional[float] = None,
        uav_energy_norm: Optional[float] = None,
        uav_info_level_norm: Optional[float] = None,
        snr_norm: Optional[float] = None,
        comm_capacity_norm: Optional[float] = None,
        steps_remaining_norm: Optional[float] = None,
        bits_budget_remaining_norm: Optional[float] = None,
    ) -> np.ndarray:
        parts = []
        if queue_norm is None:
            queue_norm = min(len(self.uav_data_queue) / float(self.queue_ref), 1.0)
        if uav_energy_norm is None:
            uav_energy_norm = float(self.uav_energy / self.uav_energy_den)
        if uav_info_level_norm is None:
            uav_info_level_norm = float(self.uav_info_level / self.uav_info_level_den)
        if snr_norm is None:
            snr_norm = self._get_snr_norm()
        if comm_capacity_norm is None:
            comm_capacity_norm = self._get_comm_capacity_norm()
        if steps_remaining_norm is None:
            steps_remaining_norm = self._get_steps_remaining_norm()
        if bits_budget_remaining_norm is None:
            bits_budget_remaining_norm = self._get_bits_budget_remaining_norm()

        parts.append(self.uav_pos / self.grid_norm_den)
        parts.append(np.array([uav_energy_norm], dtype=float))
        parts.append(np.array([queue_norm], dtype=float))
        parts.append(np.array([self.current_bw_ratio], dtype=float))
        parts.append(self._encode_local_goal_for_uav_obs(current_target=current_target))
        parts.append((self.ugv_pos / self.grid_norm_den).astype(float))
        parts.append(
            np.array(
                [
                    float(snr_norm),
                    float(np.linalg.norm(self.uav_pos - self.ugv_pos) / (self.max_grid_diag + 1e-9)),
                    float(comm_capacity_norm),
                ],
                dtype=float,
            )
        )
        parts.append(
            np.array(
                [
                    float(steps_remaining_norm),
                    float(bits_budget_remaining_norm),
                ],
                dtype=float,
            )
        )

        obs = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _build_uav_move_action_mask(self) -> np.ndarray:
        valid_dirs = np.zeros(self.uav_direction_choices, dtype=bool)
        for direction_choice_idx, direction_idx in enumerate(self.uav_direction_ids):
            valid_dirs[direction_choice_idx] = self._can_follow_direction(
                position=self.uav_pos,
                direction_idx=direction_idx,
                step_count=self.uav_step_count,
                validator=self.scene.is_uav_position_valid,
                stop_at_target=True,
            )
        return valid_dirs

    def _build_uav_bw_action_mask(self) -> np.ndarray:
        return np.ones(self.uav_bw_action_size, dtype=bool)

    def _build_uav_action_mask(self) -> np.ndarray:
        """Compatibility helper that returns the factored joint mask."""
        move_mask = self._build_uav_move_action_mask()
        bw_mask = self._build_uav_bw_action_mask()
        return np.repeat(move_mask, self.uav_bw_action_size) & np.tile(
            bw_mask,
            self.uav_move_action_size,
        )

    def _build_ugv_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.ugv_action_size, dtype=bool)
        for direction_idx in range(self.ugv_action_size):
            mask[direction_idx] = self._can_follow_direction(
                position=self.ugv_pos,
                direction_idx=direction_idx,
                step_count=self.ugv_step_count,
                validator=self.scene.is_ugv_position_valid,
            )
        return mask

    def _encode_local_goal_for_uav_obs(
        self,
        current_target: Optional[PlannerTarget] = None,
    ) -> np.ndarray:
        if current_target is None:
            current_target = self._get_current_observation_target()
        if current_target is not None:
            goal_pos = np.array([float(current_target.gx), float(current_target.gy)], dtype=float)
            goal_delta = goal_pos - self.uav_pos
            goal_dist = manhattan_distance(self.uav_pos, goal_pos)
            radius_den = float(max(self.local_planner_radius, 1))
            return np.array(
                [
                    np.clip(goal_delta[0] / radius_den, -1.0, 1.0),
                    np.clip(goal_delta[1] / radius_den, -1.0, 1.0),
                    np.clip(goal_dist / radius_den, 0.0, 1.0),
                    float(current_target.center_freq) / self.max_freq_den,
                    float(np.tanh(current_target.score)),
                ],
                dtype=float,
            )
        return np.zeros(5, dtype=float)

    def _build_ugv_obs(
        self,
        current_target: Optional[PlannerTarget] = None,
        snr_norm: Optional[float] = None,
        uav_info_level_norm: Optional[float] = None,
        queue_bits_norm: Optional[float] = None,
    ) -> np.ndarray:
        if snr_norm is None:
            snr_norm = self._get_snr_norm()
        if uav_info_level_norm is None:
            uav_info_level_norm = float(self.uav_info_level / self.uav_info_level_den)
        if queue_bits_norm is None:
            queue_bits_norm = self._get_queue_bits_norm()
        parts = []
        parts.append(self.ugv_pos / self.grid_norm_den)
        parts.append(self.uav_pos / self.grid_norm_den)
        parts.append(np.array([snr_norm], dtype=float))
        parts.append(np.array([uav_info_level_norm], dtype=float))
        parts.append(self._encode_planner_target_relative_to_ugv(current_target=current_target))
        parts.append(np.asarray(self.last_uav_heading, dtype=float))
        parts.append(np.array([queue_bits_norm], dtype=float))
        obs = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _build_critic_state(
        self,
        current_target: Optional[PlannerTarget] = None,
        snr_norm: Optional[float] = None,
        queue_norm: Optional[float] = None,
        uav_energy_norm: Optional[float] = None,
        uav_info_level_norm: Optional[float] = None,
    ) -> np.ndarray:
        if snr_norm is None:
            snr_norm = float(np.tanh(self.ugv_channel_info.snr_db / self.snr_norm_den))
        if queue_norm is None:
            queue_norm = min(len(self.uav_data_queue) / float(self.queue_ref), 1.0)
        if uav_energy_norm is None:
            uav_energy_norm = float(self.uav_energy / self.uav_energy_den)
        if uav_info_level_norm is None:
            uav_info_level_norm = float(self.uav_info_level / self.uav_info_level_den)

        parts = []
        parts.append(self.uav_pos / self.grid_norm_den)
        parts.append(self.ugv_pos / self.grid_norm_den)
        parts.append(np.array([uav_energy_norm], dtype=float))
        parts.append(np.array([snr_norm], dtype=float))
        parts.append(np.array([queue_norm], dtype=float))
        parts.append(np.array([self.radio_map_state.nmse], dtype=float))
        parts.append(np.array([self.current_bw_ratio], dtype=float))
        parts.append(np.array([uav_info_level_norm], dtype=float))
        parts.append(self._extract_planner_state_features(current_target=current_target))
        parts.append(
            np.array(
                [
                    float(self.recent_delta_nmse_norm),
                    float(self.recent_delta_queue_norm),
                    float(self.recent_delta_snr_norm),
                ],
                dtype=float,
            )
        )
        parts.append(
            np.array(
                [
                    float(np.mean(normalize_score_map(self.uncertainty.spatial_uncertainty))),
                    float(np.mean(np.any(self.sampled_mask, axis=2))),
                    float(self._get_total_bits_norm()),
                ],
                dtype=float,
            )
        )
        state = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    def _encode_planner_target_relative_to_ugv(
        self,
        current_target: Optional[PlannerTarget] = None,
    ) -> np.ndarray:
        if current_target is None:
            current_target = self._get_current_observation_target()
        if current_target is None:
            return np.zeros(3, dtype=float)

        target_pos = np.array([float(current_target.gx), float(current_target.gy)], dtype=float)
        delta = target_pos - self.ugv_pos
        dist = float(np.linalg.norm(delta))
        radius_den = float(max(self.local_planner_radius, 1))
        return np.array(
            [
                np.clip(delta[0] / radius_den, -1.0, 1.0),
                np.clip(delta[1] / radius_den, -1.0, 1.0),
                np.clip(dist / (self.max_grid_diag + 1e-9), 0.0, 1.0),
            ],
            dtype=float,
        )

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
            features[idx] = float(np.tanh(current_target.score))
            idx += 1

        if idx < n_features:
            features[idx] = float(np.linalg.norm(self.uav_pos - target_grid_pos) / (self.max_grid_diag + 1e-9))
            idx += 1
        if idx < n_features:
            features[idx] = float(np.linalg.norm(self.ugv_pos - target_grid_pos) / (self.max_grid_diag + 1e-9))
        return features


class VecUAVUGVEnvironment:
    """Simple synchronous vectorized wrapper."""

    def __init__(
        self,
        num_envs: int,
        config: Config,
        env_factory=None,
    ):
        self.num_envs = num_envs
        self.config = config
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
            scene_map=GridScene(self.config),
        )

    def _peek_reset_seed(self, env_idx: int) -> int:
        count = int(self._env_reset_counts[env_idx])
        max_seed = (2 ** 32) - 1
        seed = (self.base_seed + env_idx + count * self.num_envs) % max_seed
        return int(seed)

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
        uav_move_actions: np.ndarray,
        uav_bw_actions: np.ndarray,
        ugv_actions: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict]]:
        results = []
        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(
                int(uav_move_actions[i]),
                int(uav_bw_actions[i]),
                int(ugv_actions[i]),
            )
            if term or trunc:
                info["terminal_obs"] = obs
                info["timeout_bootstrap_critic_state"] = np.asarray(
                    obs["critic_state"],
                    dtype=np.float32,
                ).copy()
                obs, _ = env.reset(seed=self._next_reset_seed(i))
            results.append((obs, rew, term, trunc, info))

        return _pack_vec_step_results(results)

    def close(self) -> None:
        return None


class SubprocVecUAVUGVEnvironment:
    """Fork-based vectorized wrapper that instantiates each env inside its worker."""

    def __init__(
        self,
        num_envs: int,
        config: Config,
        env_factory,
    ):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if env_factory is None:
            raise ValueError("SubprocVecUAVUGVEnvironment requires a non-null env_factory")

        available_methods = mp.get_all_start_methods()
        if "fork" not in available_methods:
            raise RuntimeError(
                "SubprocVecUAVUGVEnvironment requires the 'fork' start method."
            )

        self.num_envs = int(num_envs)
        self.config = config
        self.base_seed = int(config.mappo.seed)
        self._env_reset_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.closed = False
        self.ctx = mp.get_context("fork")
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
                    args=(worker_remote, env_factory, env_idx),
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

    def _peek_reset_seed(self, env_idx: int) -> int:
        count = int(self._env_reset_counts[env_idx])
        max_seed = (2 ** 32) - 1
        seed = (self.base_seed + env_idx + count * self.num_envs) % max_seed
        return int(seed)

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
        uav_move_actions: np.ndarray,
        uav_bw_actions: np.ndarray,
        ugv_actions: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict]]:
        for env_idx in range(self.num_envs):
            self._send_command(
                env_idx,
                "step",
                (
                    int(uav_move_actions[env_idx]),
                    int(uav_bw_actions[env_idx]),
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
