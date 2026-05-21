"""
Configuration for UAV-UGV Cooperative Spectrum Sensing with MAPPO.

All hyperparameters and environment settings are centralized here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from du_iibtd_based.du_iibtd_learn_nu import (
    DEFAULT_DU_IIBTD_CHECKPOINTS,
    DU_IIBTD_BACKENDS,
    default_du_iibtd_checkpoints_for_backend,
)

DEFAULT_RADIOSEER_ROOT = "RadioSeerDPM100PSD"
DEFAULT_RADIOSEER_SAMPLE_INDEX = 8513
DEFAULT_RADIOSEER_SCENE_INDICES = [8513, 1807, 1651, 1371, 10001]


@dataclass
class SceneConfig:
    """City scene and grid configuration."""
    grid_size: Tuple[int, int] = (64, 64)       # Placeholder only; overwritten from loaded RadioSeer sample shape
    grid_spacing: float = 2                    # Meters per cell used by motion, distance, and SNR calculations
    uav_height: float = 30.0                      # Fixed UAV flight altitude (m)
    ugv_height: float = 0.0                       # UGV antenna height above ground (m)
    building_height_m: float = 25.0              # Default building height used for 3D LOS blocking (m)
    scene_source: str = "radioseerselect"        # Loader family tag for manifest-compatible RadioSeer datasets
    radioseer_root: str = DEFAULT_RADIOSEER_ROOT   # Dataset folder relative to repo root or absolute path
    radioseer_sample_index: int = DEFAULT_RADIOSEER_SAMPLE_INDEX  # Selected manifest row in the configured RadioSeer dataset
    radioseer_scene_indices: List[int] = field(default_factory=lambda: list(DEFAULT_RADIOSEER_SCENE_INDICES))

    # Frequency configuration
    total_freq_bands_nums: int = 30               # Total number of frequency bands (K)
    freq_start: float = 3.5e9                      # Start frequency (Hz)
    freq_end: float = 3.7e9                        # End frequency (Hz)

    @property
    def freq_bands(self) -> np.ndarray:
        return np.linspace(self.freq_start, self.freq_end, self.total_freq_bands_nums)

    @property
    def scene_width(self) -> float:
        return self.grid_size[0] * self.grid_spacing

    @property
    def scene_height(self) -> float:
        return self.grid_size[1] * self.grid_spacing


@dataclass
class UAVConfig:
    """UAV agent configuration."""
    # Movement
    num_directions: int = 5                        # 4 cardinal directions + stay
    step_size: float = 4.0                        # Movement distance per action in grid cells

    # Energy
    max_energy: float = 60_000                     # Maximum energy budget (Joules)
    flight_power: float = 12.0                     # Power consumption during flight (W)
    hover_power: float = 8.0                      # Power consumption during hover (W)
    sensing_power: float = 5.0                     # Power for spectrum sensing (W)
    step_duration: float = 1.0                     # Duration per grid hop / hover step (seconds)

    # Bandwidth
    total_bandwidth: float = 100e6               # Total RF bandwidth (Hz)
    total_bw_num: int = 12                        # Total discrete bandwidth units
    default_bw_ratio: float = 0.6                 # Default sensing bandwidth ratio
    queue_capacity_packets: int = 8               # Max pending sample packets buffered on the UAV

    bandwidth_ratios: List[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.5, 0.6]  # Discrete sensing bandwidth ratios
    )

    @property
    def num_bandwidth_ratios(self) -> int:
        return len(self.bandwidth_ratios)

    @property
    def unit_bandwidth_hz(self) -> float:
        return self.total_bandwidth / max(1, self.total_bw_num)


@dataclass
class UGVConfig:
    """UGV agent configuration."""
    num_directions: int = 5                        # 4 cardinal directions + stay
    step_size: float = 5.0                        # Movement distance per step in grid cells


@dataclass
class CommConfig:
    """Communication channel configuration."""
    carrier_freq: float = 3.5e9                    # Communication carrier frequency (Hz)
    tx_power_dbm: float = 1.0                      # UAV transmit power (dBm)
    noise_figure_db: float = 8.0                   # Receiver noise figure (dB)
    data_per_sample: float = 8e6                   # Data size per frequency band sample (bits)
    shadow_std_los_db: float = 2.0                 # Log-normal shadow std when LoS (dB)
    shadow_std_nlos_db: float = 6.0                # Log-normal shadow std when NLoS (dB)
    nlos_excess_db: float = 15.0                   # Extra NLoS attenuation (dB)


@dataclass
class RewardConfig:
    """Reward function coefficients."""
    alpha_nmse: float = 20.0                       # Weight for normalized NMSE-improvement reward (delta_nmse_norm)
    nmse_signed_clip: float = 0.25                # Clip signed normalized NMSE delta before scaling; <=0 disables clipping
    target_gap_penalty_coef: float = 1.0          # Diagnostic-only target-gap penalty coefficient; not added to team reward
    alpha_unc: float = 0.0                         # Weight for normalized uncertainty-reduction reward (delta_unc_norm)
    lambda_new_freq: float = 0.03                  # Reward per newly sampled frequency band; default off for NMSE-focused training
    lambda_new_spatial: float = 0.5               # Reward for visiting a previously unsampled grid cell
    beta_tx: float = 0.0                          # Retained for compatibility; tx reward is disabled in local-goal shaping
    gamma_queue: float = 1.7                      # Weight for queue-bits penalty
    lambda_uav_progress: float = 5              # Reward for UAV moving closer to active local target
    lambda_uav_backtrack: float = 8             # Penalty for UAV moving away from active target
    lambda_ugv_progress: float = 5             # Reward for UGV moving closer to UAV
    lambda_ugv_backtrack: float = 8             # Penalty for UGV moving away from UAV
    ugv_progress_uav_weight: float = 0.4        # Relative weight of UGV->UAV grid-distance progress
    ugv_progress_target_weight: float = 0.6     # Relative weight of UGV->target shortest-path progress
    bootstrap_progress_scale: float = 1.0        # Keep bootstrap targets reward-aligned with the observation target encoding
    lambda_spatial_revisit: float = 0.8          # Penalty for repeated sensing at the same grid cell with low novelty
    lambda_ugv_building_clearance: float = 0.1   # Penalty when UGV is too close to buildings
    ugv_building_safe_clearance: int = 3         # Desired Chebyshev distance from buildings in grid cells
    local_goal_arrival_bonus: float = 1.0        # Bonus when the UAV reaches the planner-selected local goal
    q_ref: float = 8.0                             # Reference queue length for normalization
    accuracy_target_nmse: float = 0.08             # Target NMSE tracked for diagnostics; does not end episodes
    terminal_failure_penalty: float = 20.0        # Penalty only when UAV energy is exhausted before the horizon


@dataclass
class ObservationConfig:
    """Observation space configuration."""
    # Planner-aware features for Critic
    num_planner_features: int = 5                 # target(x,y), center_freq, dist_uav, dist_ugv


@dataclass
class PlannerConfig:
    """UGV-side active-planner configuration."""
    target_count: int = 1                          # Number of planner targets exposed each cycle
    obs_target_slots: int = 1                      # Number of target slots encoded into UAV observation
    target_mode: str = "hybrid"                     # Planner target scope: local | global | hybrid
    initial_observation_mode: str = "prefill"    # Warmup mode before planner: bootstrap | prefill
    local_planner_radius: int = 30                  # Planner only searches targets within this Manhattan radius of the UAV
    hybrid_nmse_stall_steps: int = 2               # In hybrid mode, switch local->global after this many low-improvement map updates
    hybrid_nmse_stall_threshold: float = 0.02      # Treat planner NMSE improvement below this threshold as stalled
    hybrid_global_hold_intervals: int = 5          # In hybrid mode, hold global submode for this many ensemble intervals
    hybrid_local_reentry_min_targets: int = 2      # Effective local reentry threshold is max(this value, target_count)
    prefill_percent: float = 6                  # Prefill observations as a percentage of the sensing-budget basis
    prefill_budget_basis: int = 0                  # Prefill budget basis; <=0 falls back to episode_max_steps
    init_pair_max_distance: float = 7.0            # Max initial UAV-UGV separation in grid units
    init_building_clearance: int = 5               # Prefer initial UAV/UGV cells with this many grid cells of building clearance
    bootstrap_building_clearance: int = 5          # Prefer bootstrap targets with this many grid cells of building clearance
    flush_reconstruction_on_episode_end: bool = False  # Force one last expensive reconstruction at terminal steps; keep off for faster training
    target_arrival_radius_steps: float = 1.0       # Treat sampled UAV positions within this many UAV steps of target as arrival
    target_suppression_radius_steps: float = 1.0   # Suppress completed/stuck target neighborhoods until the next ensemble refresh
    target_stuck_no_sample_steps: int = 4          # Retarget after this many consecutive active-plan steps without a valid UAV sample

    ensemble_refresh_interval: int = 3            # Refresh ensemble mean/variance and planner targets after this many newly delivered samples

    min_samples_for_ensemble: int = 12            # Start planner/ensemble after enough UGV-side effective samples

    # ShareMem ensemble
    ensemble_size: int = 3 
    ensemble_quality_weighted: bool = True         # Weight ensemble members by observed-entry NMSE
    ensemble_init_jitter_scale: float = 1e-2       # Tiny per-member state jitter for diversity
    ensemble_full_refresh_interval: int = 0        # UAVTest2-style periodic full ensemble refit; <=0 disables periodic refresh
    nmse_refresh_delta: float = 0.1                # Trigger a full ensemble refit when incremental NMSE degrades by this much; <=0 disables it
    incremental_outer_iters: int = 2               # Outer iterations for fit_incremental between full refreshes
    incremental_max_svt_iters: int = 20            # Max SVT iterations for incremental solver updates

    # Acquisition weights
    lambda_u: float = 1.0
    beta_f: float = 0.3
    redundancy_length: float = 20.0             # Visualization-only spacing for displayed global top-k markers

    # DU-IIBTD reconstruction backend
    iibtd_mu: float = 1e-1                        # DU-IIBTD runtime penalty parameter mu
    iibtd_nu: float = 1.5                         # Fallback nu only when checkpoint metadata is missing
    iibtd_kernel_bandwidth: float = 0.25          # Fallback bandwidth only when checkpoint metadata is missing
    iibtd_backend: str = "du_iibtd"               # du_iibtd | du_iibtd_res_sr | du_iibtd_res_sr_learn_nu
    iibtd_device: str = "auto"                    # auto | cuda | cuda:0 | cpu; runtime may separate it from MAPPO
    du_iibtd_checkpoints: List[str] = field(
        default_factory=lambda: list(DEFAULT_DU_IIBTD_CHECKPOINTS)
    )
    du_iibtd_min_sensors_for_update: int = 0      # <=0 lets the adapter use checkpoint config
    du_iibtd_update_batch_size: int = 0           # <=0 lets the adapter use checkpoint config


@dataclass
class MAPPOConfig:
    """MAPPO algorithm hyperparameters."""
    # Training
    num_envs: int = 5                              # One rollout worker per DPM100PSD scene
    total_timesteps: int = 120_000                    # 120 updates with 200 steps x 5 scenes
    episode_max_steps: int = 200                   # Maximum steps per episode
    rollout_length: int = 200                      # Must match episode_max_steps
    num_minibatches: int = 4                       # Number of minibatches for PPO update
    num_epochs: int = 6                           # PPO epochs per update
    
    # Optimization
    lr_actor: float = 1e-4                         # Actor learning rate
    lr_critic: float = 1e-4                        # Critic learning rate
    gamma: float = 0.99                            # Discount factor
    gae_lambda: float = 0.95                       # GAE lambda
    clip_epsilon: float = 0.2                      # PPO clipping parameter
    max_grad_norm: float = 0.5                     # Gradient clipping norm
    entropy_coef: float = 0.005                     # Entropy bonus coefficient
    value_loss_coef: float = 0.5                   # Value loss coefficient

    # Network architecture
    actor_hidden_dims: List[int] = field(
        default_factory=lambda: [256, 128, 64]
    )
    critic_hidden_dims: List[int] = field(
        default_factory=lambda: [512, 256, 128]
    )
    use_feature_norm: bool = True                  # Layer normalization on input features
    use_orthogonal_init: bool = True               # Orthogonal weight initialization

    # Misc
    seed: int = 42
    device: str = "cuda:0"                        # "cuda" or "cpu"; runtime prefers cuda:1 when available
    vec_backend: str = "subproc"                     # "sync" or "subproc" rollout backend
    save_interval: int = 50                        # Save model every N updates
    log_interval: int = 20                         # Log metrics every N updates
    eval_interval: int = 500                        # Evaluate every N updates
    eval_episodes: int = 10                         # Number of evaluation episodes
    eval_seed_stride: int = 0                      # Eval reset-seed stride across updates; 0 keeps eval seeds fixed
    model_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    scene: SceneConfig = field(default_factory=SceneConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    ugv: UGVConfig = field(default_factory=UGVConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)

    def __post_init__(self):
        """Validate configuration consistency."""
        def _ensure(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(message)

        if len(self.scene.grid_size) != 2:
            raise ValueError(
                f"scene.grid_size must have length 2, got {self.scene.grid_size!r}"
            )
        self.scene.grid_size = tuple(int(v) for v in self.scene.grid_size)
        _ensure(
            all(v > 0 for v in self.scene.grid_size),
            f"scene.grid_size must contain positive integers, got {self.scene.grid_size!r}",
        )

        grid_spacing = float(self.scene.grid_spacing)
        _ensure(grid_spacing > 0.0, f"scene.grid_spacing must be positive, got {grid_spacing}")
        self.scene.scene_source = str(self.scene.scene_source).strip().lower() or "radioseerselect"
        _ensure(
            self.scene.scene_source == "radioseerselect",
            "scene.scene_source must be radioseerselect, got "
            f"{self.scene.scene_source!r}",
        )
        self.scene.radioseer_root = str(self.scene.radioseer_root).strip() or DEFAULT_RADIOSEER_ROOT
        self.scene.radioseer_sample_index = int(self.scene.radioseer_sample_index)
        self.scene.uav_height = float(self.scene.uav_height)
        self.scene.ugv_height = float(self.scene.ugv_height)
        self.scene.building_height_m = float(self.scene.building_height_m)
        _ensure(
            self.scene.uav_height >= 0.0,
            f"scene.uav_height must be non-negative, got {self.scene.uav_height}",
        )
        _ensure(
            self.scene.ugv_height >= 0.0,
            f"scene.ugv_height must be non-negative, got {self.scene.ugv_height}",
        )
        _ensure(
            self.scene.uav_height >= self.scene.ugv_height,
            "scene.uav_height must be >= scene.ugv_height, got "
            f"{self.scene.uav_height} < {self.scene.ugv_height}",
        )
        _ensure(
            self.scene.building_height_m >= 0.0,
            "scene.building_height_m must be non-negative, got "
            f"{self.scene.building_height_m}",
        )
        self.scene.total_freq_bands_nums = int(self.scene.total_freq_bands_nums)
        _ensure(
            self.scene.total_freq_bands_nums > 0,
            "scene.total_freq_bands_nums must be positive, got "
            f"{self.scene.total_freq_bands_nums}",
        )
        _ensure(
            float(self.scene.freq_end) > float(self.scene.freq_start),
            "scene.freq_end must be greater than scene.freq_start, got "
            f"{self.scene.freq_start} -> {self.scene.freq_end}",
        )

        self.uav.total_bw_num = int(self.uav.total_bw_num)
        _ensure(
            self.uav.total_bw_num > 1,
            f"uav.total_bw_num must be greater than 1, got {self.uav.total_bw_num}",
        )
        self.uav.queue_capacity_packets = int(self.uav.queue_capacity_packets)
        _ensure(
            self.uav.queue_capacity_packets > 0,
            "uav.queue_capacity_packets must be positive, got "
            f"{self.uav.queue_capacity_packets}",
        )
        _ensure(
            float(self.uav.total_bandwidth) > 0.0,
            f"uav.total_bandwidth must be positive, got {self.uav.total_bandwidth}",
        )
        self.uav.default_bw_ratio = float(self.uav.default_bw_ratio)
        _ensure(
            0.0 < self.uav.default_bw_ratio < 1.0,
            "uav.default_bw_ratio must be in (0, 1), got "
            f"{self.uav.default_bw_ratio}",
        )
        self.uav.bandwidth_ratios = [float(ratio) for ratio in self.uav.bandwidth_ratios]
        _ensure(len(self.uav.bandwidth_ratios) > 0, "uav.bandwidth_ratios must not be empty")
        invalid_bw_ratios = [
            ratio for ratio in self.uav.bandwidth_ratios if not (0.0 < ratio < 1.0)
        ]
        _ensure(
            not invalid_bw_ratios,
            "uav.bandwidth_ratios must all be in (0, 1), got "
            f"{invalid_bw_ratios}",
        )

        self.uav.num_directions = int(self.uav.num_directions)
        self.ugv.num_directions = int(self.ugv.num_directions)
        _ensure(
            self.uav.num_directions > 0,
            f"uav.num_directions must be positive, got {self.uav.num_directions}",
        )
        _ensure(
            self.ugv.num_directions > 0,
            f"ugv.num_directions must be positive, got {self.ugv.num_directions}",
        )

        self.uav.step_size = float(self.uav.step_size)
        self.ugv.step_size = float(self.ugv.step_size)
        _ensure(self.uav.step_size > 0.0, f"uav.step_size must be positive, got {self.uav.step_size}")
        _ensure(self.ugv.step_size > 0.0, f"ugv.step_size must be positive, got {self.ugv.step_size}")
        _ensure(
            np.isclose(self.uav.step_size, round(self.uav.step_size)),
            "uav.step_size must be an integer number of grid cells, got "
            f"{self.uav.step_size}",
        )
        _ensure(
            np.isclose(self.ugv.step_size, round(self.ugv.step_size)),
            "ugv.step_size must be an integer number of grid cells, got "
            f"{self.ugv.step_size}",
        )

        self.reward.q_ref = float(self.reward.q_ref)
        _ensure(
            self.reward.q_ref > 0.0,
            f"reward.q_ref must be positive, got {self.reward.q_ref}",
        )
        self.reward.lambda_ugv_building_clearance = float(
            self.reward.lambda_ugv_building_clearance
        )
        _ensure(
            self.reward.lambda_ugv_building_clearance >= 0.0,
            "reward.lambda_ugv_building_clearance must be non-negative, got "
            f"{self.reward.lambda_ugv_building_clearance}",
        )
        self.reward.ugv_building_safe_clearance = int(self.reward.ugv_building_safe_clearance)
        _ensure(
            self.reward.ugv_building_safe_clearance >= 1,
            "reward.ugv_building_safe_clearance must be >= 1, got "
            f"{self.reward.ugv_building_safe_clearance}",
        )
        self.reward.ugv_progress_uav_weight = float(self.reward.ugv_progress_uav_weight)
        self.reward.ugv_progress_target_weight = float(self.reward.ugv_progress_target_weight)
        _ensure(
            self.reward.ugv_progress_uav_weight >= 0.0,
            "reward.ugv_progress_uav_weight must be non-negative, got "
            f"{self.reward.ugv_progress_uav_weight}",
        )
        _ensure(
            self.reward.ugv_progress_target_weight >= 0.0,
            "reward.ugv_progress_target_weight must be non-negative, got "
            f"{self.reward.ugv_progress_target_weight}",
        )
        ugv_progress_weight_sum = (
            self.reward.ugv_progress_uav_weight + self.reward.ugv_progress_target_weight
        )
        _ensure(
            ugv_progress_weight_sum > 0.0,
            "reward.ugv_progress_uav_weight + reward.ugv_progress_target_weight "
            "must be positive",
        )
        self.reward.ugv_progress_uav_weight /= ugv_progress_weight_sum
        self.reward.ugv_progress_target_weight /= ugv_progress_weight_sum

        self.obs.num_planner_features = int(self.obs.num_planner_features)
        _ensure(
            self.obs.num_planner_features == 5,
            "obs.num_planner_features must be 5 after removing planner score from critic obs, got "
            f"{self.obs.num_planner_features}",
        )

        self.planner.target_count = int(self.planner.target_count)
        self.planner.obs_target_slots = int(self.planner.obs_target_slots)
        self.planner.target_mode = str(self.planner.target_mode).strip().lower() or "hybrid"
        self.planner.initial_observation_mode = (
            str(self.planner.initial_observation_mode).strip().lower() or "bootstrap"
        )
        self.planner.local_planner_radius = int(self.planner.local_planner_radius)
        self.planner.hybrid_nmse_stall_steps = int(self.planner.hybrid_nmse_stall_steps)
        self.planner.hybrid_nmse_stall_threshold = float(self.planner.hybrid_nmse_stall_threshold)
        self.planner.hybrid_global_hold_intervals = int(self.planner.hybrid_global_hold_intervals)
        self.planner.hybrid_local_reentry_min_targets = int(self.planner.hybrid_local_reentry_min_targets)
        self.planner.prefill_percent = float(self.planner.prefill_percent)
        self.planner.prefill_budget_basis = int(self.planner.prefill_budget_basis)
        self.planner.init_building_clearance = int(self.planner.init_building_clearance)
        self.planner.bootstrap_building_clearance = int(self.planner.bootstrap_building_clearance)
        self.planner.flush_reconstruction_on_episode_end = bool(
            self.planner.flush_reconstruction_on_episode_end
        )
        self.planner.target_arrival_radius_steps = float(self.planner.target_arrival_radius_steps)
        self.planner.target_suppression_radius_steps = float(
            self.planner.target_suppression_radius_steps
        )
        self.planner.target_stuck_no_sample_steps = int(
            self.planner.target_stuck_no_sample_steps
        )
        self.planner.ensemble_refresh_interval = int(self.planner.ensemble_refresh_interval)
        self.planner.ensemble_full_refresh_interval = int(self.planner.ensemble_full_refresh_interval)
        self.planner.nmse_refresh_delta = float(self.planner.nmse_refresh_delta)
        self.planner.incremental_outer_iters = int(self.planner.incremental_outer_iters)
        self.planner.incremental_max_svt_iters = int(self.planner.incremental_max_svt_iters)
        self.planner.ensemble_quality_weighted = bool(self.planner.ensemble_quality_weighted)
        self.planner.ensemble_init_jitter_scale = float(self.planner.ensemble_init_jitter_scale)
        self.planner.iibtd_backend = str(self.planner.iibtd_backend).strip().lower() or "du_iibtd"
        _ensure(
            self.planner.iibtd_backend in DU_IIBTD_BACKENDS,
            "planner.iibtd_backend must be one of "
            f"{sorted(DU_IIBTD_BACKENDS)}, got {self.planner.iibtd_backend!r}",
        )
        checkpoint_paths = [
            str(path).strip()
            for path in list(self.planner.du_iibtd_checkpoints or [])
            if str(path).strip()
        ]
        if (
            not checkpoint_paths
            or (
                self.planner.iibtd_backend != "du_iibtd"
                and checkpoint_paths == list(DEFAULT_DU_IIBTD_CHECKPOINTS)
            )
        ):
            checkpoint_paths = default_du_iibtd_checkpoints_for_backend(
                self.planner.iibtd_backend
            )
        self.planner.du_iibtd_checkpoints = checkpoint_paths
        self.planner.du_iibtd_min_sensors_for_update = int(
            self.planner.du_iibtd_min_sensors_for_update
        )
        self.planner.du_iibtd_update_batch_size = int(
            self.planner.du_iibtd_update_batch_size
        )
        _ensure(
            self.planner.target_count > 0,
            f"planner.target_count must be positive, got {self.planner.target_count}",
        )
        _ensure(
            self.planner.obs_target_slots > 0,
            "planner.obs_target_slots must be positive, got "
            f"{self.planner.obs_target_slots}",
        )
        _ensure(
            self.planner.obs_target_slots >= self.planner.target_count,
            "planner.obs_target_slots must be >= planner.target_count, got "
            f"{self.planner.obs_target_slots} < {self.planner.target_count}",
        )
        _ensure(
            self.planner.target_mode in {"local", "global", "hybrid"},
            "planner.target_mode must be one of local/global/hybrid, got "
            f"{self.planner.target_mode!r}",
        )
        _ensure(
            self.planner.initial_observation_mode in {"bootstrap", "prefill"},
            "planner.initial_observation_mode must be one of bootstrap/prefill, got "
            f"{self.planner.initial_observation_mode!r}",
        )
        _ensure(
            self.planner.ensemble_init_jitter_scale >= 0.0,
            "planner.ensemble_init_jitter_scale must be non-negative, got "
            f"{self.planner.ensemble_init_jitter_scale}",
        )
        _ensure(
            len(self.planner.du_iibtd_checkpoints) > 0,
            "planner.du_iibtd_checkpoints must not be empty.",
        )
        _ensure(
            self.planner.du_iibtd_min_sensors_for_update >= 0,
            "planner.du_iibtd_min_sensors_for_update must be >= 0, got "
            f"{self.planner.du_iibtd_min_sensors_for_update}",
        )
        _ensure(
            self.planner.du_iibtd_update_batch_size >= 0,
            "planner.du_iibtd_update_batch_size must be >= 0, got "
            f"{self.planner.du_iibtd_update_batch_size}",
        )
        _ensure(
            self.planner.local_planner_radius > 0,
            "planner.local_planner_radius must be positive, got "
            f"{self.planner.local_planner_radius}",
        )
        _ensure(
            self.planner.hybrid_nmse_stall_steps > 0,
            "planner.hybrid_nmse_stall_steps must be positive, got "
            f"{self.planner.hybrid_nmse_stall_steps}",
        )
        _ensure(
            self.planner.hybrid_nmse_stall_threshold >= 0.0,
            "planner.hybrid_nmse_stall_threshold must be >= 0, got "
            f"{self.planner.hybrid_nmse_stall_threshold}",
        )
        _ensure(
            self.planner.hybrid_global_hold_intervals > 0,
            "planner.hybrid_global_hold_intervals must be positive, got "
            f"{self.planner.hybrid_global_hold_intervals}",
        )
        _ensure(
            self.planner.hybrid_local_reentry_min_targets >= 0,
            "planner.hybrid_local_reentry_min_targets must be >= 0, got "
            f"{self.planner.hybrid_local_reentry_min_targets}",
        )
        _ensure(
            0.0 <= self.planner.prefill_percent <= 100.0,
            "planner.prefill_percent must be in [0, 100], got "
            f"{self.planner.prefill_percent}",
        )
        _ensure(
            self.planner.prefill_budget_basis >= 0,
            "planner.prefill_budget_basis must be >= 0, got "
            f"{self.planner.prefill_budget_basis}",
        )
        if self.planner.initial_observation_mode == "prefill":
            _ensure(
                self.planner.prefill_percent > 0.0,
                "planner.prefill_percent must be > 0 when planner.initial_observation_mode='prefill'",
            )
        _ensure(
            self.planner.init_building_clearance >= 0,
            "planner.init_building_clearance must be non-negative, got "
            f"{self.planner.init_building_clearance}",
        )
        _ensure(
            self.planner.bootstrap_building_clearance >= 0,
            "planner.bootstrap_building_clearance must be non-negative, got "
            f"{self.planner.bootstrap_building_clearance}",
        )
        self.mappo.num_envs = int(self.mappo.num_envs)
        self.mappo.total_timesteps = int(self.mappo.total_timesteps)
        self.mappo.episode_max_steps = int(self.mappo.episode_max_steps)
        self.mappo.rollout_length = int(self.mappo.rollout_length)
        self.mappo.num_minibatches = int(self.mappo.num_minibatches)
        self.mappo.num_epochs = int(self.mappo.num_epochs)
        self.mappo.eval_episodes = int(self.mappo.eval_episodes)
        _ensure(self.mappo.num_envs > 0, f"mappo.num_envs must be positive, got {self.mappo.num_envs}")
        _ensure(
            self.mappo.total_timesteps > 0,
            f"mappo.total_timesteps must be positive, got {self.mappo.total_timesteps}",
        )
        _ensure(
            self.mappo.episode_max_steps > 0,
            f"mappo.episode_max_steps must be positive, got {self.mappo.episode_max_steps}",
        )
        _ensure(
            self.mappo.rollout_length > 0,
            f"mappo.rollout_length must be positive, got {self.mappo.rollout_length}",
        )
        _ensure(
            self.mappo.num_minibatches > 0,
            f"mappo.num_minibatches must be positive, got {self.mappo.num_minibatches}",
        )
        _ensure(
            self.mappo.num_epochs > 0,
            f"mappo.num_epochs must be positive, got {self.mappo.num_epochs}",
        )
        _ensure(
            self.mappo.eval_episodes > 0,
            f"mappo.eval_episodes must be positive, got {self.mappo.eval_episodes}",
        )
        if self.planner.ensemble_refresh_interval <= 0:
            raise ValueError(
                "planner.ensemble_refresh_interval must be positive, got "
                f"{self.planner.ensemble_refresh_interval}"
            )
        _ensure(
            self.planner.target_arrival_radius_steps >= 0.0,
            "planner.target_arrival_radius_steps must be >= 0, got "
            f"{self.planner.target_arrival_radius_steps}",
        )
        _ensure(
            self.planner.target_suppression_radius_steps >= 0.0,
            "planner.target_suppression_radius_steps must be >= 0, got "
            f"{self.planner.target_suppression_radius_steps}",
        )
        _ensure(
            self.planner.target_stuck_no_sample_steps > 0,
            "planner.target_stuck_no_sample_steps must be positive, got "
            f"{self.planner.target_stuck_no_sample_steps}",
        )
        _ensure(
            self.planner.ensemble_full_refresh_interval >= 0,
            "planner.ensemble_full_refresh_interval must be >= 0, got "
            f"{self.planner.ensemble_full_refresh_interval}",
        )
        _ensure(
            self.planner.nmse_refresh_delta >= 0.0,
            "planner.nmse_refresh_delta must be >= 0, got "
            f"{self.planner.nmse_refresh_delta}",
        )
        _ensure(
            self.planner.incremental_outer_iters > 0,
            "planner.incremental_outer_iters must be positive, got "
            f"{self.planner.incremental_outer_iters}",
        )
        _ensure(
            self.planner.incremental_max_svt_iters > 0,
            "planner.incremental_max_svt_iters must be positive, got "
            f"{self.planner.incremental_max_svt_iters}",
        )
        self.planner.redundancy_length = float(self.planner.redundancy_length)
        _ensure(
            self.planner.redundancy_length >= 0.0,
            "planner.redundancy_length must be >= 0, got "
            f"{self.planner.redundancy_length}",
        )

        self.planner.iibtd_mu = float(self.planner.iibtd_mu)
        _ensure(
            self.planner.iibtd_mu > 0.0,
            f"planner.iibtd_mu must be positive, got {self.planner.iibtd_mu}",
        )
        self.planner.iibtd_nu = float(self.planner.iibtd_nu)
        _ensure(
            self.planner.iibtd_nu > 0.0,
            f"planner.iibtd_nu must be positive, got {self.planner.iibtd_nu}",
        )
        self.planner.iibtd_kernel_bandwidth = float(self.planner.iibtd_kernel_bandwidth)
        _ensure(
            self.planner.iibtd_kernel_bandwidth > 0.0,
            "planner.iibtd_kernel_bandwidth must be positive, got "
            f"{self.planner.iibtd_kernel_bandwidth}",
        )

        self.planner.iibtd_backend = str(self.planner.iibtd_backend).strip().lower() or "du_iibtd"
        _ensure(
            self.planner.iibtd_backend in DU_IIBTD_BACKENDS,
            "planner.iibtd_backend must be one of "
            f"{sorted(DU_IIBTD_BACKENDS)}, got {self.planner.iibtd_backend!r}",
        )

        self.planner.iibtd_device = str(self.planner.iibtd_device).strip() or "auto"
