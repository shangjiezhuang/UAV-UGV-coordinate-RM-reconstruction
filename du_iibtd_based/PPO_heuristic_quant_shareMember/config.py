"""
Configuration for UAV-UGV Cooperative Spectrum Sensing with MAPPO.

All hyperparameters and environment settings are centralized here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class SceneConfig:
    """City scene and grid configuration."""
    grid_size: Tuple[int, int] = (64, 64)       # Spatial grid resolution
    grid_spacing: float = 2.0                    # Meters per grid cell
    uav_height: float = 30.0                      # Fixed UAV flight altitude (m)
    ugv_height: float = 0.0                       # UGV antenna height above ground (m)
    building_height_m: float = 25.0              # Default building height used for 3D LOS blocking (m)
    scene_source: str = "radioseerselect"        # radioseerselect
    radioseer_root: str = "RadioSeerDPM100PSD"   # Dataset folder relative to repo root or absolute path
    radioseer_sample_index: int = 8513           # Selected manifest row in RadioSeerDPM100PSD
    radioseer_scene_indices: List[int] = field(default_factory=lambda: [8513, 1807, 1651, 1371, 10001])

    # Frequency configuration
    total_freq_bands_nums: int = 30                     # Total number of frequency bands (K)
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
    step_size: int = 4                            # Movement grid cells per action

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
    quant_bits: List[int] = field(
        default_factory=lambda: [10, 8, 6]         # Discrete log-domain quantization bits
    )
    default_quant_bits: int = 10                   # Initial UAV quantization bit depth

    @property
    def num_bandwidth_ratios(self) -> int:
        return len(self.bandwidth_ratios)

    @property
    def num_quant_bits(self) -> int:
        return len(self.quant_bits)

    @property
    def unit_bandwidth_hz(self) -> float:
        return self.total_bandwidth / max(1, self.total_bw_num)


@dataclass
class UGVConfig:
    """UGV agent configuration."""
    num_directions: int = 5                        # 4 cardinal directions + stay
    step_size: int = 5                            # Movement grid cells per step


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
    lambda_new_freq: float = 0.02                  # Reward per newly sampled frequency band; default off for NMSE-focused training
    lambda_new_spatial: float = 0.4               # Reward for visiting a previously unsampled grid cell
    beta_tx: float = 0.0                          # Retained for compatibility; tx reward is disabled in local-goal shaping
    gamma_queue: float = 1.2                      # Weight for queue-bits penalty
    lambda_uav_progress: float = 3.0             # Reward for normalized UAV progress toward the active target
    lambda_uav_backtrack: float = 6.0            # Penalty for normalized UAV backtracking away from target
    bootstrap_progress_scale: float = 1.0        # Keep bootstrap targets reward-aligned with the observation target encoding
    lambda_spatial_revisit: float = 1.0          # Bounded penalty for low-novelty spatial revisits
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
    target_mode: str = "hybrid"                    # Planner target scope: local | global | hybrid
    initial_observation_mode: str = "prefill"    # Warmup mode before planner: bootstrap | prefill
    local_planner_radius: int = 30                  # Planner only searches targets within this Manhattan radius of the UAV
    hybrid_stall_update_count: int = 2            # Switch local->global after this many weak map updates in local mode
    hybrid_stall_nmse_threshold: float = 0.02     # Minimum absolute NMSE improvement that resets local stall accumulation
    hybrid_global_hold_intervals: int = 6        # Stay in global mode for this many ensemble intervals before local re-check
    hybrid_local_min_candidate_count: int = 2     # Local mode can resume once at least this many viable local targets remain
    prefill_percent: float = 6.0                  # Prefill observations as a percentage of the sensing-budget basis
    prefill_budget_basis: int = 0                  # Prefill budget basis; <=0 falls back to episode_max_steps
    init_pair_max_distance: float = 7.0            # Max initial UAV-UGV separation in grid units
    init_building_clearance: int = 5               # Prefer initial UAV/UGV cells with this many grid cells of building clearance
    bootstrap_building_clearance: int = 5          # Prefer bootstrap targets with this many grid cells of building clearance
    flush_reconstruction_on_episode_end: bool = False  # Force one last expensive reconstruction at terminal steps; keep off for faster training

    ensemble_refresh_interval: int = 3            # Trigger one fused reconstruction+ensemble update after this many newly delivered samples

    min_samples_for_ensemble: int = 12            # Start planner/ensemble after enough UGV-side effective samples

    # Shared-member ensemble. Each member sees all fused observations; diversity
    # comes from member-specific kernel bandwidth and initialization jitter.
    ensemble_size: int = 3 
    ensemble_quality_weighted: bool = True         # Weight ensemble members by observed-entry NMSE
    ensemble_full_refresh_interval: int = 0        # UAVTest2-style periodic full ensemble refit; <=0 disables periodic refresh
    nmse_refresh_delta: float = 0.1                # Trigger a full ensemble refit when incremental NMSE degrades by this much; <=0 disables it
    incremental_outer_iters: int = 2               # Outer iterations for fit_incremental between full refreshes
    incremental_max_svt_iters: int = 20            # Max SVT iterations for incremental solver updates

    # Acquisition weights
    lambda_u: float = 1.0
    beta_f: float = 0.3
    redundancy_length: float = 5.0

    # II-BTD reconstruction backend
    iibtd_mu: float = 1e-1                        # II-BTD solver penalty parameter mu
    iibtd_nu: float = 1e-1                        # II-BTD solver penalty parameter nu
    iibtd_kernel_bandwidth: float = 0.6          # II-BTD kernel bandwidth
    iibtd_backend: str = "du_iibtd_sr"            # du_iibtd_sr | du_iibtd_sr_learn_nu | gpu
    iibtd_device: str = "auto"                    # auto | cuda | cuda:0 | cpu; runtime prefers cuda:2 when available
    iibtd_gpu_phi_solver: str = "pgd"           # scipy | pgd; only used by backend="gpu"
    iibtd_du_checkpoint_path: str = (
        "DU_IIBTD_res_Sr/"
        "runs_t3_h04_res_balance_bw/checkpoints/best_nmse.pth,"
        "DU_IIBTD_res_Sr/"
        "runs_t3_h05_res_balance_bw/checkpoints/best_nmse.pth,"
        "DU_IIBTD_res_Sr/"
        "runs_t3_h06_res_balance_bw/checkpoints/best_nmse.pth"
    )
    iibtd_du_min_sensors_for_update: int = 6
    iibtd_du_update_batch_size: int = 0           # <=0 consumes each fused update in one batch


@dataclass
class MAPPOConfig:
    """MAPPO algorithm hyperparameters."""
    # Training
    num_envs: int = 5                              # One rollout worker per DPM100PSD scene
    total_timesteps: int = 120_000                     # 120 updates with 200 steps x 5 scenes
    episode_max_steps: int = 200                   # Maximum steps per episode
    rollout_length: int = 200                      # Must match episode_max_steps
    num_minibatches: int = 4                       # Number of minibatches for PPO update
    num_epochs: int = 4                           # PPO epochs per update
    
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
    device: str = "cuda"                           # "cuda" or "cpu"; runtime prefers cuda:1 when available
    vec_backend: str = "subproc"                     # "sync" or "subproc" rollout backend
    save_interval: int = 50                        # Save model every N updates
    log_interval: int = 20                         # Log metrics every N updates
    eval_interval: int = 500                     # Evaluate every N updates
    eval_episodes: int = 10                        # Number of evaluation episodes
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
        self.scene.radioseer_root = str(self.scene.radioseer_root).strip() or "RadioSeerDPM100PSD"
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
        self.uav.quant_bits = [int(bits) for bits in self.uav.quant_bits]
        _ensure(len(self.uav.quant_bits) > 0, "uav.quant_bits must not be empty")
        invalid_quant_bits = [bits for bits in self.uav.quant_bits if bits <= 0]
        _ensure(
            not invalid_quant_bits,
            "uav.quant_bits must all be positive integers, got "
            f"{invalid_quant_bits}",
        )
        _ensure(
            len(set(self.uav.quant_bits)) == len(self.uav.quant_bits),
            "uav.quant_bits must not contain duplicate entries, got "
            f"{self.uav.quant_bits}",
        )
        self.uav.default_quant_bits = int(self.uav.default_quant_bits)
        _ensure(
            self.uav.default_quant_bits in self.uav.quant_bits,
            "uav.default_quant_bits must be one of uav.quant_bits, got "
            f"{self.uav.default_quant_bits} not in {self.uav.quant_bits}",
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

        def _coerce_step_size(value, name: str) -> int:
            value_float = float(value)
            _ensure(np.isfinite(value_float), f"{name}.step_size must be finite, got {value}")
            _ensure(
                value_float >= 1.0 and value_float.is_integer(),
                f"{name}.step_size must be an integer grid-cell count >= 1, got {value}",
            )
            return int(value_float)

        self.uav.step_size = _coerce_step_size(self.uav.step_size, "uav")
        self.ugv.step_size = _coerce_step_size(self.ugv.step_size, "ugv")

        self.reward.q_ref = float(self.reward.q_ref)
        _ensure(
            self.reward.q_ref > 0.0,
            f"reward.q_ref must be positive, got {self.reward.q_ref}",
        )

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
        self.planner.hybrid_stall_update_count = int(self.planner.hybrid_stall_update_count)
        self.planner.hybrid_stall_nmse_threshold = float(self.planner.hybrid_stall_nmse_threshold)
        self.planner.hybrid_global_hold_intervals = int(self.planner.hybrid_global_hold_intervals)
        self.planner.hybrid_local_min_candidate_count = int(self.planner.hybrid_local_min_candidate_count)
        self.planner.prefill_percent = float(self.planner.prefill_percent)
        self.planner.prefill_budget_basis = int(self.planner.prefill_budget_basis)
        self.planner.init_building_clearance = int(self.planner.init_building_clearance)
        self.planner.bootstrap_building_clearance = int(self.planner.bootstrap_building_clearance)
        self.planner.flush_reconstruction_on_episode_end = bool(
            self.planner.flush_reconstruction_on_episode_end
        )
        self.planner.ensemble_refresh_interval = int(self.planner.ensemble_refresh_interval)
        self.planner.ensemble_size = int(self.planner.ensemble_size)
        self.planner.ensemble_full_refresh_interval = int(self.planner.ensemble_full_refresh_interval)
        self.planner.nmse_refresh_delta = float(self.planner.nmse_refresh_delta)
        self.planner.incremental_outer_iters = int(self.planner.incremental_outer_iters)
        self.planner.incremental_max_svt_iters = int(self.planner.incremental_max_svt_iters)
        self.planner.ensemble_quality_weighted = bool(self.planner.ensemble_quality_weighted)
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
            self.planner.local_planner_radius > 0,
            "planner.local_planner_radius must be positive, got "
            f"{self.planner.local_planner_radius}",
        )
        _ensure(
            self.planner.hybrid_stall_update_count > 0,
            "planner.hybrid_stall_update_count must be positive, got "
            f"{self.planner.hybrid_stall_update_count}",
        )
        _ensure(
            self.planner.hybrid_stall_nmse_threshold >= 0.0,
            "planner.hybrid_stall_nmse_threshold must be >= 0, got "
            f"{self.planner.hybrid_stall_nmse_threshold}",
        )
        _ensure(
            self.planner.hybrid_global_hold_intervals > 0,
            "planner.hybrid_global_hold_intervals must be positive, got "
            f"{self.planner.hybrid_global_hold_intervals}",
        )
        _ensure(
            self.planner.hybrid_local_min_candidate_count > 0,
            "planner.hybrid_local_min_candidate_count must be positive, got "
            f"{self.planner.hybrid_local_min_candidate_count}",
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
        self.mappo.episode_max_steps = int(self.mappo.episode_max_steps)
        self.mappo.rollout_length = int(self.mappo.rollout_length)
        if self.mappo.episode_max_steps <= 0:
            raise ValueError(
                f"mappo.episode_max_steps must be positive, got {self.mappo.episode_max_steps}"
            )
        if self.mappo.rollout_length <= 0:
            raise ValueError(
                f"mappo.rollout_length must be positive, got {self.mappo.rollout_length}"
            )
        if int(self.mappo.num_epochs) <= 0:
            raise ValueError(
                f"mappo.num_epochs must be positive, got {self.mappo.num_epochs}"
            )
        if self.planner.ensemble_refresh_interval <= 0:
            raise ValueError(
                "planner.ensemble_refresh_interval must be positive, got "
                f"{self.planner.ensemble_refresh_interval}"
            )
        _ensure(
            self.planner.ensemble_size > 0,
            f"planner.ensemble_size must be positive, got {self.planner.ensemble_size}",
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

        backend_key = str(self.planner.iibtd_backend).strip().lower() or "du_iibtd_sr"
        backend_aliases = {
            "du_iibtd_sr": "du_iibtd_sr",
            "du-iibtd-sr": "du_iibtd_sr",
            "du_iibtd_res_sr": "du_iibtd_sr",
            "du-iibtd-res-sr": "du_iibtd_sr",
            "du_iibtd_sr_learn_nu": "du_iibtd_sr_learn_nu",
            "du-iibtd-sr-learn-nu": "du_iibtd_sr_learn_nu",
            "du_iibtd_res_sr_learn_nu": "du_iibtd_sr_learn_nu",
            "du-iibtd-res-sr-learn-nu": "du_iibtd_sr_learn_nu",
        }
        self.planner.iibtd_backend = backend_aliases.get(backend_key, backend_key)
        if self.planner.iibtd_backend not in {"du_iibtd_sr", "du_iibtd_sr_learn_nu", "gpu"}:
            raise ValueError(
                "planner.iibtd_backend must be one of "
                "du_iibtd_sr/du_iibtd_sr_learn_nu/gpu; got "
                f"{self.planner.iibtd_backend!r}. The old cpu/auto/du_iibtd backends are disabled."
            )

        self.planner.iibtd_device = str(self.planner.iibtd_device).strip() or "auto"
        self.planner.iibtd_du_checkpoint_path = str(
            self.planner.iibtd_du_checkpoint_path
        ).strip()
        if self.planner.iibtd_backend in {"du_iibtd_sr", "du_iibtd_sr_learn_nu"}:
            _ensure(
                bool(self.planner.iibtd_du_checkpoint_path),
                "planner.iibtd_du_checkpoint_path must be set for DU-IIBTD SR backends",
            )
        self.planner.iibtd_du_min_sensors_for_update = int(
            self.planner.iibtd_du_min_sensors_for_update
        )
        _ensure(
            self.planner.iibtd_du_min_sensors_for_update > 0,
            "planner.iibtd_du_min_sensors_for_update must be positive, got "
            f"{self.planner.iibtd_du_min_sensors_for_update}",
        )
        self.planner.iibtd_du_update_batch_size = int(self.planner.iibtd_du_update_batch_size)
        _ensure(
            self.planner.iibtd_du_update_batch_size >= 0,
            "planner.iibtd_du_update_batch_size must be >= 0, got "
            f"{self.planner.iibtd_du_update_batch_size}",
        )

        self.planner.iibtd_gpu_phi_solver = (
            str(self.planner.iibtd_gpu_phi_solver).strip().lower() or "scipy"
        )
        if self.planner.iibtd_gpu_phi_solver not in {"scipy", "pgd"}:
            raise ValueError(
                "planner.iibtd_gpu_phi_solver must be one of scipy/pgd, got "
                f"{self.planner.iibtd_gpu_phi_solver!r}"
            )
