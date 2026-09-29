"""
Shared quant configuration for UAV-UGV cooperative spectrum sensing.

All hyperparameters and environment settings are centralized here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os
import sys
import numpy as np

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from du_iibtd_based_fading_delta.du_iibtd_learn_nu import (
    DEFAULT_DU_IIBTD_CHECKPOINTS,
    DU_IIBTD_BACKENDS,
    default_du_iibtd_checkpoints_for_backend,
)

DEFAULT_RADIOSEER_ROOT = "RadioSeerDPM100PSD"
DEFAULT_RADIOSEER_SAMPLE_INDEX = 8513
DEFAULT_RADIOSEER_SCENE_INDICES = [8513, 1807, 1579, 1371, 10001, "FARMOmniDPM100PSD_251:905", "FARMOmniDPM100PSD_251:43", "FARMOmniDPM100PSD_251:705"]


@dataclass
class SceneConfig:
    """City scene and grid configuration."""
    grid_size: Tuple[int, int] = (1, 1)         # Placeholder only; overwritten from loaded RadioSeer sample shape
    grid_spacing: float = 2.0                  # Meters per cell used by motion, distance, and SNR calculations
    uav_height: float = 50.0                      # Fixed UAV flight altitude (m)
    ugv_height: float = 0.0                       # UGV antenna height above ground (m)
    building_height_m: float = 47.0              # Height for fixed mode (m); saved configs retain their value
    building_height_mode: str = "uniform_integer_per_building"
    building_height_min_m: float = 47.0
    building_height_max_m: float = 53.0
    building_height_seed: int = 42               # Geometry seed, independent of episode RNG
    building_split_min_area_m2: float = 1200.0   # Split a large connected footprint once; 0 disables
    scene_source: str = "radioseerselect"        # Loader family tag for manifest-compatible RadioSeer datasets
    radioseer_root: str = DEFAULT_RADIOSEER_ROOT   # Dataset folder relative to repo root or absolute path
    radioseer_sample_index: int = DEFAULT_RADIOSEER_SAMPLE_INDEX  # Selected manifest row in the configured RadioSeer dataset
    radioseer_scene_indices: List[object] = field(default_factory=lambda: list(DEFAULT_RADIOSEER_SCENE_INDICES))

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
    max_energy: float = 8_500                      # Available energy for the abstract simulation (J)
    flight_power: float = 12.0                     # Power consumption during flight (W)
    hover_power: float = 8.0                      # Power consumption during hover (W)
    sensing_power: float = 5.0                     # Sensing power at the largest selectable band count (W)
    step_duration: float = 1.0                     # Duration per grid hop / hover step (seconds)

    # Bandwidth
    total_bandwidth: float = 50e6               # Total RF bandwidth (Hz)
    total_bw_num: int = 12                        # Total discrete bandwidth units
    default_bw_ratio: float = 0.6                 # Default sensing bandwidth ratio
    queue_capacity_bits: float = 512e6            # Hard UAV queue capacity (bits)

    bandwidth_ratios: List[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.6]  # Discrete sensing bandwidth ratios
    )
    quant_bits: List[int] = field(
        default_factory=lambda: [10, 8, 6]         # Discrete numerical quantization bits
    )
    default_quant_bits: int = 10                   # Initial UAV quantization bit depth
    quantization_scheme: str = "log_first"         # log_first | raw; raw quantizes the linear power domain
    fixed_quant_min: float = 0.0                   # Dataset-wide lower quantizer endpoint
    fixed_quant_max: float = 8.75                  # Dataset-wide upper quantizer endpoint (above max + 6 sigma)
    quant_log_offset: float = 1e-6                 # Positive offset before log-domain quantization

    # Observation disturbance (IIBTD measurement model)
    small_scale_fading_std: float = 0.01            # Std. of eta_{r,t}^{(k)} before spectral weighting
    receiver_noise_std: float = 0.01                # Std. of epsilon_t^{(k)}

    @property
    def num_bandwidth_ratios(self) -> int:
        return len(self.bandwidth_ratios)

    def sensing_units_for_ratio(self, ratio: float) -> int:
        """Return the realized discrete sensing-band count for a split ratio."""
        total_units = max(1, int(self.total_bw_num))
        if total_units == 1:
            return 1
        return int(np.clip(np.ceil(total_units * float(ratio)), 1, total_units - 1))

    def sensing_power_for_units(self, sensing_units: int) -> float:
        """Scale sensing power by normalized n*log2(1+n) processing load."""
        units = max(0, int(sensing_units))
        if units == 0:
            return 0.0
        max_units = max(
            self.sensing_units_for_ratio(ratio) for ratio in self.bandwidth_ratios
        )
        load = float(units) * float(np.log2(1.0 + float(units)))
        max_load = float(max_units) * float(np.log2(1.0 + float(max_units)))
        return float(self.sensing_power) * load / max(max_load, 1e-12)

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
    step_size: float = 5.0                        # Movement distance per step in grid cells


@dataclass
class CommConfig:
    """Communication channel configuration."""
    carrier_freq: float = 3.5e9                    # Communication carrier frequency (Hz)
    tx_power_dbm: float = 12.0                     # Initial RF output power (dBm)
    tx_power_choices_dbm: List[float] = field(default_factory=lambda: [9.0, 12.0, 15.0])
    tx_energy_enabled: bool = True                # Ideal transmit energy: P_tx * t_tx
    noise_figure_db: float = 8.0                   # Receiver noise figure (dB)
    source_measurement_bits: int = 32              # Uncompressed payload bit-width reference before policy-selected quantization
    data_per_sample: float = 8e6                   # Fixed raw payload per band/window (bits), referenced to 32-bit PSD values
    shadow_std_los_db: float = 2.0                 # Log-normal shadow std when LoS (dB)
    shadow_std_nlos_db: float = 6.0                # Log-normal shadow std when NLoS (dB)
    los_excess_db: float = 1.6                    # Dense urban LoS mean loss above FSPL (dB)
    nlos_excess_db: float = 23.0                  # NLoS base excess loss before length correction (dB)
    nlos_length_loss_db_per_m: float = 0.22       # Exploratory horizontal obstruction-length slope
    nlos_length_loss_cap_db: Optional[float] = None  # None: no cap; 0: disable correction; positive: explicit legacy cap
    snr_outage_threshold_db: float = -5.0          # Minimum SNR required for link service
    outage_snr_mode: str = "received"              # received | nominal (historical reproduction only)


@dataclass
class RewardConfig:
    """Reward function coefficients."""
    alpha_nmse: float = 20.0                       # Weight for normalized NMSE-improvement reward (delta_nmse_norm)
    nmse_signed_clip: float = 0.25                # Clip signed normalized NMSE delta before scaling; <=0 disables clipping
    gamma_queue: float = 1.5                      # Weight for queue-bits penalty
    lambda_uav_progress: float = 2.0              # Reward for UAV moving closer to active local target
    lambda_uav_backtrack: float = 2.0             # Penalty for UAV moving away from active target
    lambda_ugv_progress: float = 2.0              # Reward for UGV moving closer to its guidance target
    lambda_ugv_backtrack: float = 2.0             # Penalty for UGV moving away from its guidance target
    lambda_novel_info: float = 0.0                # Reward for the fraction of sampled frequency bands that are newly observed
    lambda_full_repeat: float = 0.0               # Penalty when a valid sample contains no newly observed frequency band
    bootstrap_progress_scale: float = 1.0        # Keep bootstrap targets reward-aligned with the observation target encoding
    accuracy_target_nmse: float = 0.08             # Target NMSE tracked for diagnostics; does not end episodes


@dataclass
class ObservationConfig:
    """Observation space configuration."""
    # Planner-aware features for Critic
    num_planner_features: int = 5                 # target(x,y), center_freq, dist_uav, dist_ugv
    include_remaining_time: bool = False          # Deprecated compatibility field; not included in observations
    include_quant_context: bool = False           # Deprecated compatibility field; not included in observations
    ugv_building_safe_clearance: int = 3          # Normalization radius for UGV building-clearance observations


@dataclass
class PlannerConfig:
    """UGV-side active-planner configuration."""
    target_count: int = 1                          # Number of planner targets exposed each cycle
    target_mode: str = "hybrid"                     # Planner target scope: local | global | hybrid
    initial_observation_mode: str = "prefill"    # Warmup mode before planner: bootstrap | prefill
    local_planner_radius: int = 15                  # Local targets stay within four 4-cell UAV actions (Manhattan radius)
    hybrid_uncertainty_window_updates: int = 2        # Number of signed relative improvements in the rolling sum
    hybrid_uncertainty_improvement_threshold: float = 0.03  # Switch when the rolling signed improvement sum is below this value
    hybrid_global_hold_intervals: int = 5          # In hybrid mode, hold global submode for this many ensemble intervals
    hybrid_local_reentry_min_targets: int = 2      # Effective local reentry threshold is max(this value, target_count)
    ugv_comm_target_mode: str = "path_corridor"    # Frozen continuous communication-support target rule
    ugv_comm_backlog_threshold: float = 0.5        # Urgent recovery threshold as normalized remaining queue bits
    ugv_comm_local_path_horizon: int = 15          # Normal A*-guided road-path horizon
    ugv_comm_expanded_path_horizon: int = 20       # Expanded path horizon after the local corridor has no feasible cell
    ugv_comm_corridor_width: int = 1               # Side-road width around the A*-guided path used for link checks
    ugv_recovery_exit_backlog_threshold: float = 0.2  # Hysteresis threshold for returning to two-path support
    ugv_recovery_poor_service_steps: int = 2       # High-backlog service deficits required before recovery
    ugv_recovery_min_hold_steps: int = 2           # Minimum Support recovery duration
    ugv_recovery_good_link_steps: int = 2          # Consecutive healthy steps required to leave recovery
    ugv_recovery_service_margin: float = 1.0       # Required service/arrival ratio for entry/exit decisions
    ugv_service_action_mask: bool = False           # Optional ablation; default policy mask is collision-only
    ugv_control_mode: str = "policy"               # policy | fixed | legacy_heuristic | astar_target | astar_support | astar_2path_support | astar_2path_support_recovery
    prefill_percent: float = 5                      # Prefill observation windows as a percentage of the observation-action budget
    prefill_budget_basis: int = 200                # Frozen latest-complete observation-window budget
    init_pair_max_distance: float = 7.0            # Max initial UAV-UGV separation in grid units
    init_building_clearance: int = 5               # Prefer initial UAV/UGV cells with this many grid cells of building clearance
    bootstrap_building_clearance: int = 5          # Prefer bootstrap targets with this many grid cells of building clearance
    flush_reconstruction_on_episode_end: bool = False  # Force one last expensive reconstruction at terminal steps; keep off for faster training
    target_arrival_radius_steps: float = 1.0       # Treat sampled UAV positions within this many UAV steps of target as arrival
    target_suppression_radius_steps: float = 1.0   # Suppress completed/stuck target neighborhoods until the next ensemble refresh
    target_stuck_no_sample_steps: int = 4          # Retarget after this many consecutive active-plan steps without a valid UAV sample

    ensemble_refresh_interval: int = 3            # Refresh ensemble mean/variance and planner targets after this many newly delivered samples

    min_samples_for_ensemble: int = 1             # Minimum effective sample count; no extra floor beyond non-empty observations

    # ShareMem ensemble
    ensemble_size: int = 3
    ensemble_quality_weighted: bool = True         # Weight ensemble members by observed-entry NMSE
    ensemble_init_jitter_scale: float = 1e-2       # Tiny per-member state jitter for diversity
    hybrid_switch_metric: str = "sum_relative_frobenius"
    reconstruction_refresh_mode: str = "local_to_global"
    incremental_outer_iters: int = 2               # Outer iterations for fit_incremental between full refreshes
    incremental_max_svt_iters: int = 20            # Max SVT iterations for incremental solver updates

    # Acquisition weights
    lambda_u: float = 1.0
    beta_f: float = 0.3
    redundancy_length: float = 20.0             # Visualization-only spacing for displayed global top-k markers

    # DU-IIBTD reconstruction backend
    iibtd_mu: float = 1e-1                        # DU-IIBTD runtime penalty parameter mu
    iibtd_backend: str = "du_iibtd_res_sr_learn_nu"               # du_iibtd | du_iibtd_res_sr | du_iibtd_res_sr_learn_nu
    iibtd_device: str = "auto"                    # auto | cuda | cuda:0 | cpu; may differ from the policy-optimization device
    du_iibtd_checkpoints: List[str] = field(
        default_factory=lambda: list(DEFAULT_DU_IIBTD_CHECKPOINTS)
    )
    du_iibtd_min_sensors_for_update: int = 0      # <=0 lets the adapter use checkpoint config
    du_iibtd_update_batch_size: int = 0           # <=0 lets the adapter use checkpoint config


@dataclass
class OnPolicyConfig:
    """Shared on-policy optimization and rollout hyperparameters."""
    # Training
    num_envs: int = 8                              # One rollout worker per scene suite entry
    total_timesteps: int = 192_000                    # 120 updates with 200 steps x 8 scene entries; 960 episodes total
    episode_max_steps: int = 200                   # Episode horizon and rollout length
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
    device: str = "cuda:0"                        # "cuda" or "cpu"; bare cuda resolves to cuda:0
    vec_backend: str = "subproc"                     # "sync" or "subproc" rollout backend
    save_interval: int = 10                        # Save model every N updates
    log_interval: int = 10                         # Log metrics every N updates
    eval_interval: int = 10                        # Evaluate every N updates
    eval_episodes: int = 1                         # Number of evaluation episodes
    eval_seed_stride: int = 10_000                 # Frozen latest-complete evaluation seed stride
    model_dir: str = "checkpoints"
    log_dir: str = "logs"


MAPPOConfig = OnPolicyConfig


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
    # ``mappo`` is retained as the serialized compatibility key used by
    # existing checkpoints. New algorithm-neutral code should use ``training``.
    mappo: OnPolicyConfig = field(default_factory=OnPolicyConfig)

    @property
    def training(self) -> OnPolicyConfig:
        return self.mappo

    @training.setter
    def training(self, value: OnPolicyConfig) -> None:
        self.mappo = value

    def __post_init__(self):
        """Validate configuration consistency."""
        def _ensure(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(message)

        self.reward.lambda_novel_info = float(self.reward.lambda_novel_info)
        self.reward.lambda_full_repeat = float(self.reward.lambda_full_repeat)
        _ensure(
            self.reward.lambda_novel_info >= 0.0,
            "reward.lambda_novel_info must be non-negative, got "
            f"{self.reward.lambda_novel_info}",
        )
        _ensure(
            self.reward.lambda_full_repeat >= 0.0,
            "reward.lambda_full_repeat must be non-negative, got "
            f"{self.reward.lambda_full_repeat}",
        )

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
        from ...building_heights import validate_building_height_config
        validate_building_height_config(self.scene)
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
        self.uav.queue_capacity_bits = float(self.uav.queue_capacity_bits)
        _ensure(
            np.isfinite(self.uav.queue_capacity_bits)
            and self.uav.queue_capacity_bits > 0.0,
            "uav.queue_capacity_bits must be finite and positive, got "
            f"{self.uav.queue_capacity_bits}",
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
        self.uav.quantization_scheme = (
            str(self.uav.quantization_scheme).strip().lower()
        )
        _ensure(
            self.uav.quantization_scheme in {"log_first", "raw"},
            "uav.quantization_scheme must be 'log_first' or 'raw', got "
            f"{self.uav.quantization_scheme!r}",
        )
        self.uav.fixed_quant_min = float(self.uav.fixed_quant_min)
        self.uav.fixed_quant_max = float(self.uav.fixed_quant_max)
        self.uav.quant_log_offset = float(self.uav.quant_log_offset)
        self.uav.small_scale_fading_std = float(self.uav.small_scale_fading_std)
        self.uav.receiver_noise_std = float(self.uav.receiver_noise_std)
        _ensure(
            np.isfinite(self.uav.fixed_quant_min)
            and np.isfinite(self.uav.fixed_quant_max)
            and self.uav.fixed_quant_max > self.uav.fixed_quant_min >= 0.0,
            "uav fixed quantization endpoints must satisfy 0 <= min < max, got "
            f"{self.uav.fixed_quant_min}, {self.uav.fixed_quant_max}",
        )
        _ensure(
            np.isfinite(self.uav.quant_log_offset) and self.uav.quant_log_offset > 0.0,
            f"uav.quant_log_offset must be finite and positive, got {self.uav.quant_log_offset}",
        )
        _ensure(
            np.isfinite(self.uav.small_scale_fading_std)
            and self.uav.small_scale_fading_std >= 0.0,
            "uav.small_scale_fading_std must be finite and non-negative, got "
            f"{self.uav.small_scale_fading_std}",
        )
        _ensure(
            np.isfinite(self.uav.receiver_noise_std)
            and self.uav.receiver_noise_std >= 0.0,
            "uav.receiver_noise_std must be finite and non-negative, got "
            f"{self.uav.receiver_noise_std}",
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

        self.comm.source_measurement_bits = int(self.comm.source_measurement_bits)
        _ensure(
            self.comm.source_measurement_bits > 0,
            "comm.source_measurement_bits must be positive, got "
            f"{self.comm.source_measurement_bits}",
        )
        _ensure(
            max(self.uav.quant_bits) <= self.comm.source_measurement_bits,
            "all uav.quant_bits must be <= comm.source_measurement_bits, got "
            f"{self.uav.quant_bits} vs {self.comm.source_measurement_bits}",
        )
        self.comm.data_per_sample = float(self.comm.data_per_sample)
        _ensure(
            np.isfinite(self.comm.data_per_sample)
            and self.comm.data_per_sample > 0.0,
            "comm.data_per_sample must be finite and positive, got "
            f"{self.comm.data_per_sample}",
        )
        for loss_name in ("los_excess_db", "nlos_excess_db",
                          "nlos_length_loss_db_per_m"):
            loss = float(getattr(self.comm, loss_name))
            _ensure(np.isfinite(loss) and loss >= 0.0,
                    f"comm.{loss_name} must be finite and non-negative, got {loss}")
            setattr(self.comm, loss_name, loss)
        if self.comm.nlos_length_loss_cap_db is not None:
            cap = float(self.comm.nlos_length_loss_cap_db)
            _ensure(np.isfinite(cap) and cap >= 0.0,
                    "comm.nlos_length_loss_cap_db must be None or finite and non-negative")
            self.comm.nlos_length_loss_cap_db = cap
        self.comm.outage_snr_mode = str(self.comm.outage_snr_mode).strip().lower()
        _ensure(
            self.comm.outage_snr_mode in {"received", "nominal"},
            "comm.outage_snr_mode must be received or nominal",
        )
        self.comm.snr_outage_threshold_db = float(self.comm.snr_outage_threshold_db)
        _ensure(
            np.isfinite(self.comm.snr_outage_threshold_db),
            "comm.snr_outage_threshold_db must be finite, got "
            f"{self.comm.snr_outage_threshold_db}",
        )

        self.obs.ugv_building_safe_clearance = int(self.obs.ugv_building_safe_clearance)
        _ensure(
            self.obs.ugv_building_safe_clearance >= 1,
            "obs.ugv_building_safe_clearance must be >= 1, got "
            f"{self.obs.ugv_building_safe_clearance}",
        )
        self.obs.num_planner_features = int(self.obs.num_planner_features)
        self.obs.include_remaining_time = bool(self.obs.include_remaining_time)
        self.obs.include_quant_context = bool(self.obs.include_quant_context)
        _ensure(
            self.obs.num_planner_features == 5,
            "obs.num_planner_features must be 5 after removing planner score from critic obs, got "
            f"{self.obs.num_planner_features}",
        )

        self.planner.target_count = int(self.planner.target_count)
        self.planner.target_mode = str(self.planner.target_mode).strip().lower() or "hybrid"
        self.planner.initial_observation_mode = (
            str(self.planner.initial_observation_mode).strip().lower() or "bootstrap"
        )
        self.planner.local_planner_radius = int(self.planner.local_planner_radius)
        self.planner.hybrid_uncertainty_window_updates = int(self.planner.hybrid_uncertainty_window_updates)
        self.planner.hybrid_uncertainty_improvement_threshold = float(self.planner.hybrid_uncertainty_improvement_threshold)
        self.planner.hybrid_global_hold_intervals = int(self.planner.hybrid_global_hold_intervals)
        self.planner.hybrid_local_reentry_min_targets = int(self.planner.hybrid_local_reentry_min_targets)
        self.planner.ugv_comm_target_mode = (
            str(self.planner.ugv_comm_target_mode).strip().lower()
            or "path_corridor"
        )
        self.planner.ugv_comm_backlog_threshold = float(
            self.planner.ugv_comm_backlog_threshold
        )
        self.planner.ugv_comm_local_path_horizon = int(
            self.planner.ugv_comm_local_path_horizon
        )
        self.planner.ugv_comm_expanded_path_horizon = int(
            self.planner.ugv_comm_expanded_path_horizon
        )
        self.planner.ugv_comm_corridor_width = int(
            self.planner.ugv_comm_corridor_width
        )
        self.planner.ugv_recovery_exit_backlog_threshold = float(
            self.planner.ugv_recovery_exit_backlog_threshold
        )
        self.planner.ugv_recovery_poor_service_steps = int(
            self.planner.ugv_recovery_poor_service_steps
        )
        self.planner.ugv_recovery_min_hold_steps = int(
            self.planner.ugv_recovery_min_hold_steps
        )
        self.planner.ugv_recovery_good_link_steps = int(
            self.planner.ugv_recovery_good_link_steps
        )
        self.planner.ugv_recovery_service_margin = float(
            self.planner.ugv_recovery_service_margin
        )
        self.planner.ugv_service_action_mask = bool(
            self.planner.ugv_service_action_mask
        )
        self.planner.ugv_control_mode = str(
            self.planner.ugv_control_mode
        ).strip().lower()
        if self.planner.ugv_control_mode == "astar":
            self.planner.ugv_control_mode = "astar_target"
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
            self.planner.hybrid_uncertainty_window_updates == 2,
            "planner.hybrid_uncertainty_window_updates must be 2, got "
            f"{self.planner.hybrid_uncertainty_window_updates}",
        )
        _ensure(
            self.planner.hybrid_uncertainty_improvement_threshold >= 0.0,
            "planner.hybrid_uncertainty_improvement_threshold must be >= 0, got "
            f"{self.planner.hybrid_uncertainty_improvement_threshold}",
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
            self.planner.ugv_comm_target_mode == "path_corridor",
            "planner.ugv_comm_target_mode is frozen to 'path_corridor', got "
            f"{self.planner.ugv_comm_target_mode!r}",
        )
        _ensure(
            0.0 < self.planner.ugv_comm_backlog_threshold <= 1.0,
            "planner.ugv_comm_backlog_threshold must be within (0, 1], got "
            f"{self.planner.ugv_comm_backlog_threshold}",
        )
        _ensure(
            self.planner.ugv_comm_local_path_horizon > 0,
            "planner.ugv_comm_local_path_horizon must be positive, got "
            f"{self.planner.ugv_comm_local_path_horizon}",
        )
        _ensure(
            self.planner.ugv_comm_expanded_path_horizon
            >= self.planner.ugv_comm_local_path_horizon,
            "planner.ugv_comm_expanded_path_horizon must be >= "
            "ugv_comm_local_path_horizon, got "
            f"{self.planner.ugv_comm_expanded_path_horizon} < "
            f"{self.planner.ugv_comm_local_path_horizon}",
        )
        _ensure(
            self.planner.ugv_comm_corridor_width >= 0,
            "planner.ugv_comm_corridor_width must be non-negative, got "
            f"{self.planner.ugv_comm_corridor_width}",
        )
        _ensure(
            0.0 <= self.planner.ugv_recovery_exit_backlog_threshold
            < self.planner.ugv_comm_backlog_threshold,
            "planner.ugv_recovery_exit_backlog_threshold must be in [0, "
            "ugv_comm_backlog_threshold), got "
            f"{self.planner.ugv_recovery_exit_backlog_threshold}",
        )
        _ensure(
            self.planner.ugv_recovery_poor_service_steps > 0,
            "planner.ugv_recovery_poor_service_steps must be positive, got "
            f"{self.planner.ugv_recovery_poor_service_steps}",
        )
        _ensure(
            self.planner.ugv_recovery_min_hold_steps > 0,
            "planner.ugv_recovery_min_hold_steps must be positive, got "
            f"{self.planner.ugv_recovery_min_hold_steps}",
        )
        _ensure(
            self.planner.ugv_recovery_good_link_steps > 0,
            "planner.ugv_recovery_good_link_steps must be positive, got "
            f"{self.planner.ugv_recovery_good_link_steps}",
        )
        _ensure(
            self.planner.ugv_recovery_service_margin >= 1.0,
            "planner.ugv_recovery_service_margin must be >= 1, got "
            f"{self.planner.ugv_recovery_service_margin}",
        )
        _ensure(
            self.planner.ugv_control_mode
            in {
                "policy",
                "fixed",
                "legacy_heuristic",
                "astar_target",
                "astar_support",
                "astar_2path_support",
                "astar_2path_support_recovery",
            },
            "planner.ugv_control_mode must be one of "
            "['astar_target', 'astar_support', 'astar_2path_support', "
            "'astar_2path_support_recovery', "
            "'fixed', 'legacy_heuristic', 'policy'], got "
            f"{self.planner.ugv_control_mode!r}",
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
        self.training.num_envs = int(self.training.num_envs)
        self.training.total_timesteps = int(self.training.total_timesteps)
        self.training.episode_max_steps = int(self.training.episode_max_steps)
        self.training.num_minibatches = int(self.training.num_minibatches)
        self.training.num_epochs = int(self.training.num_epochs)
        self.training.eval_episodes = int(self.training.eval_episodes)
        _ensure(self.training.num_envs > 0, f"mappo.num_envs must be positive, got {self.training.num_envs}")
        _ensure(
            self.training.total_timesteps > 0,
            f"mappo.total_timesteps must be positive, got {self.training.total_timesteps}",
        )
        _ensure(
            self.training.episode_max_steps > 0,
            f"mappo.episode_max_steps must be positive, got {self.training.episode_max_steps}",
        )
        _ensure(
            self.training.num_minibatches > 0,
            f"mappo.num_minibatches must be positive, got {self.training.num_minibatches}",
        )
        _ensure(
            self.training.num_epochs > 0,
            f"mappo.num_epochs must be positive, got {self.training.num_epochs}",
        )
        _ensure(
            self.training.eval_episodes > 0,
            f"mappo.eval_episodes must be positive, got {self.training.eval_episodes}",
        )
        self.uav.max_energy = float(self.uav.max_energy)
        self.uav.flight_power = float(self.uav.flight_power)
        self.uav.hover_power = float(self.uav.hover_power)
        self.uav.sensing_power = float(self.uav.sensing_power)
        self.uav.step_duration = float(self.uav.step_duration)
        for field_name in (
            "max_energy",
            "flight_power",
            "hover_power",
            "sensing_power",
            "step_duration",
        ):
            value = float(getattr(self.uav, field_name))
            _ensure(
                np.isfinite(value) and value > 0.0,
                f"uav.{field_name} must be finite and positive, got {value}",
            )
        from ...power_control import validate_power_config
        validate_power_config(self.comm)
        minimum_sensing_units = min(
            self.uav.sensing_units_for_ratio(ratio)
            for ratio in self.uav.bandwidth_ratios
        )
        minimum_sensing_power = self.uav.sensing_power_for_units(
            minimum_sensing_units
        )
        minimum_horizon_energy = (
            float(self.training.episode_max_steps)
            * (self.uav.hover_power + minimum_sensing_power)
            * self.uav.step_duration
        )
        _ensure(
            self.uav.max_energy + 1e-9 >= minimum_horizon_energy,
            "uav.max_energy must cover minimum hover+minimum-band sensing energy for the full "
            f"episode horizon ({minimum_horizon_energy:.3f} J), got {self.uav.max_energy:.3f} J",
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
            self.planner.hybrid_switch_metric == "sum_relative_frobenius"
            and self.planner.reconstruction_refresh_mode == "local_to_global"
            and np.isfinite(self.planner.hybrid_uncertainty_improvement_threshold),
            "This version requires the sum-relative-Frobenius/local-to-global refit protocol",
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
        self.planner.iibtd_device = str(self.planner.iibtd_device).strip() or "auto"
