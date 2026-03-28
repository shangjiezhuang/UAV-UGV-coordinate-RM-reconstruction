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
    grid_size: Tuple[int, int] = (32, 32)       # Spatial grid resolution
    grid_spacing: float = 1.0                    # Meters per grid cell
    uav_height: float = 30.0                      # Fixed UAV flight altitude (m)

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
    num_directions: int = 5                        # stay + 4 cardinal directions
    step_size: float = 3.0                        # Movement distance per step (m), as a grid_spacing multiple

    # Energy
    max_energy: float = 100_00.0                     # Maximum energy budget (Joules)
    flight_power: float = 12.0                     # Power consumption during flight (W)
    hover_power: float = 8.0                      # Power consumption during hover (W)
    sensing_power: float = 5.0                     # Power for spectrum sensing (W)
    step_duration: float = 1.0                     # Duration per step (seconds)

    # Bandwidth
    total_bandwidth: float = 100e6               # Total RF bandwidth (Hz)
    total_bw_num: int = 10                        # Total discrete bandwidth units
    default_bw_ratio: float = 0.6                 # Default sensing bandwidth ratio

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
    step_size: float = 3.0                        # Movement distance per step (m), as a grid_spacing multiple


@dataclass
class CommConfig:
    """Communication channel configuration."""
    carrier_freq: float = 3.5e9                    # Communication carrier frequency (Hz)
    tx_power_dbm: float = 1.0                      # UAV transmit power (dBm)
    noise_figure_db: float = 8.0                   # Receiver noise figure (dB)
    data_per_sample: float = 5e7                   # Payload bits per band at high_quantization_bits
    los_model_a: float = 9.61                      # Probabilistic LoS model parameter a
    los_model_b: float = 0.16                      # Probabilistic LoS model parameter b
    shadow_std_los_db: float = 2.0                 # Log-normal shadow std when LoS (dB)
    shadow_std_nlos_db: float = 6.0                # Log-normal shadow std when NLoS (dB)
    nlos_excess_db: float = 15.0                   # Extra NLoS attenuation (dB)


@dataclass
class RewardConfig:
    """Reward function coefficients."""
    w_nmse: float = 20.0                          # Weight for normalized NMSE improvement
    w_tx: float = 0.6                             # Weight for sensing/transmission bit cost
    w_queue: float = 0.8                          # Weight for queue backlog + drop penalty
    w_comm: float = 0.2                           # Weight for communication quality penalty

    # Compatibility knobs for older configs/scripts.
    # The progress/backtrack/arrival terms below are active in the reward.
    alpha_nmse: float = 20.0
    alpha_unc: float = 0.0
    lambda_new_freq: float = 0.0
    lambda_new_spatial: float = 0.0
    beta_tx: float = 0.0
    gamma_queue: float = 0.8
    lambda_uav_progress: float = 1.0
    lambda_uav_backtrack: float = 1.5
    lambda_ugv_progress: float = 0.5
    lambda_ugv_backtrack: float = 1.0
    bootstrap_progress_scale: float = 0.0
    lambda_spatial_revisit: float = 0.0
    local_goal_arrival_bonus: float = 1.5
    q_ref: float = 5.0                             # Reference queue length for normalization
    accuracy_target_nmse: float = 0.08             # Target NMSE for episode termination
    terminal_success_bonus: float = 30.0          # Bonus when target NMSE is reached
    terminal_failure_penalty: float = 10.0        # Penalty when the episode ends without reaching target NMSE


@dataclass
class ObservationConfig:
    """Observation space configuration."""
    # Planner-aware features for Critic
    num_planner_features: int = 6                 # target(x,y), center_freq, score, dist_uav, dist_ugv


@dataclass
class PlannerConfig:
    """UGV-side active-planner configuration."""
    sensor_budget: int = 160                       # Number of candidate sensors in generated scene
    target_count: int = 1                          # Number of planner targets exposed each cycle
    obs_target_slots: int = 1                      # Number of target slots encoded into UAV observation
    local_planner_radius: int = 9                  # Planner only searches targets within this Manhattan radius of the UAV
    init_pair_max_distance: float = 7.0            # Max initial UAV-UGV separation in grid units
    low_priority_process_interval: int = 1         # Process low-priority delivered samples every N steps

    ensemble_refresh_interval: int = 3            # Refresh ensemble mean/variance and planner targets after this many newly delivered samples

    min_samples_for_ensemble: int = 10            # Start planner/ensemble after enough UGV-side effective samples

    # Ensemble resampling
    ensemble_size: int = 4 
    ensemble_keep_ratio: float = 0.85
    ensemble_keep_recent: int = 2

    # Acquisition weights
    lambda_u: float = 1.0
    beta_f: float = 0.3
    redundancy_length: float = 5.0

    # Quantization control
    adaptive_quantization_bits: bool = False      # Use fixed 8-bit quantization by default
    default_quantization_bits: int = 8
    high_quantization_bits: int = 8
    low_quantization_bits: int = 8
    uncertainty_quantization_threshold: float = 0.5

    # II-BTD reconstruction backend
    iibtd_backend: str = "auto"                   # auto | cpu | gpu
    iibtd_device: str = "auto"                    # auto | cuda | cuda:0 | cpu; runtime prefers cuda:2 when available
    iibtd_gpu_phi_solver: str = "scipy"           # scipy | pgd


@dataclass
class MAPPOConfig:
    """MAPPO algorithm hyperparameters."""
    # Training
    num_envs: int = 4                              # Number of parallel environments
    total_timesteps: int = 24_000 * 7                  # Total training timesteps
    episode_max_steps: int = 128                   # Maximum steps per episode
    rollout_length: int = 128                      # Steps per rollout before update
    num_minibatches: int = 4                       # Number of minibatches for PPO update
    num_epochs: int = 4                           # PPO epochs per update
    
    # Optimization
    lr_actor: float = 2e-4                         # Actor learning rate
    lr_critic: float = 2e-4                        # Critic learning rate
    gamma: float = 0.99                            # Discount factor
    gae_lambda: float = 0.95                       # GAE lambda
    clip_epsilon: float = 0.2                      # PPO clipping parameter
    max_grad_norm: float = 0.5                     # Gradient clipping norm
    entropy_coef: float = 0.01                     # Entropy bonus coefficient
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
    log_interval: int = 10                         # Log metrics every N updates
    eval_interval: int = 50                        # Evaluate every N updates
    eval_episodes: int = 5                         # Number of evaluation episodes
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
        grid_spacing = float(self.scene.grid_spacing)
        assert grid_spacing > 0.0, "scene.grid_spacing must be positive"
        uav_ratio = float(self.uav.step_size) / grid_spacing
        ugv_ratio = float(self.ugv.step_size) / grid_spacing
        assert self.uav.step_size > 0.0, "uav.step_size must be positive"
        assert self.ugv.step_size > 0.0, "ugv.step_size must be positive"
        assert np.isclose(uav_ratio, round(uav_ratio)), \
            "uav.step_size must be an integer multiple of grid_spacing"
        assert np.isclose(ugv_ratio, round(ugv_ratio)), \
            "ugv.step_size must be an integer multiple of grid_spacing"
        if int(self.mappo.num_epochs) <= 0:
            raise ValueError(
                f"mappo.num_epochs must be positive, got {self.mappo.num_epochs}"
            )
        if int(self.planner.ensemble_refresh_interval) <= 0:
            raise ValueError(
                "planner.ensemble_refresh_interval must be positive, got "
                f"{self.planner.ensemble_refresh_interval}"
            )

        self.planner.iibtd_backend = str(self.planner.iibtd_backend).strip().lower() or "auto"
        if self.planner.iibtd_backend not in {"auto", "cpu", "gpu"}:
            raise ValueError(
                "planner.iibtd_backend must be one of auto/cpu/gpu, got "
                f"{self.planner.iibtd_backend!r}"
            )

        self.planner.iibtd_device = str(self.planner.iibtd_device).strip() or "auto"

        self.planner.iibtd_gpu_phi_solver = (
            str(self.planner.iibtd_gpu_phi_solver).strip().lower() or "scipy"
        )
        if self.planner.iibtd_gpu_phi_solver not in {"scipy", "pgd"}:
            raise ValueError(
                "planner.iibtd_gpu_phi_solver must be one of scipy/pgd, got "
                f"{self.planner.iibtd_gpu_phi_solver!r}"
            )

        self.planner.default_quantization_bits = max(1, int(self.planner.default_quantization_bits))
        self.planner.high_quantization_bits = max(1, int(self.planner.high_quantization_bits))
        self.planner.low_quantization_bits = max(1, int(self.planner.low_quantization_bits))
        if self.planner.high_quantization_bits < self.planner.low_quantization_bits:
            raise ValueError(
                "planner.high_quantization_bits must be >= planner.low_quantization_bits, got "
                f"{self.planner.high_quantization_bits} < {self.planner.low_quantization_bits}"
            )
        if not bool(self.planner.adaptive_quantization_bits):
            self.planner.high_quantization_bits = int(self.planner.default_quantization_bits)
            self.planner.low_quantization_bits = int(self.planner.default_quantization_bits)

        self.reward.alpha_nmse = float(self.reward.w_nmse)
        self.reward.gamma_queue = float(self.reward.w_queue)
