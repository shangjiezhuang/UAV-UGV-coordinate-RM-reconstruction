"""
Utility functions for training, evaluation, logging, and visualization.
"""

import os
import json
import time
import numpy as np
from typing import Any, Dict, List, Optional
from collections import defaultdict

from config import Config
from environment import UAVUGVEnvironment


class MetricsLogger:
    """Simple metrics logger that writes to JSON and prints summaries."""

    def __init__(self, log_dir: str, metadata: Optional[Dict[str, Any]] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.episode_metrics: Dict[str, List[float]] = defaultdict(list)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.start_time = time.time()

    def set_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        """Replace the saved run metadata written alongside metrics."""
        self.metadata = dict(metadata or {})

    def log_update(self, update_idx: int, metrics: Dict[str, float]):
        """Log training update metrics."""
        for k, v in metrics.items():
            self.history[k].append(v)

        elapsed = time.time() - self.start_time
        print(
            f"\nUpdate {update_idx:5d} | "
            f"UAV π loss: {metrics.get('uav_policy_loss', 0):.4f} | "
            f"UGV π loss: {metrics.get('ugv_policy_loss', 0):.4f} | "
            f"V loss: {metrics.get('value_loss', 0):.4f} | "
            f"UAV ent: {metrics.get('uav_entropy', 0):.3f} | "
            f"UGV ent: {metrics.get('ugv_entropy', 0):.3f} | "
            f"Time: {elapsed:.0f}s"
        )

    def log_episode(self, info: Dict[str, float]):
        """Log episode-level metrics."""
        for k, v in info.items():
            if isinstance(v, (int, float)):
                self.episode_metrics[k].append(v)

    def log_eval(self, update_idx: int, eval_results: Dict[str, float]):
        """Log evaluation results."""
        print(f"\n{'='*60}")
        print(f"Evaluation at update {update_idx}:")
        for k, v in eval_results.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
        print(f"{'='*60}\n")

        # Keep eval timing aligned with the saved artifacts for later export.
        self.history["eval_update"].append(int(update_idx))

        # Save eval results to history
        for k, v in eval_results.items():
            self.history[k].append(v)

    def save(self, path: Optional[str] = None) -> str:
        """Save all metrics to JSON."""
        data = {
            "training": {k: v for k, v in self.history.items()},
            "episodes": {k: v for k, v in self.episode_metrics.items()},
        }
        if self.metadata:
            data.update(self.metadata)
        if path is None:
            path = os.path.join(self.log_dir, "metrics.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=lambda x: float(x))
        return path


def evaluate_policy(
    env: UAVUGVEnvironment,
    policy,
    num_episodes: int = 5,
    max_steps: int = 200,
    seed_base: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate current policy over multiple episodes.
    
    Args:
        env: Single environment instance.
        policy: MAPPOPolicy with trained networks.
        num_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        
    Returns:
        Dict of averaged evaluation metrics + last episode trajectories.
    """
    all_returns = []
    all_nmse = []
    all_steps = []
    all_energy_uav = []
    all_data_delivered_bits = []
    all_novel_data_delivered_bits = []
    all_uav_move_dist = []
    all_ugv_move_dist = []

    # Reward components tracking
    all_r_nmse = []
    all_r_unc = []
    all_r_new_freq = []
    all_r_new_spatial = []
    all_r_tx = []
    all_r_queue = []
    all_r_comm = []
    all_r_progress = []
    all_r_revisit = []
    all_r_terminal = []

    # Track last episode trajectory and per-step details for visualization
    last_uav_traj = []
    last_ugv_traj = []
    last_step_details = {}

    for ep in range(num_episodes):
        reset_seed = None if seed_base is None else int(seed_base + ep)
        obs, info = env.reset(seed=reset_seed)
        episode_return = 0
        step = 0

        # Per-episode reward accumulators
        ep_r_nmse = 0
        ep_r_unc = 0
        ep_r_new_freq = 0
        ep_r_new_spatial = 0
        ep_r_tx = 0
        ep_r_queue = 0
        ep_r_comm = 0
        ep_r_progress = 0
        ep_r_revisit = 0
        ep_r_terminal = 0
        ep_data_delivered_bits = 0.0
        ep_novel_data_delivered_bits = 0.0
        ep_uav_move_dist = 0.0
        ep_ugv_move_dist = 0.0

        uav_traj = [env.uav_pos.copy()]
        ugv_traj = [env.ugv_pos.copy()]

        # Per-step detail trackers
        step_queue_size = []
        step_snr_db = []
        step_bw_ratio = []
        step_sensing_ind = []
        step_sample_center_freq = []
        step_target_grid_x = []
        step_target_grid_y = []
        step_target_freq = []
        step_sensing_band_num = []
        step_sensing_bw_units = []
        step_comm_bw_units = []
        step_uav_ugv_dist = []
        step_uav_action = []
        step_uav_direction = []
        step_uav_bw_choice_idx = []
        step_ugv_action = []
        step_r_nmse = []
        step_r_unc = []
        step_r_new_freq = []
        step_r_new_spatial = []
        step_r_tx = []
        step_r_queue = []
        step_r_comm = []
        step_r_progress = []
        step_r_revisit = []
        step_r_terminal = []
        step_nmse = []
        step_planner_initialized = []
        step_target_reached = []
        step_target_source = []
        step_bootstrap_active = []
        step_bootstrap_target_reached = []
        step_bootstrap_handoff = []
        step_bootstrap_event = []
        step_reconstruction_triggered = []
        step_reconstruction_reason = []
        step_ensemble_triggered = []
        step_ensemble_reason = []
        step_spatial_revisit_count = []
        step_sample_novelty_ratio = []
        step_sample_repeat_ratio = []
        step_data_delivered_bits = []
        step_novel_data_delivered_bits = []
        step_uav_move_dist = []
        step_ugv_move_dist = []
        ensemble_events = []

        for step in range(max_steps):
            # Get deterministic actions
            action_data = policy.get_actions(
                uav_obs=obs["uav_obs"][np.newaxis],
                ugv_obs=obs["ugv_obs"][np.newaxis],
                critic_state=obs["critic_state"][np.newaxis],
                uav_move_action_mask=obs["uav_move_action_mask"][np.newaxis],
                uav_bw_action_mask=obs["uav_bw_action_mask"][np.newaxis],
                ugv_action_mask=obs["ugv_action_mask"][np.newaxis],
                deterministic=True,
            )
            uav_move_action = int(action_data["uav_move_action"][0])
            uav_bw_action = int(action_data["uav_bw_action"][0])
            uav_action = int(action_data["uav_action"][0])
            ugv_action = int(action_data["ugv_action"][0])
            uav_direction, uav_bw_choice_idx = env._decode_uav_action(uav_action)

            obs, rewards, terminated, truncated, info = env.step(
                uav_move_action,
                uav_bw_action,
                ugv_action,
            )
            episode_return += rewards["team_reward"]

            uav_traj.append(env.uav_pos.copy())
            ugv_traj.append(env.ugv_pos.copy())

            # Record per-step details
            step_queue_size.append(info.get("queue_size", 0))
            step_snr_db.append(info.get("snr_db", 0))
            step_bw_ratio.append(info.get("bw_ratio", 0.5))
            step_sensing_ind.append(info.get("sensing_ind", 0))
            step_sample_center_freq.append(info.get("sample_center_freq", info.get("sensing_ind", -1)))
            step_target_grid_x.append(info.get("target_grid_x", -1))
            step_target_grid_y.append(info.get("target_grid_y", -1))
            step_target_freq.append(info.get("target_center_freq", info.get("target_freq", -1)))
            step_sensing_band_num.append(info.get("sensing_band_num", 0))
            step_sensing_bw_units.append(info.get("sensing_bw_units", 0))
            step_comm_bw_units.append(info.get("comm_bw_units", 0))
            dist = np.sqrt(np.sum((env.uav_pos - env.ugv_pos) ** 2))
            step_uav_ugv_dist.append(float(dist))
            step_uav_action.append(uav_action)
            step_uav_direction.append(uav_direction)
            step_uav_bw_choice_idx.append(uav_bw_choice_idx)
            step_ugv_action.append(ugv_action)
            step_r_nmse.append(info.get("r_nmse", 0))
            step_r_unc.append(info.get("r_unc", 0))
            step_r_new_freq.append(info.get("r_new_freq", 0))
            step_r_new_spatial.append(info.get("r_new_spatial", 0))
            step_r_tx.append(info.get("r_tx", 0))
            step_r_queue.append(info.get("r_queue", 0))
            step_r_comm.append(info.get("r_comm", 0))
            step_r_progress.append(info.get("r_progress", 0))
            step_r_revisit.append(info.get("r_revisit", 0))
            step_r_terminal.append(info.get("r_terminal", 0))
            step_nmse.append(info.get("nmse", 0))
            step_planner_initialized.append(info.get("planner_initialized", 0))
            step_target_reached.append(info.get("target_reached", 0))
            step_target_source.append(info.get("target_source", "none"))
            step_bootstrap_active.append(info.get("bootstrap_active", 0))
            step_bootstrap_target_reached.append(info.get("bootstrap_target_reached", 0))
            step_bootstrap_handoff.append(info.get("bootstrap_handoff", 0))
            step_bootstrap_event.append(info.get("bootstrap_event", ""))
            step_reconstruction_triggered.append(info.get("reconstruction_triggered", 0))
            step_reconstruction_reason.append(info.get("reconstruction_reason", ""))
            step_spatial_revisit_count.append(info.get("spatial_revisit_count", 0))
            step_sample_novelty_ratio.append(info.get("sample_novelty_ratio", 0))
            step_sample_repeat_ratio.append(info.get("sample_repeat_ratio", 0))
            step_data_delivered_bits.append(info.get("data_delivered_bits", 0))
            step_novel_data_delivered_bits.append(info.get("novel_data_delivered_bits", 0))
            step_uav_move_dist.append(info.get("uav_move_dist", 0))
            step_ugv_move_dist.append(info.get("ugv_move_dist", 0))
            ensemble_triggered = int(info.get("ensemble_triggered", 0))
            ensemble_reason = info.get("ensemble_reason", "")
            step_ensemble_triggered.append(ensemble_triggered)
            step_ensemble_reason.append(ensemble_reason)
            if ensemble_triggered:
                ensemble_events.append(
                    {
                        "step": int(step + 1),
                        "reason": str(ensemble_reason),
                        "nmse": float(info.get("ensemble_event_nmse", info.get("nmse", 0.0))),
                        "nmse_delta": float(info.get("ensemble_event_nmse_delta", 0.0)),
                        "uav_pos": [float(env.uav_pos[0]), float(env.uav_pos[1])],
                        "ugv_pos": [float(env.ugv_pos[0]), float(env.ugv_pos[1])],
                        "target_grid_x": int(info.get("target_grid_x", -1)),
                        "target_grid_y": int(info.get("target_grid_y", -1)),
                        "target_center_freq": int(
                            info.get("target_center_freq", info.get("target_freq", -1))
                        ),
                    }
                )

            ep_r_nmse += info.get("r_nmse", 0)
            ep_r_unc += info.get("r_unc", 0)
            ep_r_new_freq += info.get("r_new_freq", 0)
            ep_r_new_spatial += info.get("r_new_spatial", 0)
            ep_r_tx += info.get("r_tx", 0)
            ep_r_queue += info.get("r_queue", 0)
            ep_r_comm += info.get("r_comm", 0)
            ep_r_progress += info.get("r_progress", 0)
            ep_r_revisit += info.get("r_revisit", 0)
            ep_r_terminal += info.get("r_terminal", 0)
            ep_data_delivered_bits += info.get("data_delivered_bits", 0)
            ep_novel_data_delivered_bits += info.get("novel_data_delivered_bits", 0)
            ep_uav_move_dist += info.get("uav_move_dist", 0)
            ep_ugv_move_dist += info.get("ugv_move_dist", 0)

            if terminated or truncated:
                break

        all_returns.append(episode_return)
        all_nmse.append(info.get("nmse", 0))
        all_steps.append(step + 1)
        all_energy_uav.append(info.get("uav_energy", 0))
        all_r_nmse.append(ep_r_nmse)
        all_r_unc.append(ep_r_unc)
        all_r_new_freq.append(ep_r_new_freq)
        all_r_new_spatial.append(ep_r_new_spatial)
        all_r_tx.append(ep_r_tx)
        all_r_queue.append(ep_r_queue)
        all_r_comm.append(ep_r_comm)
        all_r_progress.append(ep_r_progress)
        all_r_revisit.append(ep_r_revisit)
        all_r_terminal.append(ep_r_terminal)
        all_data_delivered_bits.append(ep_data_delivered_bits)
        all_novel_data_delivered_bits.append(ep_novel_data_delivered_bits)
        all_uav_move_dist.append(ep_uav_move_dist)
        all_ugv_move_dist.append(ep_ugv_move_dist)

        # Keep last episode trajectory and step details
        last_uav_traj = uav_traj
        last_ugv_traj = ugv_traj
        last_step_details = {
            "queue_size": step_queue_size,
            "snr_db": step_snr_db,
            "bw_ratio": step_bw_ratio,
            "sensing_ind": step_sensing_ind,
            "sample_center_freq": step_sample_center_freq,
            "target_grid_x": step_target_grid_x,
            "target_grid_y": step_target_grid_y,
            "target_freq": step_target_freq,
            "target_center_freq": step_target_freq,
            "sensing_band_num": step_sensing_band_num,
            "sensing_bw_units": step_sensing_bw_units,
            "comm_bw_units": step_comm_bw_units,
            "uav_ugv_dist": step_uav_ugv_dist,
            "uav_action": step_uav_action,
            "uav_direction": step_uav_direction,
            "uav_bw_choice_idx": step_uav_bw_choice_idx,
            "ugv_action": step_ugv_action,
            "r_nmse": step_r_nmse,
            "r_unc": step_r_unc,
            "r_new_freq": step_r_new_freq,
            "r_new_spatial": step_r_new_spatial,
            "r_tx": step_r_tx,
            "r_queue": step_r_queue,
            "r_comm": step_r_comm,
            "r_progress": step_r_progress,
            "r_revisit": step_r_revisit,
            "r_terminal": step_r_terminal,
            "nmse": step_nmse,
            "planner_initialized": step_planner_initialized,
            "target_reached": step_target_reached,
            "target_source": step_target_source,
            "bootstrap_active": step_bootstrap_active,
            "bootstrap_target_reached": step_bootstrap_target_reached,
            "bootstrap_handoff": step_bootstrap_handoff,
            "bootstrap_event": step_bootstrap_event,
            "reconstruction_triggered": step_reconstruction_triggered,
            "reconstruction_reason": step_reconstruction_reason,
            "ensemble_triggered": step_ensemble_triggered,
            "ensemble_reason": step_ensemble_reason,
            "spatial_revisit_count": step_spatial_revisit_count,
            "sample_novelty_ratio": step_sample_novelty_ratio,
            "sample_repeat_ratio": step_sample_repeat_ratio,
            "data_delivered_bits": step_data_delivered_bits,
            "novel_data_delivered_bits": step_novel_data_delivered_bits,
            "uav_move_dist": step_uav_move_dist,
            "ugv_move_dist": step_ugv_move_dist,
            "ensemble_events": ensemble_events,
            "bootstrap_events": list(getattr(env, "bootstrap_events", [])),
            "completed_target_nmse_records": list(getattr(env, "completed_target_nmse_records", [])),
        }

    return {
        "eval_mean_return": np.mean(all_returns),
        "eval_std_return": np.std(all_returns),
        "eval_mean_nmse": np.mean(all_nmse),
        "eval_mean_steps": np.mean(all_steps),
        "eval_mean_uav_energy_remaining": np.mean(all_energy_uav),
        "eval_mean_r_nmse": np.mean(all_r_nmse),
        "eval_mean_r_unc": np.mean(all_r_unc),
        "eval_mean_r_new_freq": np.mean(all_r_new_freq),
        "eval_mean_r_new_spatial": np.mean(all_r_new_spatial),
        "eval_mean_r_tx": np.mean(all_r_tx),
        "eval_mean_r_queue": np.mean(all_r_queue),
        "eval_mean_r_comm": np.mean(all_r_comm),
        "eval_mean_r_progress": np.mean(all_r_progress),
        "eval_mean_r_revisit": np.mean(all_r_revisit),
        "eval_mean_r_terminal": np.mean(all_r_terminal),
        "eval_mean_data_delivered_bits": np.mean(all_data_delivered_bits),
        "eval_mean_novel_data_delivered_bits": np.mean(all_novel_data_delivered_bits),
        "eval_mean_uav_move_dist": np.mean(all_uav_move_dist),
        "eval_mean_ugv_move_dist": np.mean(all_ugv_move_dist),
        # Trajectories from last eval episode
        "eval_uav_trajectory": [pos.tolist() for pos in last_uav_traj],
        "eval_ugv_trajectory": [pos.tolist() for pos in last_ugv_traj],
        # Per-step details from last eval episode
        "eval_step_details": last_step_details,
    }


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def print_config_summary(config: Config):
    """Print a readable summary of the configuration."""
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)

    print(f"\n--- Scene ---")
    print(f"  Grid: {config.scene.grid_size[0]}×{config.scene.grid_size[1]}")
    print(f"  Freq bands: {config.scene.total_freq_bands_nums}")
    print(f"  UAV height: {config.scene.uav_height}m")

    print(f"\n--- UAV ---")
    print(f"  Directions: {config.uav.num_directions} (stay + cardinal)")
    print(f"  Max energy: {config.uav.max_energy}J")
    print(f"  Total bandwidth: {config.uav.total_bandwidth/1e6:.0f} MHz")
    print(f"  Bandwidth units: {config.uav.total_bw_num} (each {config.uav.unit_bandwidth_hz/1e6:.1f} MHz)")
    print(f"  Default BW ratio: {config.uav.default_bw_ratio}")
    print(f"  BW ratio choices: {config.uav.bandwidth_ratios}")
    print(f"  Sampling mode: current grid cell + GT noise")
    print(
        f"  Quantization bits: adaptive={bool(config.planner.adaptive_quantization_bits)} "
        f"(fixed={config.planner.default_quantization_bits}, "
        f"high={config.planner.high_quantization_bits}, "
        f"low={config.planner.low_quantization_bits}, "
        f"thr={config.planner.uncertainty_quantization_threshold})"
    )

    print(f"\n--- Action Spaces ---")
    print(
        f"  UAV: move_head({config.uav.num_directions}) + "
        f"bandwidth_head({config.uav.num_bandwidth_ratios})"
    )
    print(f"  UGV: {config.ugv.num_directions}")

    print(f"\n--- Planner ---")
    print(f"  Sensor budget:    {config.planner.sensor_budget}")
    print(f"  Target count:     {config.planner.target_count}")
    print(f"  Obs target slots: {config.planner.obs_target_slots}")
    print(f"  Init pair d_max:  {config.planner.init_pair_max_distance}")
    print(f"  Planner warmup M: {config.planner.min_samples_for_ensemble}")
    print(f"  Ensemble / map update interval: {config.planner.ensemble_refresh_interval}")
    print(f"  Low-priority N:   {config.planner.low_priority_process_interval}")
    print(f"  Ensemble size:    {config.planner.ensemble_size}")
    print(f"  Ensemble keep:    {config.planner.ensemble_keep_ratio} (recent {config.planner.ensemble_keep_recent})")

    print(f"\n--- MAPPO ---")
    print(f"  Envs: {config.mappo.num_envs}")
    print(f"  Vec backend: {config.mappo.vec_backend}")
    print(f"  Total steps: {config.mappo.total_timesteps:,}")
    print(f"  Rollout length: {config.mappo.rollout_length}")
    print(f"  LR (actor/critic): {config.mappo.lr_actor}/{config.mappo.lr_critic}")
    print(f"  Gamma: {config.mappo.gamma}, GAE λ: {config.mappo.gae_lambda}")

    print(f"\n--- Reward ---")
    print(f"  w_nmse:           {config.reward.w_nmse}")
    print(f"  w_tx:             {config.reward.w_tx}")
    print(f"  w_queue:          {config.reward.w_queue}")
    print(f"  w_comm:           {config.reward.w_comm}")
    print(f"  lambda_uav_prog:  {config.reward.lambda_uav_progress}")
    print(f"  lambda_ugv_prog:  {config.reward.lambda_ugv_progress}")
    print(f"  lambda_uav_back:  {config.reward.lambda_uav_backtrack}")
    print(f"  lambda_ugv_back:  {config.reward.lambda_ugv_backtrack}")
    print(f"  goal_arrival:     {config.reward.local_goal_arrival_bonus}")
    print(f"  terminal_success: {config.reward.terminal_success_bonus}")
    print(f"  terminal_fail:    {config.reward.terminal_failure_penalty}")
    print(f"  q_ref:            {config.reward.q_ref}")
    print(f"  Target NMSE:      {config.reward.accuracy_target_nmse}")
    print("=" * 60 + "\n")
