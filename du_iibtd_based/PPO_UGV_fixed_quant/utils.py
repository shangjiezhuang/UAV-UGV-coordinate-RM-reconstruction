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


def json_default(value: Any) -> Any:
    """Serialize common NumPy values without silently coercing unrelated objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
            f"V loss: {metrics.get('value_loss', 0):.4f} | "
            f"UAV ent: {metrics.get('uav_entropy', 0):.3f} | "
            f"Time: {elapsed:.0f}s"
        )

    def log_episode(self, info: Dict[str, float]):
        """Log episode-level metrics."""
        for k, v in info.items():
            if isinstance(v, (int, float)):
                self.episode_metrics[k].append(v)

    def _eval_history_key(self, key: str, prefix: str) -> str:
        """Map evaluate_policy output keys to history keys for a given prefix."""
        if prefix == "eval":
            return key
        if key.startswith("eval_"):
            return f"{prefix}_{key[len('eval_'):]}"
        return f"{prefix}_{key}"

    def _log_eval_results(
        self,
        update_idx: int,
        eval_results: Dict[str, float],
        prefix: str,
        label: str,
    ) -> None:
        """Shared logger for periodic evals and final eval."""
        print(f"\n{'='*60}")
        print(f"{label} at update {update_idx}:")
        for k, v in eval_results.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
        print(f"{'='*60}\n")

        # Keep eval timing aligned with the saved artifacts for later export.
        self.history[f"{prefix}_update"].append(int(update_idx))

        # Save eval results to history
        for k, v in eval_results.items():
            self.history[self._eval_history_key(k, prefix)].append(v)

    def log_eval(self, update_idx: int, eval_results: Dict[str, float]):
        """Log periodic evaluation results."""
        self._log_eval_results(
            update_idx=update_idx,
            eval_results=eval_results,
            prefix="eval",
            label="Evaluation",
        )

    def log_final_eval(self, update_idx: int, eval_results: Dict[str, float]):
        """Log final evaluation results separately from periodic evals."""
        self._log_eval_results(
            update_idx=update_idx,
            eval_results=eval_results,
            prefix="final_eval",
            label="Final Evaluation",
        )

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
            json.dump(data, f, indent=2, default=json_default)
        return path


def _resolve_eval_max_steps(env, max_steps: Optional[int]) -> int:
    if max_steps is not None:
        return int(max_steps)

    env_config = getattr(env, "config", None)
    for section_name in ("mappo", "ippo", "run"):
        section = getattr(env_config, section_name, None)
        if section is not None and hasattr(section, "episode_max_steps"):
            return int(section.episode_max_steps)
    raise AttributeError("Unable to infer episode_max_steps from env.config")


def evaluate_policy(
    env: UAVUGVEnvironment,
    policy,
    num_episodes: int = 5,
    max_steps: Optional[int] = None,
    seed_base: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate current policy over multiple episodes.
    
    Args:
        env: Single environment instance.
        policy: MAPPOPolicy with trained networks.
        num_episodes: Number of evaluation episodes.
        max_steps: Optional override; None uses env.config.*.episode_max_steps.
        
    Returns:
        Dict of averaged evaluation metrics plus artifacts from the final
        episode in the evaluation batch.
    """
    if int(num_episodes) <= 0:
        raise ValueError("num_episodes must be positive")

    max_steps = _resolve_eval_max_steps(env, max_steps)
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    visualized_episode_index = int(num_episodes) - 1
    visualized_reset_seed = (
        None if seed_base is None else int(seed_base + visualized_episode_index)
    )

    all_returns = []
    all_nmse = []
    all_steps = []
    all_energy_uav = []
    all_data_delivered_bits = []
    all_novel_data_delivered_bits = []
    all_data_transmitted_bits = []
    all_uav_move_dist = []
    all_uav_move_steps = []

    # Reward components tracking
    all_r_nmse = []
    all_r_unc = []
    all_r_new_freq = []
    all_r_new_spatial = []
    all_r_tx = []
    all_r_queue = []
    all_r_progress = []
    all_r_uav_progress = []
    all_r_goal_arrival = []
    all_r_revisit = []
    all_r_terminal = []
    all_nmse_target_gap = []
    all_target_gap_penalty_diag = []

    # Keep one concrete episode for visualization: the final episode in this eval batch.
    last_uav_traj = []
    last_ugv_traj = []
    last_step_details = {}

    for ep in range(num_episodes):
        reset_seed = None if seed_base is None else int(seed_base + ep)
        obs, _ = env.reset(seed=reset_seed)
        episode_return = 0

        # Per-episode reward accumulators
        ep_r_nmse = 0
        ep_r_unc = 0
        ep_r_new_freq = 0
        ep_r_new_spatial = 0
        ep_r_tx = 0
        ep_r_queue = 0
        ep_r_progress = 0
        ep_r_uav_progress = 0
        ep_r_goal_arrival = 0
        ep_r_revisit = 0
        ep_r_terminal = 0
        ep_target_gap_penalty_diag = 0.0
        ep_data_delivered_bits = 0.0
        ep_novel_data_delivered_bits = 0.0
        ep_data_transmitted_bits = 0.0
        ep_uav_move_dist = 0.0
        ep_uav_move_steps = 0.0

        uav_traj = [env.uav_pos.copy()]
        ugv_traj = [env.ugv_pos.copy()]

        # Per-step detail trackers
        step_queue_size = []
        step_snr_db = []
        step_bw_ratio = []
        step_sample_center_freq = []
        step_target_grid_x = []
        step_target_grid_y = []
        step_target_center_freq = []
        step_sensing_band_num = []
        step_sensing_bw_units = []
        step_comm_bw_units = []
        step_uav_ugv_dist = []
        step_uav_action = []
        step_uav_direction = []
        step_uav_bw_choice_idx = []
        step_uav_quant_choice_idx = []
        step_quant_bits = []
        step_current_quant_norm = []
        step_r_nmse = []
        step_r_unc = []
        step_r_new_freq = []
        step_r_new_spatial = []
        step_r_tx = []
        step_r_queue = []
        step_r_progress = []
        step_r_uav_progress = []
        step_r_goal_arrival = []
        step_r_revisit = []
        step_r_terminal = []
        step_nmse = []
        step_nmse_target_gap = []
        step_target_gap_penalty_diag = []
        step_target_nmse = []
        step_planner_initialized = []
        step_target_reached = []
        step_target_source = []
        step_planner_submode = []
        step_planner_effective_target_mode = []
        step_planner_mode_switch = []
        step_planner_mode_switch_reason = []
        step_planner_global_steps_remaining = []
        step_planner_local_candidate_count = []
        step_planner_local_candidate_threshold = []
        step_bootstrap_active = []
        step_bootstrap_target_reached = []
        step_bootstrap_handoff = []
        step_bootstrap_event = []
        step_ensemble_triggered = []
        step_ensemble_reason = []
        step_ensemble_recon_mode = []
        step_ensemble_full_refresh_due = []
        step_ensemble_nmse_refresh_triggered = []
        step_ensemble_nmse_degradation = []
        step_spatial_revisit_count = []
        step_sample_novelty_ratio = []
        step_sample_repeat_ratio = []
        step_data_delivered_bits = []
        step_novel_data_delivered_bits = []
        step_data_transmitted_bits = []
        step_uav_move_dist = []
        step_uav_move_steps = []
        step_global_top_fields = {
            f"global_top{rank}_{field}": []
            for rank in range(1, 4)
            for field in ("x", "y", "freq", "score")
        }
        ensemble_events = []

        for step in range(max_steps):
            # Get deterministic actions
            action_data = policy.get_single_action(
                uav_obs=obs["uav_obs"],
                critic_state=obs["critic_state"],
                uav_action_mask=obs["uav_action_mask"],
                deterministic=True,
            )
            uav_action = int(action_data["uav_action"])
            uav_direction, uav_bw_choice_idx, uav_quant_choice_idx = env._decode_uav_action(uav_action)

            obs, rewards, terminated, truncated, info = env.step(
                uav_action,
            )
            episode_return += rewards["team_reward"]

            uav_traj.append(env.uav_pos.copy())
            ugv_traj.append(env.ugv_pos.copy())

            # Record per-step details
            step_queue_size.append(info["queue_size"])
            step_snr_db.append(info["snr_db"])
            step_bw_ratio.append(info["bw_ratio"])
            step_sample_center_freq.append(info["sample_center_freq"])
            step_target_grid_x.append(info["target_grid_x"])
            step_target_grid_y.append(info["target_grid_y"])
            step_target_center_freq.append(info["target_center_freq"])
            step_sensing_band_num.append(info["sensing_band_num"])
            step_sensing_bw_units.append(info["sensing_bw_units"])
            step_comm_bw_units.append(info["comm_bw_units"])
            dist = np.sqrt(np.sum((env.uav_pos - env.ugv_pos) ** 2))
            step_uav_ugv_dist.append(float(dist))
            step_uav_action.append(uav_action)
            step_uav_direction.append(uav_direction)
            step_uav_bw_choice_idx.append(uav_bw_choice_idx)
            step_uav_quant_choice_idx.append(uav_quant_choice_idx)
            step_quant_bits.append(info["quant_bits"])
            step_current_quant_norm.append(info["current_quant_norm"])
            step_r_nmse.append(info["r_nmse"])
            step_r_unc.append(info["r_unc"])
            step_r_new_freq.append(info["r_new_freq"])
            step_r_new_spatial.append(info["r_new_spatial"])
            step_r_tx.append(info["r_tx"])
            step_r_queue.append(info["r_queue"])
            step_r_progress.append(info["r_progress"])
            step_r_uav_progress.append(info["r_uav_progress"])
            step_r_goal_arrival.append(info["r_goal_arrival"])
            step_r_revisit.append(info["r_revisit"])
            step_r_terminal.append(info["r_terminal"])
            step_nmse.append(info["nmse"])
            step_nmse_target_gap.append(info["nmse_target_gap"])
            step_target_gap_penalty_diag.append(info["target_gap_penalty_diag"])
            step_target_nmse.append(info["target_nmse"])
            step_planner_initialized.append(info["planner_initialized"])
            step_target_reached.append(info["target_reached"])
            step_target_source.append(info["target_source"])
            step_planner_submode.append(info["planner_submode"])
            step_planner_effective_target_mode.append(
                info["planner_effective_target_mode"]
            )
            step_planner_mode_switch.append(info["planner_mode_switch"])
            step_planner_mode_switch_reason.append(
                info["planner_mode_switch_reason"]
            )
            step_planner_global_steps_remaining.append(
                info["planner_global_steps_remaining"]
            )
            step_planner_local_candidate_count.append(
                info["planner_local_candidate_count"]
            )
            step_planner_local_candidate_threshold.append(
                info["planner_local_candidate_threshold"]
            )
            step_bootstrap_active.append(info["bootstrap_active"])
            step_bootstrap_target_reached.append(info["bootstrap_target_reached"])
            step_bootstrap_handoff.append(info["bootstrap_handoff"])
            step_bootstrap_event.append(info["bootstrap_event"])
            step_spatial_revisit_count.append(info["spatial_revisit_count"])
            step_sample_novelty_ratio.append(info["sample_novelty_ratio"])
            step_sample_repeat_ratio.append(info["sample_repeat_ratio"])
            step_data_delivered_bits.append(info["data_delivered_bits"])
            step_novel_data_delivered_bits.append(info["novel_data_delivered_bits"])
            step_data_transmitted_bits.append(info["data_transmitted_bits"])
            step_uav_move_dist.append(info["uav_move_dist"])
            step_uav_move_steps.append(info["uav_move_steps"])
            for rank in range(1, 4):
                step_global_top_fields[f"global_top{rank}_x"].append(
                    info[f"global_top{rank}_x"]
                )
                step_global_top_fields[f"global_top{rank}_y"].append(
                    info[f"global_top{rank}_y"]
                )
                step_global_top_fields[f"global_top{rank}_freq"].append(
                    info[f"global_top{rank}_freq"]
                )
                step_global_top_fields[f"global_top{rank}_score"].append(
                    info[f"global_top{rank}_score"]
                )
            ensemble_triggered = int(info["ensemble_triggered"])
            ensemble_reason = info["ensemble_reason"]
            ensemble_recon_mode = info["ensemble_recon_mode"]
            step_ensemble_triggered.append(ensemble_triggered)
            step_ensemble_reason.append(ensemble_reason)
            step_ensemble_recon_mode.append(ensemble_recon_mode)
            step_ensemble_full_refresh_due.append(info["ensemble_full_refresh_due"])
            step_ensemble_nmse_refresh_triggered.append(
                info["ensemble_nmse_refresh_triggered"]
            )
            step_ensemble_nmse_degradation.append(
                info["ensemble_nmse_degradation"]
            )
            if ensemble_triggered:
                ensemble_events.append(
                    {
                        "step": int(step + 1),
                        "reason": str(ensemble_reason),
                        "recon_mode": str(ensemble_recon_mode),
                        "full_refresh_due": int(info["ensemble_full_refresh_due"]),
                        "nmse_refresh_triggered": int(
                            info["ensemble_nmse_refresh_triggered"]
                        ),
                        "nmse_refresh_delta": float(
                            info["ensemble_nmse_refresh_delta"]
                        ),
                        "nmse_refresh_reference_before": float(
                            info["ensemble_nmse_refresh_reference_before"]
                        ),
                        "nmse_refresh_reference_after": float(
                            info["ensemble_nmse_refresh_reference_after"]
                        ),
                        "nmse_degradation": float(
                            info["ensemble_nmse_degradation"]
                        ),
                        "nmse": float(info["ensemble_event_nmse"]),
                        "nmse_delta": float(info["ensemble_event_nmse_delta"]),
                        "uav_pos": [float(env.uav_pos[0]), float(env.uav_pos[1])],
                        "ugv_pos": [float(env.ugv_pos[0]), float(env.ugv_pos[1])],
                        "target_grid_x": int(info["target_grid_x"]),
                        "target_grid_y": int(info["target_grid_y"]),
                        "target_center_freq": int(
                            info["target_center_freq"]
                        ),
                    }
                )

            ep_r_nmse += info["r_nmse"]
            ep_r_unc += info["r_unc"]
            ep_r_new_freq += info["r_new_freq"]
            ep_r_new_spatial += info["r_new_spatial"]
            ep_r_tx += info["r_tx"]
            ep_r_queue += info["r_queue"]
            ep_r_progress += info["r_progress"]
            ep_r_uav_progress += info["r_uav_progress"]
            ep_r_goal_arrival += info["r_goal_arrival"]
            ep_r_revisit += info["r_revisit"]
            ep_r_terminal += info["r_terminal"]
            ep_target_gap_penalty_diag += info["target_gap_penalty_diag"]
            ep_data_delivered_bits += info["data_delivered_bits"]
            ep_novel_data_delivered_bits += info["novel_data_delivered_bits"]
            ep_data_transmitted_bits += info["data_transmitted_bits"]
            ep_uav_move_dist += info["uav_move_dist"]
            ep_uav_move_steps += info["uav_move_steps"]

            if terminated or truncated:
                break

        all_returns.append(episode_return)
        all_nmse.append(info["nmse"])
        all_steps.append(step + 1)
        all_energy_uav.append(info["uav_energy"])
        all_r_nmse.append(ep_r_nmse)
        all_r_unc.append(ep_r_unc)
        all_r_new_freq.append(ep_r_new_freq)
        all_r_new_spatial.append(ep_r_new_spatial)
        all_r_tx.append(ep_r_tx)
        all_r_queue.append(ep_r_queue)
        all_r_progress.append(ep_r_progress)
        all_r_uav_progress.append(ep_r_uav_progress)
        all_r_goal_arrival.append(ep_r_goal_arrival)
        all_r_revisit.append(ep_r_revisit)
        all_r_terminal.append(ep_r_terminal)
        all_nmse_target_gap.append(info["nmse_target_gap"])
        all_target_gap_penalty_diag.append(ep_target_gap_penalty_diag)
        all_data_delivered_bits.append(ep_data_delivered_bits)
        all_novel_data_delivered_bits.append(ep_novel_data_delivered_bits)
        all_data_transmitted_bits.append(ep_data_transmitted_bits)
        all_uav_move_dist.append(ep_uav_move_dist)
        all_uav_move_steps.append(ep_uav_move_steps)

        # Keep the final eval episode as the visualized artifact.
        last_uav_traj = uav_traj
        last_ugv_traj = ugv_traj
        last_step_details = {
            "queue_size": step_queue_size,
            "snr_db": step_snr_db,
            "bw_ratio": step_bw_ratio,
            "sample_center_freq": step_sample_center_freq,
            "target_grid_x": step_target_grid_x,
            "target_grid_y": step_target_grid_y,
            "target_center_freq": step_target_center_freq,
            "sensing_band_num": step_sensing_band_num,
            "sensing_bw_units": step_sensing_bw_units,
            "comm_bw_units": step_comm_bw_units,
            "uav_ugv_dist": step_uav_ugv_dist,
            "uav_action": step_uav_action,
            "uav_direction": step_uav_direction,
            "uav_bw_choice_idx": step_uav_bw_choice_idx,
            "uav_quant_choice_idx": step_uav_quant_choice_idx,
            "quant_bits": step_quant_bits,
            "current_quant_norm": step_current_quant_norm,
            "r_nmse": step_r_nmse,
            "r_unc": step_r_unc,
            "r_new_freq": step_r_new_freq,
            "r_new_spatial": step_r_new_spatial,
            "r_tx": step_r_tx,
            "r_queue": step_r_queue,
            "r_progress": step_r_progress,
            "r_uav_progress": step_r_uav_progress,
            "r_goal_arrival": step_r_goal_arrival,
            "r_revisit": step_r_revisit,
            "r_terminal": step_r_terminal,
            "nmse": step_nmse,
            "nmse_target_gap": step_nmse_target_gap,
            "target_gap_penalty_diag": step_target_gap_penalty_diag,
            "target_nmse": step_target_nmse,
            "planner_initialized": step_planner_initialized,
            "target_reached": step_target_reached,
            "target_source": step_target_source,
            "planner_submode": step_planner_submode,
            "planner_effective_target_mode": step_planner_effective_target_mode,
            "planner_mode_switch": step_planner_mode_switch,
            "planner_mode_switch_reason": step_planner_mode_switch_reason,
            "planner_global_steps_remaining": step_planner_global_steps_remaining,
            "planner_local_candidate_count": step_planner_local_candidate_count,
            "planner_local_candidate_threshold": step_planner_local_candidate_threshold,
            "bootstrap_active": step_bootstrap_active,
            "bootstrap_target_reached": step_bootstrap_target_reached,
            "bootstrap_handoff": step_bootstrap_handoff,
            "bootstrap_event": step_bootstrap_event,
            "ensemble_triggered": step_ensemble_triggered,
            "ensemble_reason": step_ensemble_reason,
            "ensemble_recon_mode": step_ensemble_recon_mode,
            "ensemble_full_refresh_due": step_ensemble_full_refresh_due,
            "ensemble_nmse_refresh_triggered": step_ensemble_nmse_refresh_triggered,
            "ensemble_nmse_degradation": step_ensemble_nmse_degradation,
            "spatial_revisit_count": step_spatial_revisit_count,
            "sample_novelty_ratio": step_sample_novelty_ratio,
            "sample_repeat_ratio": step_sample_repeat_ratio,
            "data_delivered_bits": step_data_delivered_bits,
            "novel_data_delivered_bits": step_novel_data_delivered_bits,
            "data_transmitted_bits": step_data_transmitted_bits,
            "uav_move_dist": step_uav_move_dist,
            "uav_move_steps": step_uav_move_steps,
            **step_global_top_fields,
            "ensemble_events": ensemble_events,
            "bootstrap_events": list(env.bootstrap_events),
            "completed_target_nmse_records": list(env.completed_target_nmse_records),
        }

    best_nmse_idx = int(np.argmin(all_nmse))
    worst_nmse_idx = int(np.argmax(all_nmse))

    return {
        "eval_num_episodes": int(num_episodes),
        "eval_visualized_episode_index": visualized_episode_index,
        "eval_visualized_reset_seed": visualized_reset_seed,
        "eval_mean_return": np.mean(all_returns),
        "eval_std_return": np.std(all_returns),
        # NMSE lower is better. Transmission paired with best/worst NMSE stays on the same episode.
        "eval_best_nmse": np.min(all_nmse),
        "eval_worst_nmse": np.max(all_nmse),
        "eval_mean_nmse": np.mean(all_nmse),
        "eval_std_nmse": np.std(all_nmse),
        "eval_mean_steps": np.mean(all_steps),
        "eval_mean_uav_energy_remaining": np.mean(all_energy_uav),
        "eval_mean_r_nmse": np.mean(all_r_nmse),
        "eval_mean_r_unc": np.mean(all_r_unc),
        "eval_mean_r_new_freq": np.mean(all_r_new_freq),
        "eval_mean_r_new_spatial": np.mean(all_r_new_spatial),
        "eval_mean_r_tx": np.mean(all_r_tx),
        "eval_mean_r_queue": np.mean(all_r_queue),
        "eval_mean_r_progress": np.mean(all_r_progress),
        "eval_mean_r_uav_progress": np.mean(all_r_uav_progress),
        "eval_mean_r_goal_arrival": np.mean(all_r_goal_arrival),
        "eval_mean_r_revisit": np.mean(all_r_revisit),
        "eval_mean_r_terminal": np.mean(all_r_terminal),
        "eval_mean_nmse_target_gap": np.mean(all_nmse_target_gap),
        "eval_mean_target_gap_penalty_diag": np.mean(all_target_gap_penalty_diag),
        "eval_mean_data_delivered_bits": np.mean(all_data_delivered_bits),
        "eval_mean_novel_data_delivered_bits": np.mean(all_novel_data_delivered_bits),
        "eval_data_transmitted_bits_at_best_nmse": all_data_transmitted_bits[best_nmse_idx],
        "eval_data_transmitted_bits_at_worst_nmse": all_data_transmitted_bits[worst_nmse_idx],
        "eval_min_data_transmitted_bits": np.min(all_data_transmitted_bits),
        "eval_max_data_transmitted_bits": np.max(all_data_transmitted_bits),
        "eval_mean_data_transmitted_bits": np.mean(all_data_transmitted_bits),
        "eval_mean_uav_move_dist": np.mean(all_uav_move_dist),
        "eval_mean_uav_move_steps": np.mean(all_uav_move_steps),
        # Artifacts from eval_visualized_episode_index.
        "eval_uav_trajectory": [pos.tolist() for pos in last_uav_traj],
        "eval_ugv_trajectory": [pos.tolist() for pos in last_ugv_traj],
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
    print(f"  Source: {config.scene.scene_source}")
    print(f"  RadioSeer root: {config.scene.radioseer_root}")
    print(f"  RadioSeer sample: {config.scene.radioseer_sample_index}")
    print(f"  Freq bands: {config.scene.total_freq_bands_nums}")
    print(f"  UAV height: {config.scene.uav_height}m")

    print(f"\n--- UAV ---")
    print(f"  Directions: {config.uav.num_directions} (cardinal only, no hover)")
    print(f"  Max energy: {config.uav.max_energy}J")
    print(f"  Total bandwidth: {config.uav.total_bandwidth/1e6:.0f} MHz")
    print(f"  Bandwidth units: {config.uav.total_bw_num} (each {config.uav.unit_bandwidth_hz/1e6:.1f} MHz)")
    print(f"  Default BW ratio: {config.uav.default_bw_ratio}")
    print(f"  BW ratio choices: {config.uav.bandwidth_ratios}")
    print(f"  Quant bit choices: {config.uav.quant_bits}")
    print(f"  Default quant bits: {config.uav.default_quant_bits}")
    print(f"  Queue capacity: {config.uav.queue_capacity_packets} packets")
    print(f"  Sampling mode: current grid cell + GT noise + log-domain quant/dequant")

    print(f"\n--- Action Spaces ---")
    uav_actions = (
        config.uav.num_directions
        * config.uav.num_bandwidth_ratios
        * config.uav.num_quant_bits
    )
    print(
        f"  UAV: direction({config.uav.num_directions}) "
        f"x bandwidth_select({config.uav.num_bandwidth_ratios}) "
        f"x quant_bits({config.uav.num_quant_bits}) = {uav_actions}"
    )
    print("  UGV: fixed position (stay-only compatibility action)")

    print(f"\n--- Planner ---")
    print(f"  Target count:     {config.planner.target_count}")
    print(f"  Obs target slots: {config.planner.obs_target_slots}")
    print(f"  Target mode:      {config.planner.target_mode}")
    print(f"  Init mode:        {config.planner.initial_observation_mode}")
    print(f"  Init pair d_max:  {config.planner.init_pair_max_distance}")
    print(f"  Local radius:     {config.planner.local_planner_radius}")
    print(f"  Prefill percent:  {config.planner.prefill_percent}")
    prefill_basis = (
        config.planner.prefill_budget_basis
        if int(config.planner.prefill_budget_basis) > 0
        else config.mappo.episode_max_steps
    )
    print(f"  Prefill basis:    {prefill_basis}")
    print(f"  Init clearance:   {config.planner.init_building_clearance}")
    print(f"  Bootstrap clear.: {config.planner.bootstrap_building_clearance}")
    print(f"  Terminal flush:   {config.planner.flush_reconstruction_on_episode_end}")
    print(f"  Planner warmup M: {config.planner.min_samples_for_ensemble} effective grid samples")
    print(f"  Ensemble / map update interval: {config.planner.ensemble_refresh_interval}")
    print(f"  Ensemble size:    {config.planner.ensemble_size}")
    print(f"  Ensemble obs:     shared_all")
    print(f"  Ensemble kernel:  {config.planner.ensemble_kernel_bandwidth_mode} (delta {config.planner.ensemble_kernel_bandwidth_delta})")
    print(f"  Ensemble jitter:  {config.planner.ensemble_init_jitter_scale}")
    print(f"  Ensemble quality weighting: {config.planner.ensemble_quality_weighted}")
    print(f"  Full refresh interval: {config.planner.ensemble_full_refresh_interval}")
    print(f"  NMSE refresh delta: {config.planner.nmse_refresh_delta}")
    print(
        f"  Incremental iters / SVT: "
        f"{config.planner.incremental_outer_iters} / {config.planner.incremental_max_svt_iters}"
    )
    print(f"  II-BTD backend:   {config.planner.iibtd_backend}")
    print(f"  II-BTD device:    {config.planner.iibtd_device}")
    if "du_iibtd" in str(config.planner.iibtd_backend).strip().lower():
        print("  DU-IIBTD solver params: loaded from checkpoint config")
    else:
        print(f"  II-BTD mu / nu:   {config.planner.iibtd_mu} / {config.planner.iibtd_nu}")
        print(f"  II-BTD kernel bw: {config.planner.iibtd_kernel_bandwidth}")
    du_iibtd_models = (
        ", ".join(
            os.path.basename(os.path.dirname(os.path.dirname(str(path))))
            for path in config.planner.du_iibtd_checkpoints
        )
        if config.planner.iibtd_backend in {"du_iibtd_res_sr", "du_iibtd_res_sr_learn_nu"}
        else "inactive"
    )
    print(f"  DU-IIBTD models:  {du_iibtd_models}")

    print(f"\n--- MAPPO ---")
    print(f"  Envs: {config.mappo.num_envs}")
    print(f"  Vec backend: {config.mappo.vec_backend}")
    print(f"  Total steps: {config.mappo.total_timesteps:,}")
    print(f"  Rollout length: {config.mappo.rollout_length}")
    print(f"  LR (actor/critic): {config.mappo.lr_actor}/{config.mappo.lr_critic}")
    print(f"  Gamma: {config.mappo.gamma}, GAE λ: {config.mappo.gae_lambda}")
    print(f"  Eval seed stride: {config.mappo.eval_seed_stride}")

    print(f"\n--- Reward ---")
    print(f"  α_nmse:           {config.reward.alpha_nmse}")
    print(f"  δ_nmse clip:      {config.reward.nmse_signed_clip}")
    print(f"  gap diag coef:    {config.reward.target_gap_penalty_coef}")
    print(f"  α_unc:            {config.reward.alpha_unc}")
    print(f"  λ_new_freq:       {config.reward.lambda_new_freq}")
    print(f"  λ_new_spatial:    {config.reward.lambda_new_spatial}")
    print(f"  β (tx delivered): {config.reward.beta_tx}")
    print(f"  γ (queue bits):   {config.reward.gamma_queue}")
    print(f"  λ_uav_progress+:  {config.reward.lambda_uav_progress}")
    print(f"  λ_uav_progress-:  {config.reward.lambda_uav_backtrack}")
    print(f"  bootstrap scale:  {config.reward.bootstrap_progress_scale}")
    print(f"  λ_spatial_revisit:{config.reward.lambda_spatial_revisit}")
    print(f"  energy_fail:      {config.reward.terminal_failure_penalty}")
    print(f"  q_ref:            {config.reward.q_ref}")
    print(f"  Target NMSE:      {config.reward.accuracy_target_nmse} (diagnostic only)")
    print("=" * 60 + "\n")
