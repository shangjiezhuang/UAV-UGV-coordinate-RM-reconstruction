"""
Shared noquant utilities for training, evaluation, logging, and visualization.
"""

import os
import json
import time
import numpy as np
from typing import Any, Dict, List, Optional
from collections import defaultdict

from du_iibtd_based_fading_delta.shared.noquant.config import Config
from du_iibtd_based_fading_delta.shared.noquant.environment import UAVUGVEnvironment
from du_iibtd_based_fading_delta.scene_suite import data_accounting_violation_bits
from du_iibtd_based_fading_delta.ugv_control import uses_service_action_mask
from du_iibtd_based_fading_delta.evaluation_common import (
    add_cumulative_step_metrics,
    eval_scene_metadata as _eval_scene_metadata,
    json_default,
    resolve_eval_max_steps as _resolve_eval_max_steps,
    set_seeds,
    zero_machine_roundoff,
)


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
        policy: Any learned or non-learning policy exposing get_single_action.
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
    all_terminal_failures = []
    all_data_produced_bits = []
    all_data_delivered_bits = []
    all_novel_data_delivered_bits = []
    all_completed_packet_bits = []
    all_link_transmitted_bits = []
    all_dropped_bits = []
    all_final_queue_bits = []
    all_prefill_observed_band_units = []
    all_prefill_equivalent_data_bits = []
    all_data_accounting_violation_bits = []
    all_outage_ratios = []
    all_ugv_comm_target_ratios = []
    all_uav_move_dist = []
    all_ugv_move_dist = []
    all_uav_move_steps = []
    all_ugv_move_steps = []
    all_uav_stay_ratios = []
    all_ugv_stay_ratios = []
    all_energy_limited_ratios = []
    all_uav_motion_energy = []
    all_uav_sensing_energy = []
    all_uav_step_energy = []
    episode_records: List[Dict[str, Any]] = []

    # Reward components tracking
    all_r_nmse = []
    all_r_queue = []
    all_r_progress = []
    all_r_uav_progress = []
    all_r_ugv_progress = []
    all_r_novel_info = []
    all_r_full_repeat = []
    all_sample_novelty_ratios = []
    all_full_repeat_ratios = []
    all_nmse_target_gap = []

    # Keep one concrete episode for visualization: the final episode in this eval batch.
    last_uav_traj = []
    last_ugv_traj = []
    last_step_details = {}
    episode_artifacts = []

    for ep in range(num_episodes):
        reset_seed = None if seed_base is None else int(seed_base + ep)
        obs, reset_info = env.reset(seed=reset_seed)
        on_episode_reset = getattr(policy, "on_episode_reset", None)
        if callable(on_episode_reset):
            on_episode_reset(reset_seed=reset_seed, env=env)
        all_prefill_observed_band_units.append(
            float(reset_info.get("prefill_observed_band_units", 0.0))
        )
        all_prefill_equivalent_data_bits.append(
            float(reset_info.get("prefill_equivalent_data_bits", 0.0))
        )
        episode_return = 0

        # Per-episode reward accumulators
        ep_r_nmse = 0
        ep_r_queue = 0
        ep_r_progress = 0
        ep_r_uav_progress = 0
        ep_r_ugv_progress = 0
        ep_r_novel_info = 0
        ep_r_full_repeat = 0
        ep_sample_novelty_sum = 0.0
        ep_full_repeat_steps = 0
        ep_data_produced_bits = 0.0
        ep_data_delivered_bits = 0.0
        ep_novel_data_delivered_bits = 0.0
        ep_completed_packet_bits = 0.0
        ep_link_transmitted_bits = 0.0
        ep_dropped_bits = 0.0
        ep_ugv_comm_target_steps = 0
        ep_uav_move_dist = 0.0
        ep_ugv_move_dist = 0.0
        ep_uav_move_steps = 0.0
        ep_ugv_move_steps = 0.0
        ep_uav_stay_steps = 0
        ep_ugv_stay_steps = 0
        ep_energy_limited_steps = 0
        ep_uav_motion_energy = 0.0
        ep_uav_sensing_energy = 0.0
        ep_uav_step_energy = 0.0

        uav_traj = [env.uav_pos.copy()]
        ugv_traj = [env.ugv_pos.copy()]

        # Per-step detail trackers
        step_queue_size = []
        step_snr_db = []
        step_large_scale_snr_db = []
        step_snr_outage_threshold_db = []
        step_channel_outage = []
        step_channel_capacity = []
        step_channel_shannon_capacity = []
        step_bw_ratio = []
        step_sample_center_freq = []
        step_target_grid_x = []
        step_target_grid_y = []
        step_target_center_freq = []
        step_executed_target_grid_x = []
        step_executed_target_grid_y = []
        step_executed_target_center_freq = []
        step_executed_target_source = []
        step_executed_ugv_guidance_target_x = []
        step_executed_ugv_guidance_target_y = []
        step_executed_ugv_guidance_target_source = []
        step_executed_ugv_guidance_target_has_los = []
        step_ugv_comm_target_active = []
        step_ugv_control_submode = []
        step_ugv_control_mode_switch = []
        step_ugv_recovery_trigger_reason = []
        step_ugv_recovery_poor_service_steps = []
        step_ugv_recovery_good_link_steps = []
        step_ugv_recovery_service_margin_ok = []
        step_sensing_band_num = []
        step_sensing_bw_units = []
        step_comm_bw_units = []
        step_uav_ugv_dist = []
        step_uav_action = []
        step_uav_direction = []
        step_uav_bw_choice_idx = []
        step_ugv_commanded_action = []
        step_ugv_action = []
        step_r_nmse = []
        step_r_queue = []
        step_r_progress = []
        step_r_uav_progress = []
        step_r_ugv_progress = []
        step_r_novel_info = []
        step_r_full_repeat = []
        step_full_repeat_sample = []
        step_ugv_building_clearance_norm = []
        step_ugv_building_clearance_deficit = []
        step_nmse = []
        step_nmse_target_gap = []
        step_target_nmse = []
        step_planner_initialized = []
        step_planner_submode = []
        step_planner_mode_switch = []
        step_planner_switched_to_global = []
        step_target_reached = []
        step_target_source = []
        step_target_retargeted = []
        step_target_retarget_reason = []
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
        step_data_produced_bits = []
        step_data_delivered_bits = []
        step_novel_data_delivered_bits = []
        step_completed_packet_bits = []
        step_novel_completed_packet_bits = []
        step_link_transmitted_bits = []
        step_novel_link_transmitted_bits = []
        step_uav_move_dist = []
        step_ugv_move_dist = []
        step_uav_move_steps = []
        step_ugv_move_steps = []
        step_uav_motion_energy = []
        step_uav_sensing_power = []
        step_uav_sensing_energy = []
        step_uav_step_energy = []
        step_global_top_fields = {
            f"global_top{rank}_{field}": []
            for rank in range(1, 4)
            for field in ("x", "y", "freq", "score")
        }
        ensemble_events = []

        for step in range(max_steps):
            actual_uav_mask = np.asarray(obs["uav_action_mask"], dtype=bool)
            geometric_uav_mask = env._uav_action_mask_cache.get(
                env._uav_action_mask_cache_key()
            )
            if geometric_uav_mask is not None:
                ep_energy_limited_steps += int(
                    np.any(np.asarray(geometric_uav_mask, dtype=bool) & ~actual_uav_mask)
                )
            # Get deterministic actions
            action_data = policy.get_single_action(
                uav_obs=obs["uav_obs"],
                ugv_obs=obs["ugv_obs"],
                critic_state=obs["critic_state"],
                uav_action_mask=obs["uav_action_mask"],
                ugv_action_mask=obs["ugv_action_mask"],
                deterministic=True,
            )
            uav_action = int(action_data["uav_action"])
            ugv_action = int(action_data["ugv_action"])
            uav_direction, uav_bw_choice_idx = env._decode_uav_action(uav_action)
            ep_uav_stay_steps += int(uav_direction == 0)

            obs, rewards, terminated, truncated, info = env.step(
                uav_action,
                ugv_action,
            )
            executed_ugv_action = int(info.get("ugv_executed_action", ugv_action))
            ep_ugv_stay_steps += int(executed_ugv_action == 0)
            episode_return += rewards["team_reward"]

            uav_traj.append(env.uav_pos.copy())
            ugv_traj.append(env.ugv_pos.copy())

            # Record per-step details
            step_queue_size.append(info["queue_size"])
            step_snr_db.append(info["snr_db"])
            step_large_scale_snr_db.append(info["large_scale_snr_db"])
            step_snr_outage_threshold_db.append(info["snr_outage_threshold_db"])
            step_channel_outage.append(info["channel_outage"])
            step_channel_capacity.append(info["channel_capacity"])
            step_channel_shannon_capacity.append(info["channel_shannon_capacity"])
            step_bw_ratio.append(info["bw_ratio"])
            step_sample_center_freq.append(info["sample_center_freq"])
            step_target_grid_x.append(info["target_grid_x"])
            step_target_grid_y.append(info["target_grid_y"])
            step_target_center_freq.append(info["target_center_freq"])
            step_executed_target_grid_x.append(info["executed_target_grid_x"])
            step_executed_target_grid_y.append(info["executed_target_grid_y"])
            step_executed_target_center_freq.append(
                info["executed_target_center_freq"]
            )
            step_executed_target_source.append(info["executed_target_source"])
            step_executed_ugv_guidance_target_x.append(
                info["executed_ugv_guidance_target_x"]
            )
            step_executed_ugv_guidance_target_y.append(
                info["executed_ugv_guidance_target_y"]
            )
            step_executed_ugv_guidance_target_source.append(
                info["executed_ugv_guidance_target_source"]
            )
            step_executed_ugv_guidance_target_has_los.append(
                info["executed_ugv_guidance_target_has_los"]
            )
            step_ugv_comm_target_active.append(info["ugv_comm_target_active"])
            step_ugv_control_submode.append(info.get("ugv_control_submode", ""))
            step_ugv_control_mode_switch.append(
                info.get("ugv_control_mode_switch", "")
            )
            step_ugv_recovery_trigger_reason.append(
                info.get("ugv_recovery_trigger_reason", "")
            )
            step_ugv_recovery_poor_service_steps.append(
                info.get("ugv_recovery_poor_service_steps", 0)
            )
            step_ugv_recovery_good_link_steps.append(
                info.get("ugv_recovery_good_link_steps", 0)
            )
            step_ugv_recovery_service_margin_ok.append(
                info.get("ugv_recovery_service_margin_ok", 1)
            )
            step_sensing_band_num.append(info["sensing_band_num"])
            step_sensing_bw_units.append(info["sensing_bw_units"])
            step_comm_bw_units.append(info["comm_bw_units"])
            dist = np.sqrt(np.sum((env.uav_pos - env.ugv_pos) ** 2))
            step_uav_ugv_dist.append(float(dist))
            step_uav_action.append(uav_action)
            step_uav_direction.append(uav_direction)
            step_uav_bw_choice_idx.append(uav_bw_choice_idx)
            step_ugv_commanded_action.append(ugv_action)
            step_ugv_action.append(executed_ugv_action)
            step_r_nmse.append(info["r_nmse"])
            step_r_queue.append(info["r_queue"])
            step_r_progress.append(info["r_progress"])
            step_r_uav_progress.append(info["r_uav_progress"])
            step_r_ugv_progress.append(info["r_ugv_progress"])
            step_r_novel_info.append(info["r_novel_info"])
            step_r_full_repeat.append(info["r_full_repeat"])
            step_full_repeat_sample.append(info["full_repeat_sample"])
            step_ugv_building_clearance_norm.append(info["ugv_building_clearance_norm"])
            step_ugv_building_clearance_deficit.append(
                info["ugv_building_clearance_deficit"]
            )
            step_nmse.append(info["nmse"])
            step_nmse_target_gap.append(info["nmse_target_gap"])
            step_target_nmse.append(info["target_nmse"])
            step_planner_initialized.append(info["planner_initialized"])
            step_planner_submode.append(info["planner_submode"])
            step_planner_mode_switch.append(info["planner_mode_switch"])
            step_planner_switched_to_global.append(info["planner_switched_to_global"])
            step_target_reached.append(info["target_reached"])
            step_target_source.append(info["target_source"])
            step_target_retargeted.append(info["target_retargeted"])
            step_target_retarget_reason.append(info["target_retarget_reason"])
            step_bootstrap_active.append(info["bootstrap_active"])
            step_bootstrap_target_reached.append(info["bootstrap_target_reached"])
            step_bootstrap_handoff.append(info["bootstrap_handoff"])
            step_bootstrap_event.append(info["bootstrap_event"])
            step_spatial_revisit_count.append(info["spatial_revisit_count"])
            step_sample_novelty_ratio.append(info["sample_novelty_ratio"])
            step_sample_repeat_ratio.append(info["sample_repeat_ratio"])
            step_data_produced_bits.append(info["data_produced_bits"])
            step_data_delivered_bits.append(info["data_delivered_bits"])
            step_novel_data_delivered_bits.append(info["novel_data_delivered_bits"])
            step_completed_packet_bits.append(info["completed_packet_bits"])
            step_novel_completed_packet_bits.append(info["novel_completed_packet_bits"])
            step_link_transmitted_bits.append(info["link_transmitted_bits"])
            step_novel_link_transmitted_bits.append(info["novel_link_transmitted_bits"])
            step_uav_move_dist.append(info["uav_move_dist"])
            step_ugv_move_dist.append(info["ugv_move_dist"])
            step_uav_move_steps.append(info["uav_move_steps"])
            step_ugv_move_steps.append(info["ugv_move_steps"])
            step_uav_motion_energy.append(info["uav_motion_energy"])
            step_uav_sensing_power.append(info["uav_sensing_power"])
            step_uav_sensing_energy.append(info["uav_sensing_energy"])
            step_uav_step_energy.append(info["uav_step_energy"])
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
                        "target_grid_x": int(info["ensemble_target_grid_x"]),
                        "target_grid_y": int(info["ensemble_target_grid_y"]),
                        "target_center_freq": int(info["ensemble_target_center_freq"]),
                    }
                )

            ep_r_nmse += info["r_nmse"]
            ep_r_queue += info["r_queue"]
            ep_r_progress += info["r_progress"]
            ep_r_uav_progress += info["r_uav_progress"]
            ep_r_ugv_progress += info["r_ugv_progress"]
            ep_r_novel_info += info["r_novel_info"]
            ep_r_full_repeat += info["r_full_repeat"]
            ep_sample_novelty_sum += info["sample_novelty_ratio"]
            ep_full_repeat_steps += int(info["full_repeat_sample"])
            ep_data_produced_bits += info["data_produced_bits"]
            ep_data_delivered_bits += info["data_delivered_bits"]
            ep_novel_data_delivered_bits += info["novel_data_delivered_bits"]
            ep_completed_packet_bits += info["completed_packet_bits"]
            ep_link_transmitted_bits += info["link_transmitted_bits"]
            ep_dropped_bits += info["dropped_bits"]
            ep_ugv_comm_target_steps += int(info["ugv_comm_target_active"])
            ep_uav_move_dist += info["uav_move_dist"]
            ep_ugv_move_dist += info["ugv_move_dist"]
            ep_uav_move_steps += info["uav_move_steps"]
            ep_ugv_move_steps += info["ugv_move_steps"]
            ep_uav_motion_energy += info["uav_motion_energy"]
            ep_uav_sensing_energy += info["uav_sensing_energy"]
            ep_uav_step_energy += info["uav_step_energy"]

            if terminated or truncated:
                break

        all_returns.append(episode_return)
        all_nmse.append(info["nmse"])
        all_steps.append(step + 1)
        all_energy_uav.append(info["uav_energy"])
        all_terminal_failures.append(int(info["terminal_failure"]))
        episode_steps = float(max(step + 1, 1))
        all_uav_stay_ratios.append(float(ep_uav_stay_steps) / episode_steps)
        all_ugv_stay_ratios.append(float(ep_ugv_stay_steps) / episode_steps)
        all_energy_limited_ratios.append(float(ep_energy_limited_steps) / episode_steps)
        all_uav_motion_energy.append(ep_uav_motion_energy)
        all_uav_sensing_energy.append(ep_uav_sensing_energy)
        all_uav_step_energy.append(ep_uav_step_energy)
        all_r_nmse.append(ep_r_nmse)
        all_r_queue.append(ep_r_queue)
        all_r_progress.append(ep_r_progress)
        all_r_uav_progress.append(ep_r_uav_progress)
        all_r_ugv_progress.append(ep_r_ugv_progress)
        all_r_novel_info.append(ep_r_novel_info)
        all_r_full_repeat.append(ep_r_full_repeat)
        all_sample_novelty_ratios.append(ep_sample_novelty_sum / episode_steps)
        all_full_repeat_ratios.append(float(ep_full_repeat_steps) / episode_steps)
        all_nmse_target_gap.append(info["nmse_target_gap"])
        all_data_produced_bits.append(ep_data_produced_bits)
        all_data_delivered_bits.append(ep_data_delivered_bits)
        all_novel_data_delivered_bits.append(ep_novel_data_delivered_bits)
        all_completed_packet_bits.append(ep_completed_packet_bits)
        all_link_transmitted_bits.append(ep_link_transmitted_bits)
        all_dropped_bits.append(ep_dropped_bits)
        final_queue_bits = float(info["queue_bits_after_tx"])
        all_final_queue_bits.append(final_queue_bits)
        accounting_violation_bits = data_accounting_violation_bits(
            produced_bits=ep_data_produced_bits,
            link_transmitted_bits=ep_link_transmitted_bits,
            completed_packet_bits=ep_completed_packet_bits,
            delivered_bits=ep_data_delivered_bits,
            dropped_bits=ep_dropped_bits,
            final_queue_bits=final_queue_bits,
        )
        all_data_accounting_violation_bits.append(accounting_violation_bits)
        scene_metadata = _eval_scene_metadata(env, seed_base)
        energy_budget_j = float(env.config.uav.max_energy)
        energy_remaining_j = float(info["uav_energy"])
        service_completion_ratio = (
            float(ep_completed_packet_bits) / float(ep_data_produced_bits)
            if ep_data_produced_bits > 1e-9
            else 1.0
        )
        episode_outage_ratio = float(np.mean(step_channel_outage))
        episode_records.append(
            {
                "episode_index": int(ep),
                "reset_seed": None if reset_seed is None else int(reset_seed),
                "training_seed": int(getattr(env.config.training, "seed", -1)),
                "scene_source": str(scene_metadata["eval_scene_source"]),
                "scene_sample_index": int(scene_metadata["eval_scene_sample_index"]),
                "scene_sample_tag": str(scene_metadata["eval_scene_sample_tag"]),
                "energy_budget_j": energy_budget_j,
                "energy_consumed_j": float(ep_uav_step_energy),
                "energy_remaining_j": energy_remaining_j,
                "energy_balance_error_j": zero_machine_roundoff(
                    energy_budget_j - ep_uav_step_energy - energy_remaining_j,
                    energy_budget_j,
                    ep_uav_step_energy,
                    energy_remaining_j,
                ),
                "nmse": float(info["nmse"]),
                "link_transmitted_bits": float(ep_link_transmitted_bits),
                "data_produced_bits": float(ep_data_produced_bits),
                "data_delivered_bits": float(ep_data_delivered_bits),
                "completed_packet_bits": float(ep_completed_packet_bits),
                "dropped_bits": float(ep_dropped_bits),
                "final_queue_bits": final_queue_bits,
                "service_completion_ratio": float(service_completion_ratio),
                "outage_ratio": episode_outage_ratio,
                "sample_novelty_ratio": float(ep_sample_novelty_sum / episode_steps),
                "full_repeat_ratio": float(ep_full_repeat_steps) / episode_steps,
                "data_accounting_violation_bits": float(accounting_violation_bits),
                "steps": int(step + 1),
                "terminal_failure": int(info["terminal_failure"]),
                "ugv_control_mode": str(info.get("ugv_control_mode", "policy")),
            }
        )
        all_outage_ratios.append(episode_outage_ratio)
        all_ugv_comm_target_ratios.append(
            float(ep_ugv_comm_target_steps) / float(max(step + 1, 1))
        )
        all_uav_move_dist.append(ep_uav_move_dist)
        all_ugv_move_dist.append(ep_ugv_move_dist)
        all_uav_move_steps.append(ep_uav_move_steps)
        all_ugv_move_steps.append(ep_ugv_move_steps)

        # Keep the final eval episode as the visualized artifact.
        last_uav_traj = uav_traj
        last_ugv_traj = ugv_traj
        last_step_details = {
            "queue_size": step_queue_size,
            "snr_db": step_snr_db,
            "large_scale_snr_db": step_large_scale_snr_db,
            "snr_outage_threshold_db": step_snr_outage_threshold_db,
            "channel_outage": step_channel_outage,
            "channel_capacity": step_channel_capacity,
            "channel_shannon_capacity": step_channel_shannon_capacity,
            "bw_ratio": step_bw_ratio,
            "sample_center_freq": step_sample_center_freq,
            "target_grid_x": step_target_grid_x,
            "target_grid_y": step_target_grid_y,
            "target_center_freq": step_target_center_freq,
            "executed_target_grid_x": step_executed_target_grid_x,
            "executed_target_grid_y": step_executed_target_grid_y,
            "executed_target_center_freq": step_executed_target_center_freq,
            "executed_target_source": step_executed_target_source,
            "executed_ugv_guidance_target_x": step_executed_ugv_guidance_target_x,
            "executed_ugv_guidance_target_y": step_executed_ugv_guidance_target_y,
            "executed_ugv_guidance_target_source": step_executed_ugv_guidance_target_source,
            "executed_ugv_guidance_target_has_los": step_executed_ugv_guidance_target_has_los,
            "ugv_comm_target_active": step_ugv_comm_target_active,
            "ugv_control_submode": step_ugv_control_submode,
            "ugv_control_mode_switch": step_ugv_control_mode_switch,
            "ugv_recovery_trigger_reason": step_ugv_recovery_trigger_reason,
            "ugv_recovery_poor_service_steps": step_ugv_recovery_poor_service_steps,
            "ugv_recovery_good_link_steps": step_ugv_recovery_good_link_steps,
            "ugv_recovery_service_margin_ok": step_ugv_recovery_service_margin_ok,
            "sensing_band_num": step_sensing_band_num,
            "sensing_bw_units": step_sensing_bw_units,
            "comm_bw_units": step_comm_bw_units,
            "uav_ugv_dist": step_uav_ugv_dist,
            "uav_action": step_uav_action,
            "uav_direction": step_uav_direction,
            "uav_bw_choice_idx": step_uav_bw_choice_idx,
            "ugv_commanded_action": step_ugv_commanded_action,
            "ugv_action": step_ugv_action,
            "r_nmse": step_r_nmse,
            "r_queue": step_r_queue,
            "r_progress": step_r_progress,
            "r_uav_progress": step_r_uav_progress,
            "r_ugv_progress": step_r_ugv_progress,
            "r_novel_info": step_r_novel_info,
            "r_full_repeat": step_r_full_repeat,
            "full_repeat_sample": step_full_repeat_sample,
            "ugv_building_clearance_norm": step_ugv_building_clearance_norm,
            "ugv_building_clearance_deficit": step_ugv_building_clearance_deficit,
            "nmse": step_nmse,
            "nmse_target_gap": step_nmse_target_gap,
            "target_nmse": step_target_nmse,
            "planner_initialized": step_planner_initialized,
            "planner_submode": step_planner_submode,
            "planner_mode_switch": step_planner_mode_switch,
            "planner_switched_to_global": step_planner_switched_to_global,
            "target_reached": step_target_reached,
            "target_source": step_target_source,
            "target_retargeted": step_target_retargeted,
            "target_retarget_reason": step_target_retarget_reason,
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
            "data_produced_bits": step_data_produced_bits,
            "data_delivered_bits": step_data_delivered_bits,
            "novel_data_delivered_bits": step_novel_data_delivered_bits,
            "completed_packet_bits": step_completed_packet_bits,
            "novel_completed_packet_bits": step_novel_completed_packet_bits,
            "link_transmitted_bits": step_link_transmitted_bits,
            "novel_link_transmitted_bits": step_novel_link_transmitted_bits,
            "uav_move_dist": step_uav_move_dist,
            "ugv_move_dist": step_ugv_move_dist,
            "uav_move_steps": step_uav_move_steps,
            "ugv_move_steps": step_ugv_move_steps,
            "uav_motion_energy": step_uav_motion_energy,
            "uav_sensing_power": step_uav_sensing_power,
            "uav_sensing_energy": step_uav_sensing_energy,
            "uav_step_energy": step_uav_step_energy,
            **step_global_top_fields,
            "ensemble_events": ensemble_events,
            "bootstrap_events": list(env.bootstrap_events),
            "completed_target_nmse_records": list(env.completed_target_nmse_records),
        }
        last_step_details = add_cumulative_step_metrics(last_step_details)
        episode_artifacts.append(
            {
                "episode_index": int(ep),
                "reset_seed": None if reset_seed is None else int(reset_seed),
                "steps": int(step + 1),
                "uav_trajectory": [pos.tolist() for pos in uav_traj],
                "ugv_trajectory": [pos.tolist() for pos in ugv_traj],
                "step_details": last_step_details,
            }
        )

    best_nmse_idx = int(np.argmin(all_nmse))
    worst_nmse_idx = int(np.argmax(all_nmse))
    min_link_bits_idx = int(np.argmin(all_link_transmitted_bits))
    service_completion_ratios = np.divide(
        np.asarray(all_completed_packet_bits, dtype=float),
        np.asarray(all_data_produced_bits, dtype=float),
        out=np.ones(len(all_completed_packet_bits), dtype=float),
        where=np.asarray(all_data_produced_bits, dtype=float) > 1e-9,
    )

    return {
        **_eval_scene_metadata(env, seed_base),
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
        "eval_min_steps": np.min(all_steps),
        "eval_full_horizon_rate": np.mean(np.asarray(all_steps) >= int(max_steps)),
        "eval_mean_uav_energy_remaining": np.mean(all_energy_uav),
        "eval_min_uav_energy_remaining": np.min(all_energy_uav),
        "eval_max_uav_energy_remaining": np.max(all_energy_uav),
        "eval_energy_failure_rate": np.mean(all_terminal_failures),
        "eval_mean_uav_stay_ratio": np.mean(all_uav_stay_ratios),
        "eval_mean_ugv_stay_ratio": np.mean(all_ugv_stay_ratios),
        "eval_mean_energy_limited_ratio": np.mean(all_energy_limited_ratios),
        "eval_mean_uav_motion_energy": np.mean(all_uav_motion_energy),
        "eval_mean_uav_sensing_energy": np.mean(all_uav_sensing_energy),
        "eval_mean_uav_step_energy": np.mean(all_uav_step_energy),
        "eval_mean_r_nmse": np.mean(all_r_nmse),
        "eval_mean_r_queue": np.mean(all_r_queue),
        "eval_mean_r_progress": np.mean(all_r_progress),
        "eval_mean_r_uav_progress": np.mean(all_r_uav_progress),
        "eval_mean_r_ugv_progress": np.mean(all_r_ugv_progress),
        "eval_mean_r_novel_info": np.mean(all_r_novel_info),
        "eval_mean_r_full_repeat": np.mean(all_r_full_repeat),
        "eval_mean_sample_novelty_ratio": np.mean(all_sample_novelty_ratios),
        "eval_mean_full_repeat_ratio": np.mean(all_full_repeat_ratios),
        "eval_mean_nmse_target_gap": np.mean(all_nmse_target_gap),
        "eval_mean_data_produced_bits": np.mean(all_data_produced_bits),
        "eval_mean_data_delivered_bits": np.mean(all_data_delivered_bits),
        "eval_mean_novel_data_delivered_bits": np.mean(all_novel_data_delivered_bits),
        "eval_data_delivered_bits_at_best_nmse": all_data_delivered_bits[best_nmse_idx],
        "eval_data_delivered_bits_at_worst_nmse": all_data_delivered_bits[worst_nmse_idx],
        "eval_min_data_delivered_bits": np.min(all_data_delivered_bits),
        "eval_max_data_delivered_bits": np.max(all_data_delivered_bits),
        "eval_mean_completed_packet_bits": np.mean(all_completed_packet_bits),
        "eval_mean_service_completion_ratio": np.mean(service_completion_ratios),
        "eval_min_service_completion_ratio": np.min(service_completion_ratios),
        "eval_service_completion_ratio_at_best_nmse": service_completion_ratios[best_nmse_idx],
        "eval_service_completion_ratio_at_worst_nmse": service_completion_ratios[worst_nmse_idx],
        "eval_mean_link_transmitted_bits": np.mean(all_link_transmitted_bits),
        "eval_mean_dropped_bits": np.mean(all_dropped_bits),
        "eval_mean_final_queue_bits": np.mean(all_final_queue_bits),
        "eval_min_link_transmitted_bits": np.min(all_link_transmitted_bits),
        "eval_max_link_transmitted_bits": np.max(all_link_transmitted_bits),
        "eval_link_transmitted_bits_at_best_nmse": all_link_transmitted_bits[best_nmse_idx],
        "eval_link_transmitted_bits_at_worst_nmse": all_link_transmitted_bits[worst_nmse_idx],
        "eval_nmse_at_min_link_transmitted_bits": all_nmse[min_link_bits_idx],
        "eval_mean_prefill_observed_band_units": np.mean(all_prefill_observed_band_units),
        "eval_mean_prefill_equivalent_data_bits": np.mean(all_prefill_equivalent_data_bits),
        "eval_mean_data_accounting_violation_bits": np.mean(all_data_accounting_violation_bits),
        "eval_max_data_accounting_violation_bits": np.max(all_data_accounting_violation_bits),
        "eval_mean_outage_ratio": np.mean(all_outage_ratios),
        "eval_outage_ratio_at_best_nmse": all_outage_ratios[best_nmse_idx],
        "eval_outage_ratio_at_worst_nmse": all_outage_ratios[worst_nmse_idx],
        "eval_mean_ugv_comm_target_ratio": np.mean(all_ugv_comm_target_ratios),
        "eval_mean_uav_move_dist": np.mean(all_uav_move_dist),
        "eval_mean_ugv_move_dist": np.mean(all_ugv_move_dist),
        "eval_mean_uav_move_steps": np.mean(all_uav_move_steps),
        "eval_mean_ugv_move_steps": np.mean(all_ugv_move_steps),
        "eval_episode_records": episode_records,
        "eval_episode_artifacts": episode_artifacts,
        # Artifacts from eval_visualized_episode_index.
        "eval_uav_trajectory": [pos.tolist() for pos in last_uav_traj],
        "eval_ugv_trajectory": [pos.tolist() for pos in last_ugv_traj],
        "eval_step_details": last_step_details,
    }


def print_config_summary(config: Config):
    """Print a readable summary of the configuration."""
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)

    print(f"\n--- Scene ---")
    print(f"  Grid: {config.scene.grid_size[0]}×{config.scene.grid_size[1]}")
    print(f"  Scene loader: {config.scene.scene_source}")
    print(f"  RadioSeer root: {config.scene.radioseer_root}")
    print(f"  RadioSeer sample: {config.scene.radioseer_sample_index}")
    print(f"  Freq bands: {config.scene.total_freq_bands_nums}")
    print(f"  UAV height: {config.scene.uav_height}m")

    print(f"\n--- UAV ---")
    print(f"  Directions: {config.uav.num_directions} (stay + 4 cardinals)")
    print(f"  Max energy: {config.uav.max_energy}J")
    print(f"  Total bandwidth: {config.uav.total_bandwidth/1e6:.0f} MHz")
    print(f"  Bandwidth units: {config.uav.total_bw_num} (each {config.uav.unit_bandwidth_hz/1e6:.1f} MHz)")
    print(f"  Default BW ratio: {config.uav.default_bw_ratio}")
    print(f"  BW ratio choices: {config.uav.bandwidth_ratios}")
    print(
        f"  Queue capacity: {float(config.uav.queue_capacity_bits) / 1e6:.1f} Mbit "
        "(hard bit limit and reward normalization reference)"
    )
    print(f"  Source payload bit-width reference: {config.comm.source_measurement_bits} bit")
    print(
        "  Source payload per sensed band/window: "
        f"{config.comm.data_per_sample / 1e6:.1f} Mbit"
    )
    print(f"  Sampling mode: current grid cell + GT noise (no numerical quantization)")
    print(f"  Outage SNR:     {config.comm.snr_outage_threshold_db} dB (large-scale)")

    print(f"\n--- Action Spaces ---")
    uav_actions = config.uav.num_directions * config.uav.num_bandwidth_ratios
    print(
        f"  UAV: direction({config.uav.num_directions}) "
        f"x bandwidth_select({config.uav.num_bandwidth_ratios}) = {uav_actions}"
    )
    print(f"  UGV: {config.ugv.num_directions}")

    print(f"\n--- Planner ---")
    print(f"  Target count:     {config.planner.target_count}")
    print(f"  Target mode:      {config.planner.target_mode}")
    print(f"  Init mode:        {config.planner.initial_observation_mode}")
    print(f"  Init pair d_max:  {config.planner.init_pair_max_distance}")
    print(f"  Local radius:     {config.planner.local_planner_radius}")
    print(f"  Hybrid uncertainty stall N: {config.planner.hybrid_uncertainty_stall_steps}")
    print(f"  Hybrid uncertainty improvement: {config.planner.hybrid_uncertainty_improvement_threshold}")
    print(f"  Hybrid global k:  {config.planner.hybrid_global_hold_intervals}")
    local_reentry = (
        max(
            int(config.planner.hybrid_local_reentry_min_targets),
            int(config.planner.target_count),
        )
    )
    print(f"  Hybrid reentry T: {local_reentry}")
    print(f"  UGV comm mode:    {config.planner.ugv_comm_target_mode}")
    print(f"  UGV backlog T:    {config.planner.ugv_comm_backlog_threshold}")
    print(
        f"  UGV path horizons:{config.planner.ugv_comm_local_path_horizon} -> "
        f"{config.planner.ugv_comm_expanded_path_horizon} road steps"
    )
    print(f"  UGV path corridor:{config.planner.ugv_comm_corridor_width}")
    print(f"  UGV controller:   {config.planner.ugv_control_mode}")
    print(
        f"  UGV service mask: {uses_service_action_mask(config)} "
        f"(configured={config.planner.ugv_service_action_mask})"
    )
    print(f"  Prefill percent:  {config.planner.prefill_percent}")
    prefill_basis = (
        config.planner.prefill_budget_basis
        if int(config.planner.prefill_budget_basis) > 0
        else config.training.episode_max_steps
    )
    print(f"  Prefill basis:    {prefill_basis}")
    print(f"  Init clearance:   {config.planner.init_building_clearance}")
    print(f"  Bootstrap clear.: {config.planner.bootstrap_building_clearance}")
    print(f"  Terminal flush:   {config.planner.flush_reconstruction_on_episode_end}")
    print(f"  Planner warmup M: {config.planner.min_samples_for_ensemble} effective grid samples")
    print(f"  Ensemble / map update interval: {config.planner.ensemble_refresh_interval}")
    print(f"  Ensemble size:    {config.planner.ensemble_size}")
    print(f"  Ensemble quality weighting: {config.planner.ensemble_quality_weighted}")
    print(f"  Full refresh interval: {config.planner.ensemble_full_refresh_interval}")
    print(f"  NMSE refresh delta: {config.planner.nmse_refresh_delta}")
    print(
        f"  Incremental iters / SVT: "
        f"{config.planner.incremental_outer_iters} / {config.planner.incremental_max_svt_iters}"
    )
    print(f"  DU-IIBTD backend:   {config.planner.iibtd_backend}")
    print(f"  DU-IIBTD device:    {config.planner.iibtd_device}")
    if "du_iibtd" in str(config.planner.iibtd_backend).strip().lower():
        print("  DU-IIBTD solver params: loaded from checkpoint config")
    else:
        print(f"  DU-IIBTD mu:        {config.planner.iibtd_mu}")
    print(
        "  DU-IIBTD checkpoints: "
        + ", ".join(
            os.path.basename(os.path.dirname(os.path.dirname(str(path))))
            for path in config.planner.du_iibtd_checkpoints
        )
    )

    print(f"\n--- Observation Ablations ---")
    print(f"  Remaining time:   {config.obs.include_remaining_time}")
    print("  Queue state:      normalized remaining bits (always on)")

    print("\n--- PPO Training ---")
    print(f"  Envs: {config.training.num_envs}")
    print(f"  Vec backend: {config.training.vec_backend}")
    print(f"  Total steps: {config.training.total_timesteps:,}")
    print(f"  Episode/rollout horizon: {config.training.episode_max_steps}")
    print(f"  LR (actor/critic): {config.training.lr_actor}/{config.training.lr_critic}")
    print(f"  Gamma: {config.training.gamma}, GAE λ: {config.training.gae_lambda}")
    print(f"  Eval seed stride: {config.training.eval_seed_stride}")

    print(f"\n--- Reward ---")
    print(f"  α_nmse:           {config.reward.alpha_nmse}")
    print(f"  δ_nmse clip:      {config.reward.nmse_signed_clip}")
    print("  Physical link bits: evaluation-only metric (no direct reward penalty)")
    print(f"  γ (queue bits):   {config.reward.gamma_queue}")
    print(f"  λ_uav_progress+:  {config.reward.lambda_uav_progress}")
    print(f"  λ_uav_progress-:  {config.reward.lambda_uav_backtrack}")
    print(f"  λ_ugv_progress+:  {config.reward.lambda_ugv_progress}")
    print(f"  λ_ugv_progress-:  {config.reward.lambda_ugv_backtrack}")
    print(f"  λ_novel_info:     {config.reward.lambda_novel_info}")
    print(f"  λ_full_repeat:    {config.reward.lambda_full_repeat}")
    print("  UGV progress ref: executed controller target shortest-road distance")
    print(f"  bootstrap scale:  {config.reward.bootstrap_progress_scale}")
    print(f"  UGV clearance observation radius: {config.obs.ugv_building_safe_clearance}")
    print("  queue norm:       remaining bits / worst-case bit reference")
    print(f"  Target NMSE:      {config.reward.accuracy_target_nmse} (diagnostic only)")
    print("=" * 60 + "\n")
