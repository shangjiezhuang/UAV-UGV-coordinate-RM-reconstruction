"""Small utility layer for Greedy UAV-UGV evaluation."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from config import Config


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _eval_scene_metadata(
    env,
    seed_base: Optional[int],
) -> Dict[str, Any]:
    sim_data = getattr(env, "sim_data", None)
    raw_data = getattr(sim_data, "_data", {})
    scene_data = raw_data.get("config", {}) if isinstance(raw_data, dict) else {}

    def _int_or_default(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    actual_scene_source = str(scene_data.get("scene_source", env.config.scene.scene_source))
    actual_sample_index = _int_or_default(
        scene_data.get("sample_index", env.config.scene.radioseer_sample_index)
    )

    i_mask = getattr(env, "I_mask", getattr(sim_data, "I_mask", None))
    non_building_mask = getattr(sim_data, "non_building_mask", i_mask)
    i_mask_arr = np.asarray(i_mask, dtype=bool) if i_mask is not None else np.asarray([], dtype=bool)
    non_building_mask_arr = (
        np.asarray(non_building_mask, dtype=bool)
        if non_building_mask is not None
        else np.asarray([], dtype=bool)
    )
    mask_is_non_building = bool(
        i_mask_arr.size > 0
        and non_building_mask_arr.shape == i_mask_arr.shape
        and np.array_equal(i_mask_arr, non_building_mask_arr)
    )
    eval_nmse_mask = "non_building/I_mask" if mask_is_non_building else "custom_or_full_map"

    return {
        "eval_scene_source": actual_scene_source,
        "eval_scene_sample_index": actual_sample_index,
        "eval_scene_sample_tag": str(scene_data.get("sample_tag", "")),
        "eval_reset_seed_base": _int_or_default(seed_base),
        "eval_nmse_mask": eval_nmse_mask,
        "eval_nmse_mask_is_non_building": int(mask_is_non_building),
        "eval_nmse_mask_cells": int(np.sum(i_mask_arr)),
    }


class MetricsLogger:
    """Minimal JSON metrics logger used by the Greedy runner."""

    def __init__(self, log_dir: str, metadata: Optional[Dict[str, Any]] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.eval_results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.start_time = time.time()

    def log_evaluation(self, eval_results: Dict[str, Any]) -> None:
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 60)
        print(f"Greedy evaluation complete ({elapsed:.1f}s):")
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        print("=" * 60 + "\n")

        self.eval_results = dict(eval_results)

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            path = os.path.join(self.log_dir, "metrics.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"evaluation": dict(self.eval_results)}
        if self.metadata:
            payload.update(self.metadata)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=json_default)
        return path


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _resolve_eval_max_steps(env, max_steps: Optional[int]) -> int:
    if max_steps is not None:
        return int(max_steps)

    env_config = getattr(env, "config", None)
    for section_name in ("mappo", "ippo", "run"):
        section = getattr(env_config, section_name, None)
        if section is not None and hasattr(section, "episode_max_steps"):
            return int(section.episode_max_steps)
    raise AttributeError("Unable to infer episode_max_steps from env.config")


def _eval_progress_interval(max_steps: int) -> int:
    """Print at most about ten progress heartbeats per episode."""
    return max(1, int(max_steps) // 10)


def _print_eval_progress(
    label: str,
    episode_idx: int,
    num_episodes: int,
    step_idx: int,
    max_steps: int,
    episode_return: float,
    info: Dict[str, Any],
    *,
    final: bool = False,
) -> None:
    step_num = int(step_idx) + 1
    prefix = "done" if final else "step"
    ensemble_suffix = ""
    if int(info.get("ensemble_triggered", 0)):
        reason = str(info.get("ensemble_reason", ""))
        mode = str(info.get("ensemble_recon_mode", ""))
        ensemble_suffix = f" | ensemble={reason or 'triggered'}:{mode or '-'}"
    print(
        f"[Eval][{label}] ep {episode_idx + 1}/{num_episodes} {prefix} "
        f"{step_num}/{max_steps} | return {float(episode_return):.3f} "
        f"| nmse {float(info.get('nmse', float('nan'))):.6f} "
        f"| energy {float(info.get('uav_energy', float('nan'))):.1f} "
        f"| queue {float(info.get('queue_size', float('nan'))):.0f}"
        f"{ensemble_suffix}",
        flush=True,
    )


def evaluate_policy(
    env,
    policy,
    num_episodes: int = 5,
    max_steps: Optional[int] = None,
    seed_base: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate the deterministic Greedy policy on one environment."""
    if int(num_episodes) <= 0:
        raise ValueError("num_episodes must be positive")

    max_steps = _resolve_eval_max_steps(env, max_steps)
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    num_episodes = int(num_episodes)
    max_steps = int(max_steps)
    progress_interval = _eval_progress_interval(max_steps)
    print(
        f"[Eval][Greedy] starting {num_episodes} episode(s), "
        f"max_steps={max_steps}, progress_interval={progress_interval}",
        flush=True,
    )
    visualized_episode_index = int(num_episodes) - 1
    visualized_reset_seed = (
        None if seed_base is None else int(seed_base + visualized_episode_index)
    )

    returns: List[float] = []
    final_nmse: List[float] = []
    episode_steps: List[int] = []
    energy_remaining: List[float] = []
    data_delivered_bits: List[float] = []
    novel_data_delivered_bits: List[float] = []
    data_transmitted_bits: List[float] = []
    uav_move_dist: List[float] = []
    ugv_move_dist: List[float] = []
    uav_move_steps: List[float] = []
    ugv_move_steps: List[float] = []

    last_uav_traj: List[np.ndarray] = []
    last_ugv_traj: List[np.ndarray] = []
    last_step_details: Dict[str, List[Any]] = {}

    for episode_idx in range(int(num_episodes)):
        reset_seed = None if seed_base is None else int(seed_base + episode_idx)
        obs, _ = env.reset(seed=reset_seed)
        print(
            f"[Eval][Greedy] ep {episode_idx + 1}/{num_episodes} start"
            + ("" if reset_seed is None else f" | seed={reset_seed}"),
            flush=True,
        )
        episode_return = 0.0
        ep_data_delivered = 0.0
        ep_novel_data_delivered = 0.0
        ep_data_transmitted = 0.0
        ep_uav_move_dist = 0.0
        ep_ugv_move_dist = 0.0
        ep_uav_move_steps = 0.0
        ep_ugv_move_steps = 0.0

        uav_traj = [env.uav_pos.copy()]
        ugv_traj = [env.ugv_pos.copy()]
        step_details: Dict[str, List[Any]] = {}
        ensemble_events: List[Dict[str, Any]] = []

        for step_idx in range(int(max_steps)):
            action_data = policy.get_single_action(
                uav_obs=obs["uav_obs"],
                critic_state=obs["critic_state"],
                uav_action_mask=obs["uav_action_mask"],
                deterministic=True,
            )
            greedy_plan = dict(policy.last_plan)
            uav_action = int(action_data["uav_action"])
            decoded_action = env._decode_uav_action(uav_action)
            if len(decoded_action) == 3:
                uav_direction, uav_bw_choice_idx, uav_quant_choice_idx = decoded_action
            else:
                uav_direction, uav_bw_choice_idx = decoded_action
                uav_quant_choice_idx = -1

            obs, rewards, terminated, truncated, info = env.step(uav_action)
            reward = float(rewards["team_reward"])
            episode_return += reward

            uav_traj.append(env.uav_pos.copy())
            ugv_traj.append(env.ugv_pos.copy())

            step_details.setdefault("uav_action", []).append(uav_action)
            step_details.setdefault("uav_direction", []).append(uav_direction)
            step_details.setdefault("uav_bw_choice_idx", []).append(uav_bw_choice_idx)
            step_details.setdefault("uav_quant_choice_idx", []).append(uav_quant_choice_idx)
            step_details.setdefault("greedy_plan_strategy", []).append(
                str(greedy_plan.get("mode", ""))
            )
            step_details.setdefault("greedy_planned_score", []).append(
                float(greedy_plan.get("planned_score", float("nan")))
            )
            step_details.setdefault("greedy_first_action_uncertainty", []).append(
                float(greedy_plan.get("first_action_uncertainty", float("nan")))
            )
            step_details.setdefault("greedy_planned_action_count", []).append(
                len(greedy_plan.get("planned_actions", []) or [])
            )
            step_details.setdefault("bw_ratio", []).append(info["bw_ratio"])

            for key in (
                "step",
                "nmse",
                "uav_energy",
                "queue_size",
                "snr_db",
                "quant_bits",
                "current_quant_norm",
                "target_grid_x",
                "target_grid_y",
                "target_center_freq",
                "target_source",
                "executed_target_grid_x",
                "executed_target_grid_y",
                "executed_target_center_freq",
                "executed_target_source",
                "target_reached",
                "target_retargeted",
                "target_retarget_reason",
                "planner_initialized",
                "planner_submode",
                "planner_mode_switch",
                "ensemble_triggered",
                "ensemble_reason",
                "ensemble_recon_mode",
                "ensemble_full_refresh_due",
                "ensemble_nmse_refresh_triggered",
                "ensemble_nmse_degradation",
                "data_delivered_bits",
                "novel_data_delivered_bits",
                "data_transmitted_bits",
                "uav_move_dist",
                "ugv_move_dist",
                "uav_move_steps",
                "ugv_move_steps",
                "sample_center_freq",
                "sample_novelty_ratio",
                "sample_repeat_ratio",
                "bootstrap_active",
                "bootstrap_target_reached",
                "bootstrap_handoff",
                "bootstrap_event",
            ):
                step_details.setdefault(key, []).append(info[key])

            if int(info["ensemble_triggered"]):
                ensemble_events.append(
                    {
                        "step": int(step_idx + 1),
                        "reason": str(info["ensemble_reason"]),
                        "recon_mode": str(info["ensemble_recon_mode"]),
                        "full_refresh_due": int(info["ensemble_full_refresh_due"]),
                        "nmse_refresh_triggered": int(info["ensemble_nmse_refresh_triggered"]),
                        "nmse_refresh_delta": float(info["ensemble_nmse_refresh_delta"]),
                        "nmse_refresh_reference_before": float(
                            info["ensemble_nmse_refresh_reference_before"]
                        ),
                        "nmse_refresh_reference_after": float(
                            info["ensemble_nmse_refresh_reference_after"]
                        ),
                        "nmse_degradation": float(info["ensemble_nmse_degradation"]),
                        "nmse": float(info["ensemble_event_nmse"]),
                        "nmse_delta": float(info["ensemble_event_nmse_delta"]),
                        "uav_pos": [float(env.uav_pos[0]), float(env.uav_pos[1])],
                        "ugv_pos": [float(env.ugv_pos[0]), float(env.ugv_pos[1])],
                        "target_grid_x": int(info["target_grid_x"]),
                        "target_grid_y": int(info["target_grid_y"]),
                        "target_center_freq": int(info["target_center_freq"]),
                    }
                )

            ep_data_delivered += float(info["data_delivered_bits"])
            ep_novel_data_delivered += float(info["novel_data_delivered_bits"])
            ep_data_transmitted += float(info["data_transmitted_bits"])
            ep_uav_move_dist += float(info["uav_move_dist"])
            ep_ugv_move_dist += float(info["ugv_move_dist"])
            ep_uav_move_steps += float(info["uav_move_steps"])
            ep_ugv_move_steps += float(info["ugv_move_steps"])

            should_print_progress = (
                ((step_idx + 1) % progress_interval == 0)
                or bool(terminated)
                or bool(truncated)
                or bool(info.get("ensemble_triggered", 0))
            )
            if should_print_progress:
                _print_eval_progress(
                    "Greedy",
                    episode_idx,
                    num_episodes,
                    step_idx,
                    max_steps,
                    episode_return,
                    info,
                    final=bool(terminated) or bool(truncated),
                )

            if bool(terminated) or bool(truncated):
                break

        returns.append(float(episode_return))
        final_nmse.append(float(info["nmse"]))
        episode_steps.append(int(step_idx + 1))
        energy_remaining.append(float(info["uav_energy"]))
        data_delivered_bits.append(ep_data_delivered)
        novel_data_delivered_bits.append(ep_novel_data_delivered)
        data_transmitted_bits.append(ep_data_transmitted)
        uav_move_dist.append(ep_uav_move_dist)
        ugv_move_dist.append(ep_ugv_move_dist)
        uav_move_steps.append(ep_uav_move_steps)
        ugv_move_steps.append(ep_ugv_move_steps)

        last_uav_traj = uav_traj
        last_ugv_traj = ugv_traj
        step_details["ensemble_events"] = ensemble_events
        step_details["bootstrap_events"] = list(env.bootstrap_events)
        step_details["completed_target_nmse_records"] = list(env.completed_target_nmse_records)
        last_step_details = step_details

    best_nmse_idx = int(np.argmin(final_nmse))
    worst_nmse_idx = int(np.argmax(final_nmse))

    results: Dict[str, Any] = {
        **_eval_scene_metadata(env, seed_base),
        "eval_mean_return": _mean(returns),
        "eval_std_return": _std(returns),
        # NMSE lower is better. Transmission paired with best/worst NMSE stays on the same episode.
        "eval_best_nmse": float(np.min(final_nmse)),
        "eval_worst_nmse": float(np.max(final_nmse)),
        "eval_mean_nmse": _mean(final_nmse),
        "eval_std_nmse": _std(final_nmse),
        "eval_num_episodes": int(num_episodes),
        "eval_visualized_episode_index": visualized_episode_index,
        "eval_visualized_reset_seed": visualized_reset_seed,
        "eval_mean_steps": _mean(episode_steps),
        "eval_mean_uav_energy_remaining": _mean(energy_remaining),
        "eval_mean_data_delivered_bits": _mean(data_delivered_bits),
        "eval_mean_novel_data_delivered_bits": _mean(novel_data_delivered_bits),
        "eval_data_transmitted_bits_at_best_nmse": float(data_transmitted_bits[best_nmse_idx]),
        "eval_data_transmitted_bits_at_worst_nmse": float(data_transmitted_bits[worst_nmse_idx]),
        "eval_min_data_transmitted_bits": float(np.min(data_transmitted_bits)),
        "eval_max_data_transmitted_bits": float(np.max(data_transmitted_bits)),
        "eval_mean_data_transmitted_bits": _mean(data_transmitted_bits),
        "eval_mean_uav_move_dist": _mean(uav_move_dist),
        "eval_mean_ugv_move_dist": _mean(ugv_move_dist),
        "eval_mean_uav_move_steps": _mean(uav_move_steps),
        "eval_mean_ugv_move_steps": _mean(ugv_move_steps),
        "eval_uav_trajectory": [pos.tolist() for pos in last_uav_traj],
        "eval_ugv_trajectory": [pos.tolist() for pos in last_ugv_traj],
        "eval_step_details": last_step_details,
    }
    return results


def set_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def print_config_summary(config: Config) -> None:
    print("\n" + "=" * 60)
    print("Greedy UAV-UGV Configuration")
    print("=" * 60)
    print(f"Scene grid:        {config.scene.grid_size[0]} x {config.scene.grid_size[1]}")
    print(f"RadioSeer root:   {config.scene.radioseer_root}")
    print(f"RadioSeer sample: {config.scene.radioseer_sample_index}")
    print(f"Freq bands:       {config.scene.total_freq_bands_nums}")
    print(f"UAV step size:    {config.uav.step_size}")
    print(f"UGV step size:    {config.ugv.step_size}")
    print(f"BW choices:       {config.uav.bandwidth_ratios}")
    if hasattr(config.uav, "quant_bits"):
        print(f"Quant choices:    {config.uav.quant_bits}")
        print(f"Default quant:    {config.uav.default_quant_bits}")
    print(f"Queue capacity:   {config.uav.queue_capacity_packets}")
    print(f"Episode steps:    {config.mappo.episode_max_steps}")
    print(f"Seed:             {config.mappo.seed}")
    print(f"Planner mode:     {config.planner.target_mode}")
    print(f"Init mode:        {config.planner.initial_observation_mode}")
    print(f"Local radius:     {config.planner.local_planner_radius}")
    print(f"Ensemble interval:{config.planner.ensemble_refresh_interval}")
    print(f"Ensemble size:    {config.planner.ensemble_size}")
    print("Ensemble obs:     shared_all")
    print(f"Ensemble kernel:  {config.planner.ensemble_kernel_bandwidth_mode} (delta {config.planner.ensemble_kernel_bandwidth_delta})")
    print(f"Ensemble jitter:  {config.planner.ensemble_init_jitter_scale}")
    print(f"II-BTD backend:   {config.planner.iibtd_backend}")
    print(f"II-BTD device:    {config.planner.iibtd_device}")
    if "du_iibtd" in str(config.planner.iibtd_backend).strip().lower():
        print("DU-IIBTD solver params: loaded from checkpoint config")
    else:
        print(f"II-BTD mu / nu:   {config.planner.iibtd_mu} / {config.planner.iibtd_nu}")
        print(f"II-BTD kernel bw: {config.planner.iibtd_kernel_bandwidth}")
    du_iibtd_models = (
        ", ".join(
            os.path.basename(os.path.dirname(os.path.dirname(str(path))))
            for path in config.planner.du_iibtd_checkpoints
        )
        if config.planner.iibtd_backend in {"du_iibtd_res_sr", "du_iibtd_res_sr_learn_nu"}
        else "inactive"
    )
    print(f"DU-IIBTD models:  {du_iibtd_models}")
    print(f"Target NMSE:      {config.reward.accuracy_target_nmse} (diagnostic only)")
    print("=" * 60 + "\n")
