#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/home/zsj/works/work1/code
PKG_ROOT="$CODE_ROOT/du_iibtd_based_fading_delta"
PY=/home/zsj/miniconda3/envs/work1/bin/python
RUN_ROOT="$PKG_ROOT/formal_runs/astar4_quant_cleanobs_h160_uimp008_192k_s42"
FROZEN_ROOT="$RUN_ROOT/frozen_test_seed200042_n3"
mkdir -p "$RUN_ROOT" "$FROZEN_ROOT" "$RUN_ROOT/console_logs"

SCENES="8513,1807,1579,1371,10001,FARMOmniDPM100PSD_251:905,FARMOmniDPM100PSD_251:43,FARMOmniDPM100PSD_251:705"
COMMON=(
  --variant quant
  --num_envs 8
  --total_timesteps 192000
  --episode_max_steps 160
  --num_minibatches 4
  --num_epochs 6
  --lr_actor 0.0001
  --lr_critic 0.0001
  --seed 42
  --device cuda:0
  --vec_backend subproc
  --radioseer_scene_indices "$SCENES"
  --uav_max_energy 9000
  --alpha_nmse 20
  --nmse_signed_clip 0.25
  --gamma_queue 1.5
  --lambda_uav_progress 2.0
  --lambda_uav_backtrack 2.0
  --lambda_ugv_progress 0.0
  --lambda_ugv_backtrack 0.0
  --lambda_novel_info 0
  --lambda_full_repeat 0.05
  --ensemble_refresh_interval 3
  --ensemble_full_refresh_interval 0
  --nmse_refresh_delta 0.1
  --incremental_outer_iters 2
  --incremental_max_svt_iters 20
  --ensemble_quality_weighted true
  --target_arrival_radius_steps 1
  --target_suppression_radius_steps 1
  --target_stuck_no_sample_steps 4
  --planner_target_mode hybrid
  --initial_observation_mode prefill
  --local_planner_radius 15
  --hybrid_uncertainty_stall_steps 2
  --hybrid_uncertainty_improvement_threshold 0.08
  --hybrid_global_hold_intervals 5
  --hybrid_local_reentry_min_targets 2
  --ugv_comm_target_mode path_corridor
  --ugv_comm_backlog_threshold 0.5
  --ugv_recovery_exit_backlog_threshold 0.2
  --ugv_recovery_poor_service_steps 2
  --ugv_recovery_min_hold_steps 2
  --ugv_recovery_good_link_steps 2
  --ugv_recovery_service_margin 1.0
  --ugv_comm_local_path_horizon 15
  --ugv_comm_expanded_path_horizon 20
  --ugv_comm_corridor_width 1
  --ugv_service_action_mask false
  --obs_remaining_time false
  --obs_quant_context false
  --quantization_scheme log_first
  --prefill_percent 5
  --prefill_budget_basis 200
  --init_building_clearance 5
  --bootstrap_building_clearance 5
  --flush_reconstruction_on_episode_end false
  --iibtd_backend du_iibtd_res_sr_learn_nu
  --iibtd_device cuda:0
  --log_interval 10
  --eval_interval 10
  --eval_episodes 1
  --eval_seed_stride 10000
  --save_interval 10
  --export_figures false
  --data_per_sample 8000000
  --building_height_m 25
)

write_state() {
  local status="$1"
  local method="${2:-}"
  printf '{"updated_at":"%s","status":"%s","method":"%s"}\n' \
    "$(date --iso-8601=seconds)" "$status" "$method" >"$RUN_ROOT/current_state.json"
}

run_learning() {
  local name="$1"
  local entry="$2"
  local trainer="$3"
  local out="$RUN_ROOT/$name"
  mkdir -p "$out/logs" "$out/checkpoints"

  if [[ ! -s "$out/checkpoints/final_model.pt" ]]; then
    write_state training "$name"
    echo "[TRAIN START] $name"
    "$PY" -u "$PKG_ROOT/$entry/train.py" \
      "${COMMON[@]}" \
      --log_dir "$out/logs" \
      --model_dir "$out/checkpoints" \
      >"$RUN_ROOT/console_logs/$name.train.log" 2>&1
    test -s "$out/checkpoints/final_model.pt"
    echo "[TRAIN PASS] $name"
  else
    echo "[TRAIN SKIP COMPLETE] $name"
  fi

  local config
  config="$(find "$out/logs" -type f -name config.json | sort | tail -n 1)"
  test -s "$config"
  if [[ ! -s "$FROZEN_ROOT/$name.json" ]]; then
    write_state frozen_eval "$name"
    echo "[FROZEN EVAL START] $name"
    "$PY" -u -m du_iibtd_based_fading_delta.shared.evaluate_checkpoint \
      --variant quant \
      --trainer "$trainer" \
      --checkpoint "$out/checkpoints/final_model.pt" \
      --config "$config" \
      --output "$FROZEN_ROOT/$name.json" \
      --device cuda:0 \
      --iibtd_device cuda:0 \
      --num_episodes 3 \
      --max_steps 160 \
      --seed_base 200042 \
      --scene_seed_stride 10000 \
      >"$RUN_ROOT/console_logs/$name.eval.log" 2>&1
    test -s "$FROZEN_ROOT/$name.json"
    echo "[FROZEN EVAL PASS] $name"
  fi
}

cd "$CODE_ROOT"
write_state started ""

run_learning ppo_astar_target_quant PPO_AStar ppo_astar
run_learning ppo_astar_support_quant PPO_AStar_Support ppo_astar_support
run_learning ppo_astar_2path_support_quant PPO_AStar_2PathSupport ppo_astar_2path_support
run_learning ppo_astar_2path_support_recovery_lite_quant PPO_AStar_2PathSupportRecovery ppo_astar_2path_support_recovery

write_state complete ""
touch "$RUN_ROOT/COMPLETE"
echo "[ASTAR4 COMPLETE] $RUN_ROOT"
