#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/home/zsj/works/work1/code
PKG_ROOT="$CODE_ROOT/du_iibtd_based_fading_delta"
PY=/home/zsj/miniconda3/envs/work1/bin/python
RUN_ROOT="$PKG_ROOT/formal_runs/latest_complete_p2_fading_delta_uimp002_repeat005_192k_s42"
FROZEN_ROOT="$RUN_ROOT/frozen_test_seed200042_n3"
mkdir -p "$RUN_ROOT" "$FROZEN_ROOT"

SCENES="8513,1807,1579,1371,10001,FARMOmniDPM100PSD_251:905,FARMOmniDPM100PSD_251:43,FARMOmniDPM100PSD_251:705"
COMMON=(
  --num_envs 8
  --total_timesteps 192000
  --episode_max_steps 200
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
  --lambda_novel_info 0
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
  --hybrid_uncertainty_improvement_threshold 0.02
  --hybrid_global_hold_intervals 5
  --hybrid_local_reentry_min_targets 2
  --ugv_comm_target_mode path_corridor
  --ugv_comm_backlog_threshold 0.5
  --ugv_comm_local_path_horizon 15
  --ugv_comm_expanded_path_horizon 20
  --ugv_comm_corridor_width 1
  --ugv_service_action_mask false
  --obs_remaining_time true
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
  local variant="$3"
  local trainer="$4"
  local ugv_weight="$5"
  local full_repeat_weight="$6"
  local out="$RUN_ROOT/$name"
  mkdir -p "$out/logs" "$out/checkpoints" "$RUN_ROOT/console_logs"

  if [[ ! -s "$out/checkpoints/final_model.pt" ]]; then
    write_state training "$name"
    echo "[TRAIN START] $name"
    local variant_args=()
    if [[ "$entry" == "HAPPO" || "$entry" == "MAPPO_CF" || "$entry" == PPO_AStar* ]]; then
      variant_args=(--variant "$variant")
    fi
    local quant_args=()
    if [[ "$variant" == "quant" ]]; then
      quant_args=(--obs_quant_context true --quantization_scheme log_first)
    fi
    "$PY" -u "$PKG_ROOT/$entry/train.py" \
      "${variant_args[@]}" \
      "${COMMON[@]}" \
      --lambda_ugv_progress "$ugv_weight" \
      --lambda_ugv_backtrack "$ugv_weight" \
      --lambda_full_repeat "$full_repeat_weight" \
      "${quant_args[@]}" \
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
      --variant "$variant" \
      --trainer "$trainer" \
      --checkpoint "$out/checkpoints/final_model.pt" \
      --config "$config" \
      --output "$FROZEN_ROOT/$name.json" \
      --device cuda:0 \
      --iibtd_device cuda:0 \
      --num_episodes 3 \
      --max_steps 200 \
      --seed_base 200042 \
      --scene_seed_stride 10000 \
      >"$RUN_ROOT/console_logs/$name.eval.log" 2>&1
    test -s "$FROZEN_ROOT/$name.json"
    echo "[FROZEN EVAL PASS] $name"
  fi
}

cd "$CODE_ROOT"
write_state started ""

run_learning mappo_quant MAPPO_quant quant mappo 2.0 0.0
run_learning mappo_noquant MAPPO_noquant noquant mappo 2.0 0.0
run_learning ippo_quant IPPO_quant quant ippo 2.0 0.0
run_learning ippo_noquant IPPO_noquant noquant ippo 2.0 0.0
run_learning happo_quant HAPPO quant happo 2.0 0.0
run_learning happo_noquant HAPPO noquant happo 2.0 0.0
run_learning mappo_cf_quant MAPPO_CF quant mappo_cf 2.0 0.0
run_learning mappo_cf_noquant MAPPO_CF noquant mappo_cf 2.0 0.0
run_learning ppo_fixed_quant PPO_UGV_fixed_quant quant ppo_fixed 0.0 0.0
run_learning ppo_fixed_noquant PPO_UGV_fixed noquant ppo_fixed 0.0 0.0
run_learning ppo_heuristic_quant PPO_heuristic_quant_shareMember quant ppo_heuristic 0.0 0.0
run_learning ppo_heuristic_noquant PPO_heuristic_shareMember noquant ppo_heuristic 0.0 0.0
run_learning ppo_astar_target_quant PPO_AStar quant ppo_astar 0.0 0.05
run_learning ppo_astar_target_noquant PPO_AStar noquant ppo_astar 0.0 0.05
run_learning ppo_astar_support_quant PPO_AStar_Support quant ppo_astar_support 0.0 0.05
run_learning ppo_astar_support_noquant PPO_AStar_Support noquant ppo_astar_support 0.0 0.05
run_learning ppo_astar_2path_support_quant PPO_AStar_2PathSupport quant ppo_astar_2path_support 0.0 0.05
run_learning ppo_astar_2path_support_noquant PPO_AStar_2PathSupport noquant ppo_astar_2path_support 0.0 0.05

QCONFIG="$(find "$RUN_ROOT/ppo_astar_2path_support_quant/logs" -type f -name config.json | sort | tail -n 1)"
NCONFIG="$(find "$RUN_ROOT/ppo_astar_2path_support_noquant/logs" -type f -name config.json | sort | tail -n 1)"

run_baseline() {
  local name="$1"
  local entry="$2"
  local variant="$3"
  local config="$4"
  if [[ -s "$FROZEN_ROOT/$name.json" ]]; then
    echo "[BASELINE SKIP COMPLETE] $name"
    return
  fi
  write_state baseline_eval "$name"
  echo "[BASELINE START] $name"
  local variant_args=()
  if [[ "$entry" == "Greedy_AStar_2PathSupport" ]]; then
    variant_args=(--variant "$variant")
  fi
  "$PY" -u "$PKG_ROOT/$entry/train.py" \
    "${variant_args[@]}" \
    --config "$config" \
    --output "$FROZEN_ROOT/$name.json" \
    --device cuda:0 \
    --iibtd-device cuda:0 \
    --num-episodes 3 \
    --max-steps 200 \
    --seed-base 200042 \
    --scene-seed-stride 10000 \
    --local-planner-radius 15 \
    >"$RUN_ROOT/console_logs/$name.eval.log" 2>&1
  test -s "$FROZEN_ROOT/$name.json"
  echo "[BASELINE PASS] $name"
}

run_baseline total_random_quant TotalRandom_quant quant "$QCONFIG"
run_baseline total_random_noquant TotalRandom_noquant noquant "$NCONFIG"
run_baseline greedy_heuristic_quant Greedy_heuristic_quant quant "$QCONFIG"
run_baseline greedy_heuristic_noquant Greedy_heuristic_noquant noquant "$NCONFIG"
run_baseline greedy_astar_2path_support_quant Greedy_AStar_2PathSupport quant "$QCONFIG"
run_baseline greedy_astar_2path_support_noquant Greedy_AStar_2PathSupport noquant "$NCONFIG"

write_state complete ""
touch "$RUN_ROOT/COMPLETE"
echo "[FORMAL BATCH COMPLETE] $RUN_ROOT"
