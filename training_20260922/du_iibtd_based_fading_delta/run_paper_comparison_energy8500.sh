#!/usr/bin/env bash
set -Eeuo pipefail
# Methods used in paper/realMain.tex: 12 learned runs + 4 baseline evaluations.
# All outputs use a separate directory; historical scripts remain archival.

CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_ROOT="$CODE_ROOT/du_iibtd_based_fading_delta"
PY=/home/zsj/miniconda3/envs/work1/bin/python
RUN_ROOT="$PKG_ROOT/formal_runs/paper_coupled_sum03_maskfix_grid2m_greedyLocal15_len022_uncapped_bw50_ref32_split1200_heightavoid_power091215_uav50_h47to53int_e8500_h160_192k_s42"
FROZEN_ROOT="$RUN_ROOT/frozen_test_seed200042_n3"
mkdir -p "$RUN_ROOT" "$FROZEN_ROOT" "$RUN_ROOT/console_logs"

SCENES="8513,1807,1579,1371,10001,FARMOmniDPM100PSD_251:905,FARMOmniDPM100PSD_251:43,FARMOmniDPM100PSD_251:705"
COMMON=(
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
  --uav_max_energy 8500
  --grid_spacing 2
  --tx_power_choices_dbm 9 12 15
  --tx_power_dbm 12
  --outage_snr_mode received
  --los_excess_db 1.6
  --nlos_excess_db 23
  --nlos_length_loss_db_per_m 0.22
  --nlos_length_loss_cap_db none
  --alpha_nmse 20
  --nmse_signed_clip 0.25
  --gamma_queue 1.5
  --lambda_uav_progress 2.0
  --lambda_uav_backtrack 2.0
  --lambda_novel_info 0
  --ensemble_refresh_interval 3
  --incremental_outer_iters 2
  --incremental_max_svt_iters 20
  --ensemble_quality_weighted true
  --target_arrival_radius_steps 1
  --target_suppression_radius_steps 1
  --target_stuck_no_sample_steps 4
  --planner_target_mode hybrid
  --initial_observation_mode prefill
  --local_planner_radius 15
  --hybrid_uncertainty_window_updates 2
  --hybrid_uncertainty_improvement_threshold 0.03
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
  --total_bandwidth 50000000
  --source_measurement_bits 32
  --data_per_sample 8000000
  --uav_height 50
  --ugv_height 0
  --building_height_m 47
  --building_height_mode uniform_integer_per_building
  --building_height_min_m 47
  --building_height_max_m 53
  --building_height_seed 42
  --building_split_min_area_m2 1200
)

write_state() {
  local status="$1"
  local method="${2:-}"
  ACTIVE_METHOD="$method"
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
  mkdir -p "$out/logs" "$out/checkpoints"

  local variant_args=()
  if [[ "$entry" == "HAPPO" || "$entry" == "MAPPO_CF" || "$entry" == PPO_AStar* ]]; then
    variant_args=(--variant "$variant")
  fi
  local quant_args=()
  if [[ "$variant" == "quant" ]]; then
    quant_args=(--obs_quant_context false --quantization_scheme log_first)
  fi

  if [[ ! -s "$out/checkpoints/final_model.pt" ]]; then
    write_state training "$name"
    echo "[TRAIN START] $name"
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
  "$PY" "$PKG_ROOT/validate_clean160_contract.py" \
    --config "$config" \
    --train-log "$RUN_ROOT/console_logs/$name.train.log" \
    --variant "$variant" \
    --trainer "$trainer" \
    --ugv-weight "$ugv_weight" \
    --full-repeat-weight "$full_repeat_weight" \
    --grid-spacing 2
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
      --max_steps 160 \
      --grid_spacing 2 \
      --seed_base 200042 \
      --scene_seed_stride 10000 \
      >"$RUN_ROOT/console_logs/$name.eval.log" 2>&1
    test -s "$FROZEN_ROOT/$name.json"
    echo "[FROZEN EVAL PASS] $name"
  fi
}

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
  local variant_args=()
  if [[ "$entry" == "Greedy_AStar_2PathSupport" ]]; then
    variant_args=(--variant "$variant")
  fi
  echo "[BASELINE START] $name"
  "$PY" -u "$PKG_ROOT/$entry/train.py" \
    "${variant_args[@]}" \
    --config "$config" \
    --output "$FROZEN_ROOT/$name.json" \
    --device cuda:0 \
    --iibtd-device cuda:0 \
    --num-episodes 3 \
    --max-steps 160 \
    --seed-base 200042 \
    --scene-seed-stride 10000 \
    --local-planner-radius 15 \
    --grid-spacing 2 \
    >"$RUN_ROOT/console_logs/$name.eval.log" 2>&1
  test -s "$FROZEN_ROOT/$name.json"
  echo "[BASELINE PASS] $name"
}

cd "$CODE_ROOT"
exec 9>"$RUN_ROOT/run.lock"
flock -n 9 || { echo "This paper comparison is already running" >&2; exit 1; }
echo "$$" >"$RUN_ROOT/driver.pid"
ACTIVE_METHOD=""
trap 'exit_code=$?; write_state failed "${ACTIVE_METHOD:-}"; echo "[PAPER COMPARISON FAILED] exit=$exit_code method=${ACTIVE_METHOD:-}" >&2; exit "$exit_code"' ERR
write_state started ""

run_learning ppo_astar_2path_support_recovery_lite_quant PPO_AStar_2PathSupportRecovery quant ppo_astar_2path_support_recovery 0.0 0.05
run_learning ppo_astar_2path_support_recovery_lite_noquant PPO_AStar_2PathSupportRecovery noquant ppo_astar_2path_support_recovery 0.0 0.05
run_learning mappo_quant MAPPO_quant quant mappo 2.0 0.0
run_learning mappo_noquant MAPPO_noquant noquant mappo 2.0 0.0
run_learning ippo_quant IPPO_quant quant ippo 2.0 0.0
run_learning ippo_noquant IPPO_noquant noquant ippo 2.0 0.0
run_learning happo_quant HAPPO quant happo 2.0 0.0
run_learning happo_noquant HAPPO noquant happo 2.0 0.0
run_learning ppo_astar_target_quant PPO_AStar quant ppo_astar 0.0 0.05
run_learning ppo_astar_target_noquant PPO_AStar noquant ppo_astar 0.0 0.05
run_learning ppo_fixed_quant PPO_UGV_fixed_quant quant ppo_fixed 0.0 0.0
run_learning ppo_fixed_noquant PPO_UGV_fixed noquant ppo_fixed 0.0 0.0

QCONFIG="$(find "$RUN_ROOT/ppo_astar_2path_support_recovery_lite_quant/logs" -type f -name config.json | sort | tail -n 1)"
NCONFIG="$(find "$RUN_ROOT/ppo_astar_2path_support_recovery_lite_noquant/logs" -type f -name config.json | sort | tail -n 1)"
test -s "$QCONFIG"
test -s "$NCONFIG"

run_baseline total_random_quant TotalRandom_quant quant "$QCONFIG"
run_baseline total_random_noquant TotalRandom_noquant noquant "$NCONFIG"
run_baseline greedy_astar_2path_support_quant Greedy_AStar_2PathSupport quant "$QCONFIG"
run_baseline greedy_astar_2path_support_noquant Greedy_AStar_2PathSupport noquant "$NCONFIG"

write_state complete ""
touch "$RUN_ROOT/COMPLETE"
echo "[PAPER 16-METHOD COMPARISON COMPLETE] $RUN_ROOT"
