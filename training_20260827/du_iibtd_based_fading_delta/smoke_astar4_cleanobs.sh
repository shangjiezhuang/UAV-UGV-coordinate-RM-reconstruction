#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/home/zsj/works/work1/code
PKG_ROOT="$CODE_ROOT/du_iibtd_based_fading_delta"
PY=/home/zsj/miniconda3/envs/work1/bin/python
SMOKE_ROOT=/tmp/astar4_cleanobs_smoke_20260824

rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
trap 'rm -rf /tmp/astar4_cleanobs_smoke_20260824' EXIT

run_smoke() {
  local name="$1"
  local entry="$2"
  local out="$SMOKE_ROOT/$name"
  mkdir -p "$out/logs" "$out/models"
  "$PY" -u "$PKG_ROOT/$entry/train.py" \
    --variant quant \
    --num_envs 1 \
    --total_timesteps 4 \
    --episode_max_steps 4 \
    --num_minibatches 1 \
    --num_epochs 1 \
    --seed 420 \
    --device cuda:0 \
    --vec_backend sync \
    --radioseer_scene_indices 8513 \
    --hybrid_uncertainty_improvement_threshold 0.08 \
    --ugv_comm_backlog_threshold 0.5 \
    --ugv_recovery_exit_backlog_threshold 0.2 \
    --ugv_recovery_poor_service_steps 2 \
    --ugv_recovery_min_hold_steps 2 \
    --ugv_recovery_good_link_steps 2 \
    --ugv_recovery_service_margin 1.0 \
    --obs_remaining_time false \
    --obs_quant_context false \
    --initial_observation_mode prefill \
    --prefill_percent 5 \
    --prefill_budget_basis 200 \
    --iibtd_backend du_iibtd_res_sr_learn_nu \
    --iibtd_device cuda:0 \
    --eval_interval 0 \
    --save_interval 0 \
    --log_interval 1 \
    --export_figures false \
    --log_dir "$out/logs" \
    --model_dir "$out/models" \
    >"$out/smoke.log" 2>&1
  test -s "$out/models/final_model.pt"
  echo "[SMOKE PASS] $name"
  grep -E 'Observation dims|Action dims|Training complete|Final Eval' "$out/smoke.log" || true
}

cd "$CODE_ROOT"
run_smoke ppo_astar_target_quant PPO_AStar
run_smoke ppo_astar_support_quant PPO_AStar_Support
run_smoke ppo_astar_2path_support_quant PPO_AStar_2PathSupport
run_smoke ppo_astar_2path_support_recovery_lite_quant PPO_AStar_2PathSupportRecovery
