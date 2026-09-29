#!/usr/bin/env bash
set -euo pipefail

PKG=/home/zsj/works/work1/code/du_iibtd_based_fading_delta
ROOT="$PKG/formal_runs/latest_complete_p2_fading_delta_uimp002_repeat005_192k_s42"
mkdir -p "$ROOT"

if [[ -s "$ROOT/driver.pid" ]]; then
  old_pid="$(cat "$ROOT/driver.pid")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "formal training is already running as PID $old_pid" >&2
    exit 1
  fi
fi

nohup bash "$PKG/run_full_training_latest_complete.sh" >"$ROOT/driver.log" 2>&1 &
pid=$!
echo "$pid" >"$ROOT/driver.pid"
echo "$pid"
