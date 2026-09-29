#!/usr/bin/env bash
set -euo pipefail

SESSION=full_2psr_q008
RUNNER=/home/zsj/works/work1/code/du_iibtd_based_fading_delta/PPO_AStar_2PathSupportRecovery/run_full_quant_uimp008_192k_s42.sh

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already running: $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash '$RUNNER'"
echo "started tmux session: $SESSION"

