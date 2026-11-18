#!/bin/bash
# Wait for run_hybrid_target.sh to finish, then run the in-memory cost benchmark on the same GPU.
# Launch from tmux after set_slot <gpu>.   Usage: bash run_bench_after.sh <gpu> [mosaic n]
# Log: ~/resinv_exp/nnunet_paper/logs/bench_cost.log
GPU="${1:?usage: run_bench_after.sh <gpu> [mosaic]}"
MOSAIC="${2:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}/logs"
LOG="${LOGS}/bench_cost.log"
echo "[$(date '+%F %T')] waiting for run_hybrid_target.sh" | tee -a "${LOG}"
until grep -q "Done." "${LOGS}/hybrid_target.log" 2>/dev/null; do sleep 300; done
# shellcheck disable=SC1091
source "${RESINV_VENV:-${HOME}/resinv_exp/venv_resinv_v2}/bin/activate"
CUDA_VISIBLE_DEVICES="${GPU}" python "${HERE}/bench_cost.py" --mosaic "${MOSAIC}" --repeats 3 2>&1 | tee -a "${LOG}"
echo "[$(date '+%F %T')] benchmark finished (exit ${PIPESTATUS[0]})" | tee -a "${LOG}"
