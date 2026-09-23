#!/bin/bash
# Run a list of trainings one after the other on one GPU.
#
# Usage: bash run_queue.sh <gpu> <model_key>:<fold> [<model_key>:<fold> ...]
#   e.g. bash run_queue.sh 0 witness:0 witness:1 da5:0
#
# A failed job is logged and the queue moves on, so one crash doesn't idle the GPU all night.
# Summary at the end, and in ~/resinv_exp/nnunet_paper/logs/queue_gpu<gpu>.log.

GPU="${1:?usage: run_queue.sh <gpu> <model_key>:<fold> ...}"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QLOG="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}/logs/queue_gpu${GPU}.log"
mkdir -p "$(dirname "${QLOG}")"

ok=(); failed=()
for job in "$@"; do
    key="${job%%:*}"; fold="${job##*:}"
    echo "[$(date '+%F %T')] START ${key} fold ${fold} (GPU ${GPU})" | tee -a "${QLOG}"
    if bash "${HERE}/train_paper.sh" "${key}" "${fold}" "${GPU}"; then
        ok+=("${job}"); echo "[$(date '+%F %T')] OK    ${job}" | tee -a "${QLOG}"
    else
        failed+=("${job}"); echo "[$(date '+%F %T')] FAIL  ${job} (see logs/train_${key}_fold${fold}.log)" | tee -a "${QLOG}"
    fi
done

echo "[$(date '+%F %T')] Queue done on GPU ${GPU}. OK: ${ok[*]:-none}. FAILED: ${failed[*]:-none}" | tee -a "${QLOG}"
[ "${#failed[@]}" -eq 0 ]
