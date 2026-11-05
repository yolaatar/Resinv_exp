#!/bin/bash
# Run a list of trainings one after the other on one GPU.
#
# Usage: bash run_queue.sh <gpu> <model_key>:<fold> [<model_key>:<fold> ...]
#   e.g. bash run_queue.sh 0 control:0 control:1 da5:0
#
#   EVAL_AFTER=1     after each successful training, run the standard evaluation
#                    (run_eval_paper.sh) on the same GPU before the next training starts.
#   WAIT_FOR=<file>  don't start until <file> exists and none of our evaluate_nnunet.py /
#                    recompute_metrics.py processes are left, i.e. the job already on the GPU
#                    has finished. Checked every 5 minutes.
#
# A failed job is logged and the queue moves on, so one crash doesn't idle the GPU all night.
# Summary at the end, and in ~/resinv_exp/nnunet_paper/logs/queue_gpu<gpu>.log.

GPU="${1:?usage: run_queue.sh <gpu> <model_key>:<fold> ...}"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QLOG="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}/logs/queue_gpu${GPU}.log"
mkdir -p "$(dirname "${QLOG}")"

if [ -n "${WAIT_FOR:-}" ]; then
    echo "[$(date '+%F %T')] WAIT  for ${WAIT_FOR} and no running evaluation" | tee -a "${QLOG}"
    while [ ! -f "${WAIT_FOR}" ] || pgrep -u "$(id -u)" -f "evaluate_nnunet.py|recompute_metrics.py" >/dev/null; do
        sleep 300
    done
    echo "[$(date '+%F %T')] GO" | tee -a "${QLOG}"
fi

ok=(); failed=()
for job in "$@"; do
    key="${job%%:*}"; fold="${job##*:}"
    echo "[$(date '+%F %T')] START ${key} fold ${fold} (GPU ${GPU})" | tee -a "${QLOG}"
    if bash "${HERE}/train_paper.sh" "${key}" "${fold}" "${GPU}"; then
        ok+=("${job}"); echo "[$(date '+%F %T')] OK    ${job}" | tee -a "${QLOG}"
        if [ "${EVAL_AFTER:-0}" = 1 ]; then
            echo "[$(date '+%F %T')] EVAL  ${job}" | tee -a "${QLOG}"
            if bash "${HERE}/run_eval_paper.sh" "${key}" "${fold}" "${GPU}"; then
                echo "[$(date '+%F %T')] EVAL OK   ${job}" | tee -a "${QLOG}"
            else
                failed+=("${job}(eval)"); echo "[$(date '+%F %T')] EVAL FAIL ${job}" | tee -a "${QLOG}"
            fi
        fi
    else
        failed+=("${job}"); echo "[$(date '+%F %T')] FAIL  ${job} (see logs/train_${key}_fold${fold}.log)" | tee -a "${QLOG}"
    fi
done

echo "[$(date '+%F %T')] Queue done on GPU ${GPU}. OK: ${ok[*]:-none}. FAILED: ${failed[*]:-none}" | tee -a "${QLOG}"
[ "${#failed[@]}" -eq 0 ]
