#!/bin/bash
# Native test on TEM2: single-resolution, multi-resolution and canonical resampling, five folds each,
# on the 21 annotated TEM2 ROIs (acquired at 4.93 nm, never seen in training), at the native size and
# a short sweep from there (run_eval_paper.sh, DATASET=tem2). Inference only: about 15 short runs.
#
# Usage (tassan, in tmux, after set_slot <gpu>):  bash run_tem2_native.sh <gpu>
# Results: ~/resinv_exp/results_paper/tem2/<model>_f<fold>/results.csv
# Log:     ~/resinv_exp/nnunet_paper/logs/tem2_native.log

GPU="${1:?usage: run_tem2_native.sh <gpu>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}/logs/tem2_native.log"
mkdir -p "$(dirname "${LOG}")"
export DATASET=tem2

failed=()
for fold in 0 1 2 3 4; do
    for run in control multires canonical; do
        echo "[$(date '+%F %T')] START ${run} fold ${fold}" | tee -a "${LOG}"
        if [ "${run}" = canonical ]; then
            CANONICAL_PX=0.00236 bash "${HERE}/run_eval_paper.sh" control "${fold}" "${GPU}"
        else
            bash "${HERE}/run_eval_paper.sh" "${run}" "${fold}" "${GPU}"
        fi
        rc=$?
        if [ "${rc}" -eq 0 ]; then
            echo "[$(date '+%F %T')] OK    ${run} fold ${fold}" | tee -a "${LOG}"
        else
            failed+=("${run}:${fold}"); echo "[$(date '+%F %T')] FAIL  ${run} fold ${fold} (exit ${rc})" | tee -a "${LOG}"
        fi
    done
done
echo "[$(date '+%F %T')] Done. FAILED: ${failed[*]:-none}" | tee -a "${LOG}"
[ "${#failed[@]}" -eq 0 ]
