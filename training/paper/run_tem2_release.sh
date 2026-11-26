#!/bin/bash
# Native test on the annotated images of the TEM2 release (DANDI 001350): the 10 images with manual masks
# (8 full 4675 x 4675 images from 4 mice + 2 ROIs from sub-373C), as a complement to the 21-ROI TEM2 test
# set. Single-resolution, multi-resolution (multires_fine) and canonical resampling, five folds, same
# 7-size grid as DATASET=tem2. Fold 0 of all three first, so the first comparison is available early.
#
# Usage (tassan, in tmux, after set_slot <gpu>):  bash run_tem2_release.sh <gpu>
# Needs:   ~/resinv_exp/data/tem2_release_gt   (python prepare_tem2_release_gt.py ...)
# Results: ~/resinv_exp/results_paper/tem2release/<model>_f<fold>/results.csv
# Log:     ~/resinv_exp/nnunet_paper/logs/tem2_release.log

GPU="${1:?usage: run_tem2_release.sh <gpu>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}/logs/tem2_release.log"
mkdir -p "$(dirname "${LOG}")"
export DATASET=tem2release

failed=()
for fold in 0 1 2 3 4; do
    for run in control multires_fine canonical; do
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
