#!/bin/bash
# Hybrid with another resampling target: canonical resampling of the multires model to TARGET (um/px),
# results tagged with TAG so they sit next to the 6.06 nm runs of run_hybrid.sh. TEM2 (five folds) first,
# then the TEM1 grid (fold 0). Finished runs are skipped.
#
# Why: on TEM2 every model gains from upsampling (about 0.925 at 2.36 nm), so the 6.06 nm target of the
# hybrid trades accuracy for tolerance. A 2.36 nm target should recover accuracy but only tolerate pixel
# sizes given as 0.11 to 1.36 times the true one (band 1.74-21.3 nm).
#
# Usage (tassan, tmux, after set_slot <gpu>):  bash run_hybrid_target.sh <gpu> [target_um] [tag]
#   default target 0.00236, tag _to236. Log: ~/resinv_exp/nnunet_paper/logs/hybrid_target.log

GPU="${1:?usage: run_hybrid_target.sh <gpu> [target_um] [tag]}"
TARGET="${2:-0.00236}"
TAG="${3:-_to236}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${HOME}/resinv_exp/results_paper"
LOG="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}/logs/hybrid_target.log"
mkdir -p "$(dirname "${LOG}")"

run() {  # run <dataset> <fold>
    local ds="$1" fold="$2" out="${RESULTS}/$1/multires_canonical${TAG}_f$2"
    if [ -f "${out}/results.csv" ]; then echo "[$(date '+%F %T')] SKIP  ${ds} fold ${fold}" | tee -a "${LOG}"; return; fi
    echo "[$(date '+%F %T')] START ${ds} fold ${fold}, target ${TARGET}" | tee -a "${LOG}"
    if DATASET="${ds}" CANONICAL_PX="${TARGET}" CANONICAL_TAG="${TAG}" bash "${HERE}/run_eval_paper.sh" multires "${fold}" "${GPU}"; then
        echo "[$(date '+%F %T')] OK    ${ds} fold ${fold}" | tee -a "${LOG}"
    else
        echo "[$(date '+%F %T')] FAIL  ${ds} fold ${fold}" | tee -a "${LOG}"
    fi
}
for fold in 0 1 2 3 4; do run tem2 "${fold}"; done
run tem1 0
echo "[$(date '+%F %T')] Done." | tee -a "${LOG}"
