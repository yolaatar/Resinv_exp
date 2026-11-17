#!/bin/bash
# Hybrid: canonical resampling applied to the multi-resolution model, and its tolerance to wrong
# pixel-size metadata, against canonical resampling of the single-resolution model.
#
# Canonical resampling brings every image to one target pixel size before inference. Told a pixel
# size k times the true one, it shows the model the image as if at target / k. A model therefore
# tolerates the metadata errors that keep target / k inside its own robust band:
#   control  (band 1.74-4.00 nm, target 2.36 nm):  k ~ 0.6 to 1.4  (measured, Section canonical)
#   multires (band 1.74-21.3 nm, target 6.06 nm):  k ~ 0.28 to 3.5 (predicted from the sweep)
# 6.06 nm is the grid size nearest the geometric centre of the multires band (6.09 nm).
#
# Runs, all through run_eval_paper.sh (same scoring as every other result):
#   1. multires, canonical to 6.06 nm, no error, full TEM1 grid, fold 0
#   2. multires, canonical to 6.06 nm, errors k (below), at true 2.36, 4.93, 10, 21.3 nm, fold 0
#   3. control, canonical to 2.36 nm, the extra large errors k = 3, 4, 5, 10 (0.1-2 already done)
#   4. multires, canonical to 6.06 nm, no error, TEM2 (native 4.93 nm), folds 0-4
# A run whose results.csv exists is skipped, so the queue can be relaunched after an interruption.
#
# Usage (tassan, in tmux, after set_slot <gpu>):  bash run_hybrid.sh <gpu>
# Log: ~/resinv_exp/nnunet_paper/logs/hybrid.log

GPU="${1:?usage: run_hybrid.sh <gpu>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}"
RESULTS="${HOME}/resinv_exp/results_paper"
LOG="${BASE}/logs/hybrid.log"
mkdir -p "$(dirname "${LOG}")"

TARGET_MULTIRES=0.00606
ERR_PX="0.00236 0.00493 0.01 0.0213"
K_ALL="0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.67 0.8 0.9 1.1 1.25 1.5 2 3 4 5 10"
K_EXTRA_CONTROL="3 4 5 10"

failed=()
run() {  # run <name for the log> <results dir> <env assignments...> -- <run_eval_paper args...>
    local name="$1" out="$2"; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    if [ -f "${out}/results.csv" ]; then
        echo "[$(date '+%F %T')] SKIP  ${name} (done)" | tee -a "${LOG}"; return
    fi
    echo "[$(date '+%F %T')] START ${name}" | tee -a "${LOG}"
    if env "${envs[@]}" bash "${HERE}/run_eval_paper.sh" "$@"; then
        echo "[$(date '+%F %T')] OK    ${name}" | tee -a "${LOG}"
    else
        failed+=("${name}"); echo "[$(date '+%F %T')] FAIL  ${name}" | tee -a "${LOG}"
    fi
}

# 1. error-free hybrid, full grid
run "multires canonical 6.06 nm, full grid" "${RESULTS}/tem1/multires_canonical_f0" \
    CANONICAL_PX=${TARGET_MULTIRES} -- multires 0 "${GPU}"

# 2. hybrid with metadata errors
for k in ${K_ALL}; do
    run "multires canonical 6.06 nm, k=${k}" "${RESULTS}/tem1/multires_canonical_err${k}_f0" \
        CANONICAL_PX=${TARGET_MULTIRES} CANONICAL_PX_ERROR=${k} "PX_SIZES=${ERR_PX}" -- multires 0 "${GPU}"
done

# 3. control, larger errors than the existing 0.1-2 runs
for k in ${K_EXTRA_CONTROL}; do
    run "control canonical 2.36 nm, k=${k}" "${RESULTS}/tem1/control_canonical_err${k}_f0" \
        CANONICAL_PX=0.00236 CANONICAL_PX_ERROR=${k} "PX_SIZES=${ERR_PX}" -- control 0 "${GPU}"
done

# 4. hybrid on TEM2, five folds
for fold in 0 1 2 3 4; do
    run "multires canonical 6.06 nm, TEM2 fold ${fold}" "${RESULTS}/tem2/multires_canonical_f${fold}" \
        DATASET=tem2 CANONICAL_PX=${TARGET_MULTIRES} -- multires "${fold}" "${GPU}"
done

echo "[$(date '+%F %T')] Done. FAILED: ${failed[*]:-none}" | tee -a "${LOG}"
[ "${#failed[@]}" -eq 0 ]
