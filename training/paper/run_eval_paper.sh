#!/bin/bash
# Evaluate one paper-batch model on both test sets at all 22 pixel sizes, then compute
# Dice against GT. Same protocol for every model so curves are directly comparable.
#
# Usage: bash run_eval_paper.sh <model_key> <fold> [gpu]
#
#   TEM1 test split  : 4 subjects from subject_split.json, native 2.36 nm
#   TEM2 held-out set: testset_armand_uaxon, native 4.93 nm, GT images only
#
# Uses checkpoint_final.pth for every model (not checkpoint_best): "best" is picked on
# nnUNet's internal validation pseudo-Dice at the native patch scale, which adds a selection
# step that differs per model. Final = same 1000-epoch budget for everyone.
#
# Metrics use --gt-only, so an image without GT is skipped rather than silently scored
# against the model's own prediction (the bug found in results_nnunet_testset_tem2 v1).

set -euo pipefail

MODEL_KEY="${1:?usage: run_eval_paper.sh <model_key> <fold> [gpu]}"
FOLD="${2:?usage: run_eval_paper.sh <model_key> <fold> [gpu]}"
GPU="${3:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "${HERE}")"
REPO_DIR="$(dirname "${TRAINING_DIR}")"
BASE="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}"
RESULTS="${HOME}/resinv_exp/results_paper"
TEM1_DIR="${HOME}/resinv_exp/data/TEM1"
TEM2_TEST_DIR="${HOME}/resinv_exp/data/testset_armand_uaxon"
# shellcheck disable=SC1091
source "${RESINV_VENV:-${HOME}/resinv_exp/venv_resinv_v2}/bin/activate"

# Resolve dataset / trainer / plans from the training registry, so the two never disagree.
eval "$(bash "${HERE}/train_paper.sh" --list | awk -v k="${MODEL_KEY}" '$1==k {print "DS_NAME="$2"; TRAINER="$3"; PLANS="$4}')"
[ -n "${DS_NAME:-}" ] || { echo "Unknown model key ${MODEL_KEY}"; exit 1; }

MODEL_DIR="${BASE}/nnUNet_results/${DS_NAME}/${TRAINER}__${PLANS}__2d"
[ -f "${MODEL_DIR}/fold_${FOLD}/checkpoint_final.pth" ] || { echo "Not trained yet: ${MODEL_DIR}/fold_${FOLD}"; exit 1; }
NAME="${MODEL_KEY}_f${FOLD}"
mkdir -p "${RESULTS}/tem1" "${RESULTS}/tem2test" "${BASE}/logs"

echo "=== ${NAME}: TEM1 test split ==="
CUDA_VISIBLE_DEVICES="${GPU}" python "${TRAINING_DIR}/evaluate_nnunet.py" \
    --model-dir "${MODEL_DIR}" --model-name "${NAME}" \
    --data-dir "${TEM1_DIR}" --split-file "${TEM1_DIR}/subject_split.json" \
    --original-px 0.00236 --fold "${FOLD}" --checkpoint checkpoint_final.pth \
    --output-dir "${RESULTS}/tem1" --gpu-id 0 2>&1 | tee "${BASE}/logs/eval_${NAME}_tem1.log"

echo "=== ${NAME}: TEM2 held-out test set ==="
CUDA_VISIBLE_DEVICES="${GPU}" python "${TRAINING_DIR}/evaluate_nnunet.py" \
    --model-dir "${MODEL_DIR}" --model-name "${NAME}" \
    --data-dir "${TEM2_TEST_DIR}" --original-px 0.00493 --gt-only \
    --fold "${FOLD}" --checkpoint checkpoint_final.pth \
    --output-dir "${RESULTS}/tem2test" --gpu-id 0 2>&1 | tee "${BASE}/logs/eval_${NAME}_tem2test.log"

echo "=== ${NAME}: Dice vs GT ==="
python "${REPO_DIR}/recompute_metrics.py" --results-dir "${RESULTS}/tem1" \
    --data-dir "${TEM1_DIR}" --models "${NAME}" --gt-only
python "${REPO_DIR}/recompute_metrics.py" --results-dir "${RESULTS}/tem2test" \
    --data-dir "${TEM2_TEST_DIR}" --models "${NAME}" --gt-only

echo "Results: ${RESULTS}/tem1/${NAME}/results.csv and ${RESULTS}/tem2test/${NAME}/results.csv"
