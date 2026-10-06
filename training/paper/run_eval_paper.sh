#!/bin/bash
# Evaluate one paper-batch model on the TEM1 test set across the paper's pixel-size grid,
# then compute MONAI Dice against GT. Same protocol for every model, so curves are directly
# comparable.
#
# Usage: bash run_eval_paper.sh <model_key> <fold> [gpu]
#        CHECKPOINT=checkpoint_best.pth bash run_eval_paper.sh ...   (supplementary comparison)
#
#   Test set: the 4 TEM1 test subjects in subject_split.json (30 images), native 2.36 nm.
#   TEM2 is not evaluated for now (decision 2026-10-06: too different from TEM1 to be useful).
#
# Grid (25 sizes, nm): log-spaced with ratio ~1.15 over the TEM working range 1-16 nm, with the
# TEM1 and TEM2 native sizes (2.36, 4.93) and the multires training sizes (7, 10, 16) included
# exactly, plus four sparser sizes up to 50 nm, where structures stop being discernible.
#
# Checkpoint: checkpoint_final.pth for every model by default. "best" is picked on each model's
# own validation set, and multires validates on all its resolution copies while control
# validates at native only. In batch 1, control's best epochs were 898-999 but multires' were
# 343-437 in three folds, so "best" would compare a fully trained control against multires
# models with 40% of the training. Final = the same 1000-epoch budget for everyone.
#
# Metrics use --gt-only: an image or class without GT is skipped, never scored against the
# model's own prediction.

set -euo pipefail

MODEL_KEY="${1:?usage: run_eval_paper.sh <model_key> <fold> [gpu]}"
FOLD="${2:?usage: run_eval_paper.sh <model_key> <fold> [gpu]}"
GPU="${3:-0}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"

# Pixel sizes in um/px (the unit evaluate_nnunet.py expects).
PX_GRID=(
    0.001 0.00115 0.00132 0.00152 0.00174 0.002 0.00236 0.00264 0.00303 0.00348
    0.004 0.00459 0.00493 0.00606 0.007 0.008 0.00919 0.01 0.01213 0.01393 0.016
    0.0213 0.0283 0.0376 0.05
)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "${HERE}")"
REPO_DIR="$(dirname "${TRAINING_DIR}")"
BASE="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}"
RESULTS="${HOME}/resinv_exp/results_paper"
TEM1_DIR="${HOME}/resinv_exp/data/TEM1"
# shellcheck disable=SC1091
source "${RESINV_VENV:-${HOME}/resinv_exp/venv_resinv_v2}/bin/activate"

# Resolve dataset / trainer / plans from the training registry, so the two never disagree.
eval "$(bash "${HERE}/train_paper.sh" --list | awk -v k="${MODEL_KEY}" '$1==k {print "DS_NAME="$2"; TRAINER="$3"; PLANS="$4}')"
[ -n "${DS_NAME:-}" ] || { echo "Unknown model key ${MODEL_KEY}"; exit 1; }

MODEL_DIR="${BASE}/nnUNet_results/${DS_NAME}/${TRAINER}__${PLANS}__2d"
[ -f "${MODEL_DIR}/fold_${FOLD}/${CHECKPOINT}" ] || { echo "Missing ${MODEL_DIR}/fold_${FOLD}/${CHECKPOINT}"; exit 1; }
SUFFIX=""
[ "${CHECKPOINT}" = checkpoint_final.pth ] || SUFFIX="_${CHECKPOINT%.pth}"
NAME="${MODEL_KEY}_f${FOLD}${SUFFIX}"
mkdir -p "${RESULTS}/tem1" "${BASE}/logs"

echo "=== ${NAME}: TEM1 test set, ${#PX_GRID[@]} pixel sizes, ${CHECKPOINT} ==="
CUDA_VISIBLE_DEVICES="${GPU}" python "${TRAINING_DIR}/evaluate_nnunet.py" \
    --model-dir "${MODEL_DIR}" --model-name "${NAME}" \
    --data-dir "${TEM1_DIR}" --split-file "${TEM1_DIR}/subject_split.json" \
    --original-px 0.00236 --px-sizes "${PX_GRID[@]}" \
    --fold "${FOLD}" --checkpoint "${CHECKPOINT}" \
    --output-dir "${RESULTS}/tem1" --gpu-id 0 2>&1 | tee "${BASE}/logs/eval_${NAME}_tem1.log"

echo "=== ${NAME}: MONAI Dice vs GT ==="
# TEM1 labels axon and myelin only (its unmyelinated axons are unannotated).
python "${REPO_DIR}/recompute_metrics.py" --results-dir "${RESULTS}/tem1" \
    --data-dir "${TEM1_DIR}" --models "${NAME}" --gt-only --gt-labels axon myelin

echo "Results: ${RESULTS}/tem1/${NAME}/results.csv"
