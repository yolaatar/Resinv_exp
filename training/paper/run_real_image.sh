#!/bin/bash
# Segment one real image (no ground truth) with several paper models, fold 0 by default.
# Usage: bash run_real_image.sh <image.png> <pixel size um/px> <gpu> [model_key ...]
#   default models: control multires_fine multires scaleaug_wide, plus canonical resampling
#   (control resampled to 2.36 nm and back). FOLD=0 by default.
# Output: ~/resinv_exp/results_paper/real/<image stem>/<model>_f<fold>/{seg.png, seg-axon.png, seg-myelin.png, meta.json}
# Run inside tmux after `set_slot <gpu>` (the image can be hundreds of Mpx).

set -euo pipefail

IMAGE="${1:?usage: run_real_image.sh <image.png> <px um/px> <gpu> [model_key ...]}"
PX="${2:?pixel size in um/px}"
GPU="${3:?gpu}"
shift 3
KEYS=("$@")
[ ${#KEYS[@]} -gt 0 ] || KEYS=(control multires_fine multires scaleaug_wide canonical)
FOLD="${FOLD:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}"
VENV="${RESINV_VENV:-${HOME}/resinv_exp/venv_resinv_v2}"
OUT="${HOME}/resinv_exp/results_paper/real/$(basename "${IMAGE%.*}")"
mkdir -p "${OUT}" "${BASE}/logs"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export nnUNet_results="${BASE}/nnUNet_results" nnUNet_raw="${BASE}/nnUNet_raw"
export nnUNet_preprocessed="${RESINV_PAPER_PREPROCESSED:-/tmp/yolaatar/nnunet_preprocessed_paper}"

for key in "${KEYS[@]}"; do
    model="${key}"; canon=()
    if [ "${key}" = canonical ]; then model=control; canon=(--canonical-px 0.00236); fi
    # Resolve dataset / trainer / plans from the training registry, as run_eval_paper.sh does.
    eval "$(bash "${HERE}/train_paper.sh" --list | awk -v k="${model}" '$1==k {print "DS_NAME="$2"; TRAINER="$3"; PLANS="$4}')"
    MODEL_DIR="${BASE}/nnUNet_results/${DS_NAME}/${TRAINER}__${PLANS}__2d"
    NAME="${key}_f${FOLD}"
    echo "=== ${NAME}: ${MODEL_DIR} ==="
    CUDA_VISIBLE_DEVICES="${GPU}" python "${HERE}/predict_real_image.py" --image "${IMAGE}" --px "${PX}" \
        --model-dir "${MODEL_DIR}" --name "${NAME}" --out-dir "${OUT}" --fold "${FOLD}" ${canon[@]+"${canon[@]}"} \
        2>&1 | tee -a "${BASE}/logs/real_$(basename "${IMAGE%.*}").log"
done
echo "Outputs: ${OUT}"
