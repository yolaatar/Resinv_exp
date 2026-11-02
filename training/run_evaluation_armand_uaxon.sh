#!/bin/bash
# Evaluate Armand's unmyelinated-axon model (Dataset044, fold_all) on the
# Stanford testset_TEM2 test set at 22 pixel sizes.
#
# GPU: CUDA_VISIBLE_DEVICES=0 (shows as cuda:0 inside scripts)
# Log: ~/output_eval_armand_uaxon.log

set -e

DATA_DIR="${HOME}/resinv_exp/data/testset_armand_uaxon"
MODEL_DIR="${HOME}/nnunet_results/Dataset044_unmyelinated_armand_multires/nnUNetTrainer__nnUNetPlans__2d"
OUTPUT_DIR="${HOME}/resinv_exp/results_armand_uaxon"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " Armand uaxon model — testset_TEM2 (Stanford), 21 images"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=1 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${MODEL_DIR}" \
    --model-name armand_uaxon \
    --data-dir "${DATA_DIR}" \
    --original-px 0.00493 \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint checkpoint_final.pth \
    --fold all \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_armand_uaxon.log

echo ""
echo "=== Done ==="
echo "Retrieve results with:"
echo "  scp -r yolaa@tassan.neuro.polymtl.ca:${OUTPUT_DIR} <local path>"
