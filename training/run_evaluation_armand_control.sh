#!/bin/bash
# Evaluate Armand's released "control" unmyelinated-axon model (v3.0.0,
# Dataset043_TEM_UNMYELINATED_AGG, fold_all) on the Stanford testset_TEM2 test
# set at 22 pixel sizes. Same testset/pipeline as run_evaluation_armand_uaxon.sh,
# used as a baseline to compare against the multires model.
#
# GPU: CUDA_VISIBLE_DEVICES=0 (shows as cuda:0 inside scripts)
# Log: ~/output_eval_armand_control.log

set -e

DATA_DIR="${HOME}/resinv_exp/data/testset_armand_uaxon"
MODEL_DIR="${HOME}/resinv_exp/models/control_v3/model_seg_unmyelinated_srf_app"
OUTPUT_DIR="${HOME}/resinv_exp/results_armand_uaxon"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " Armand control model (v3.0.0) — testset_TEM2 (Stanford), 21 images"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=1 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${MODEL_DIR}" \
    --model-name armand_control \
    --data-dir "${DATA_DIR}" \
    --original-px 0.00493 \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint checkpoint_final.pth \
    --fold all \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_armand_control.log

echo ""
echo "=== Done ==="
echo "Retrieve results with:"
echo "  scp -r yolaa@tassan.neuro.polymtl.ca:${OUTPUT_DIR}/armand_control <local path>"
