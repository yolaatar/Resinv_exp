#!/bin/bash
# Evaluate Armand's released "control" model (v3.0.0, Dataset043_TEM_UNMYELINATED_AGG,
# fold_all) on TEM3 (data_axondeepseg_stanford_app) at 22 pixel sizes. Same
# dataset/pipeline as run_evaluation_armand_uaxon_tem3.sh, used as a baseline
# to compare against the multires model.
#
# TEM3 mixes native pixel sizes across images (0.00493 and 0.005648 um/px
# seen so far) — per-image PixelSize sidecar JSON is read automatically by
# evaluate_nnunet.py, --original-px below is only a fallback for images
# missing a sidecar.
#
# No Dice/GT-vs-prediction step here — this dataset has manual axon/uaxon
# counts instead of segmentation GT, compared separately after counting.
#
# GPU: CUDA_VISIBLE_DEVICES=0 (shows as cuda:0 inside scripts)
# Log: ~/output_eval_armand_control_tem3.log

set -e

DATA_DIR="${HOME}/resinv_exp/data/data_axondeepseg_stanford_app"
MODEL_DIR="${HOME}/resinv_exp/models/control_v3/model_seg_unmyelinated_srf_app"
OUTPUT_DIR="${HOME}/resinv_exp/results_tem3"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " Armand control model (v3.0.0) — TEM3 (data_axondeepseg_stanford_app)"
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
    2>&1 | tee ~/output_eval_armand_control_tem3.log

echo ""
echo "=== Done ==="
echo "Retrieve results with:"
echo "  scp -r yolaa@tassan.neuro.polymtl.ca:${OUTPUT_DIR}/armand_control <local path>"
