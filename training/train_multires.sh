#!/bin/bash
# Train multi-resolution model: standard nnUNet, TEM1 + downsampled copies (model 2/3)
#
# Same subject split as witness (reads subject_split.json from data dir).
# Training data: original (0.00236 μm/px) + 0.007 + 0.01 + 0.016 μm/px = 4x images
#
# Dependencies: nnunetv2==2.2.1, Pillow, numpy, scikit-image
# GPU: single GPU, CUDA_VISIBLE_DEVICES=1
#
# Usage: bash train_multires.sh
# Log:   ~/output_multires.log

set -e

DATASET_ID=2
DATA_DIR="${HOME}/duke/temp/yolaatar/resinv_exp/data/TEM1"
BASE_DIR="${HOME}/duke/temp/yolaatar/nnunet_resinv"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="/tmp/yolaatar/nnunet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "======================================================"
echo " Multi-resolution model training"
echo " Dataset: ${DATA_DIR}"
echo " Extra resolutions: 0.007, 0.01, 0.016 um/px"
echo "======================================================"

# Step 1: Prepare dataset
echo ""
echo "=== Step 1: Preparing nnUNet dataset ==="
python "${SCRIPTS_DIR}/prepare_dataset_multires.py" \
    --data-dir "${DATA_DIR}" \
    --nnunet-raw "${nnUNet_raw}"

# Step 2: Plan and preprocess (2D only)
echo ""
echo "=== Step 2: Planning and preprocessing ==="
nnUNetv2_plan_and_preprocess \
    -d ${DATASET_ID} \
    -c 2d \
    --verify_dataset_integrity

# Step 3: Train (fold 0, 2D)
echo ""
echo "=== Step 3: Training (fold 0, 2D) ==="
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train \
    ${DATASET_ID} \
    2d \
    0 \
    2>&1 | tee ~/output_multires.log

echo ""
echo "=== Done ==="
echo "Checkpoint: ${nnUNet_results}/Dataset002_TEM_multires/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
