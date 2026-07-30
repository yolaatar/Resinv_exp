#!/bin/bash
# Train multi-resolution model, v2: TEM1 + downsampled copies, nnUNet 2.8.1 retrain.
#
# Same recipe as train_multires.sh (model 2, Dataset002, nnunetv2==2.2.1), retrained with
# the newer nnUNet version now pinned in ADS (nnunetv2==2.8.1). Written to a new dataset ID
# so the original 2.2.1 checkpoint (Dataset002) stays untouched for comparison.
#
# Same subject split as witness (reads subject_split.json from data dir).
# Training data: original (0.00236 μm/px) + 0.007 + 0.01 + 0.016 μm/px = 4x images
#
# fold_0's train/val split is NOT nnUNet's default random split: after plan_and_preprocess,
# generate_splits_multires.py overwrites splits_final.json with a group-aware split that
# keeps all resolution-duplicates of a source image on the same side (train or val), so
# there's no cross-resolution leakage and val gets a proportional mix of every resolution.
#
# Dependencies: nnunetv2==2.8.1, scikit-learn, Pillow, numpy, scikit-image
# GPU: single GPU, CUDA_VISIBLE_DEVICES=0
#
# Usage: bash train_multires_v2.sh
# Log:   ~/output_multires_v2.log

set -e

DATASET_ID=3
DATASET_NAME="Dataset003_TEM_multires_v2"
DATA_DIR="${HOME}/duke/temp/yolaatar/resinv_exp/data/TEM1"
BASE_DIR="${HOME}/duke/temp/yolaatar/nnunet_resinv_v2"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="/tmp/yolaatar/nnunet_preprocessed_v2"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "======================================================"
echo " Multi-resolution model training (v2, nnUNet 2.8.1)"
echo " Dataset: ${DATA_DIR}"
echo " Extra resolutions: 0.007, 0.01, 0.016 um/px"
echo "======================================================"

# Step 1: Prepare dataset
echo ""
echo "=== Step 1: Preparing nnUNet dataset ==="
python "${SCRIPTS_DIR}/prepare_dataset_multires_v2.py" \
    --data-dir "${DATA_DIR}" \
    --nnunet-raw "${nnUNet_raw}"

# Step 2: Plan and preprocess (2D only)
echo ""
echo "=== Step 2: Planning and preprocessing ==="
nnUNetv2_plan_and_preprocess \
    -d ${DATASET_ID} \
    -c 2d \
    --verify_dataset_integrity

# Step 2.5: Overwrite fold splits with the group-aware, leakage-free version
echo ""
echo "=== Step 2.5: Generating group-aware splits (fold_0) ==="
python "${SCRIPTS_DIR}/generate_splits_multires.py" \
    --nnunet-raw "${nnUNet_raw}" \
    --nnunet-preprocessed "${nnUNet_preprocessed}" \
    --dataset-name "${DATASET_NAME}"

# Step 3: Train (fold 0, 2D)
echo ""
echo "=== Step 3: Training (fold 0, 2D) ==="
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train \
    ${DATASET_ID} \
    2d \
    0 \
    2>&1 | tee ~/output_multires_v2.log

echo ""
echo "=== Done ==="
echo "Checkpoint: ${nnUNet_results}/${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
