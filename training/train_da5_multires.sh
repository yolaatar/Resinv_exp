#!/bin/bash
# Train DA5 multi-resolution model: nnUNetTrainerDA5, TEM1 multi-res (model 4/4)
#
# Same dataset as multi-res (Dataset002_TEM_multires, already prepared).
# Only difference: DA5 trainer with more aggressive augmentation.
#
# GPU: CUDA_VISIBLE_DEVICES=1
# Log: ~/output_da5_multires.log

set -e

DATASET_ID=2
BASE_DIR="${HOME}/duke/temp/yolaatar/nnunet_resinv"

export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="/tmp/yolaatar/nnunet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

mkdir -p "${nnUNet_preprocessed}"

echo "======================================================"
echo " DA5 multi-resolution model training (model 4)"
echo " Dataset: Dataset002_TEM_multires (already preprocessed)"
echo " Trainer: nnUNetTrainerDA5"
echo "======================================================"

# Preprocessed data may still be in /tmp from multires training.
# If not, rerun preprocessing (no dataset prep needed, same raw data).
if [ ! -f "${nnUNet_preprocessed}/Dataset002_TEM_multires/nnUNetPlans.json" ]; then
    echo "=== Preprocessing (plans not found in /tmp) ==="
    nnUNetv2_plan_and_preprocess -d ${DATASET_ID} -c 2d
fi

echo ""
echo "=== Training (fold 0, 2D, DA5 trainer) ==="
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train \
    ${DATASET_ID} \
    2d \
    0 \
    -tr nnUNetTrainerDA5 \
    2>&1 | tee ~/output_da5_multires.log

echo ""
echo "=== Done ==="
echo "Checkpoint: ${nnUNet_results}/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
