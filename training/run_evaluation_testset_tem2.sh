#!/bin/bash
# Evaluate multires-family models on the held-out TEM2 test set.
#
# Data lives at ~/resinv_exp/data/testset_armand_uaxon on tassan — same physical BIDS
# dataset used for Armand's uaxon model eval (10 subjects), reused here for axon/myelin
# Dice since it was never seen during training by any model here:
#   - sub-366A, sub-367A, sub-368A, sub-369B, sub-371: subjects never used in training
#   - sub-370, sub-372, sub-373C, sub-374, sub-375: same subjects as training, but different
#     sample images (training used s6/7, s5, s1c1/c2, s4, s1/5/6/7 — this set uses s2/s3/s4
#     range instead, no exact image overlap)
#
# This gives a genuinely held-out comparison for multires_full_v2 (trained fold_all, so its
# TEM1/TEM2-training-data numbers aren't held-out — see run_commands.md).
#
# Usage: bash run_evaluation_testset_tem2.sh <gpu-id> <model-name> [<model-name> ...]
#   bash run_evaluation_testset_tem2.sh 0 witness multires da5 da5_multires
#   bash run_evaluation_testset_tem2.sh 1 multires_full multires_v2 multires_full_v2
#
# Valid model names: witness multires da5 da5_multires multires_full multires_v2 multires_full_v2
# Logs: ~/output_eval_testset_tem2_{model}.log

set -e

GPU_ID="$1"
shift
MODELS=("$@")

if [ -z "${GPU_ID}" ] || [ "${#MODELS[@]}" -eq 0 ]; then
    echo "Usage: bash run_evaluation_testset_tem2.sh <gpu-id> <model-name> [<model-name> ...]"
    exit 1
fi

DATA_DIR="${HOME}/resinv_exp/data/testset_armand_uaxon"
OUTPUT_DIR="${HOME}/resinv_exp/results_nnunet_testset_tem2"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

NNUNET_RESULTS_V1="${HOME}/nnunet_results"
NNUNET_DA5_MODELS="${HOME}/nnunet_da5_models"
NNUNET_RESULTS_V2="${HOME}/resinv_exp/nnunet_resinv_v2/nnUNet_results"

# Confirm this path before relying on multires_full results — the Dataset005 (multires_full,
# 2.2.1, fold_all) checkpoint was trained via train_multires_full.sh on duke, and this repo
# has no confirmed record of where it ended up on tassan (only a Compute Canada Vulcan path
# exists, in slurm_eval_multires_full.sh). Adjust if it's elsewhere.
MULTIRES_FULL_V1_DIR="${NNUNET_RESULTS_V1}/Dataset005_TEM12_multires/nnUNetTrainer__nnUNetPlans__2d"

PY_V1="${HOME}/resinv_exp/venv_resinv/bin/python"
PY_V2="${HOME}/resinv_exp/venv_resinv_v2/bin/python"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " nnUNet evaluation — held-out TEM2 test set (GPU ${GPU_ID})"
echo " Data: ${DATA_DIR}"
echo " Models: ${MODELS[*]}"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

run_eval() {
    local py="$1" model_dir="$2" model_name="$3" checkpoint="$4" fold="$5"
    if [ ! -d "${model_dir}" ]; then
        echo ""
        echo "=== SKIP ${model_name}: model dir not found (${model_dir}) ==="
        return
    fi
    echo ""
    echo "=== Model: ${model_name} (fold_${fold}) ==="
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${py}" "${SCRIPTS_DIR}/evaluate_nnunet.py" \
        --model-dir "${model_dir}" \
        --model-name "${model_name}" \
        --data-dir "${DATA_DIR}" \
        --original-px 0.00493 \
        --gt-only \
        --fold "${fold}" \
        --checkpoint "${checkpoint}" \
        --output-dir "${OUTPUT_DIR}" \
        --gpu-id 0 \
        2>&1 | tee "${HOME}/output_eval_testset_tem2_${model_name}.log"
}

for model in "${MODELS[@]}"; do
    case "${model}" in
        witness)
            run_eval "${PY_V1}" "${NNUNET_RESULTS_V1}/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d" \
                witness checkpoint_best.pth 0 ;;
        multires)
            run_eval "${PY_V1}" "${NNUNET_RESULTS_V1}/Dataset002_TEM_multires/nnUNetTrainer__nnUNetPlans__2d" \
                multires checkpoint_best.pth 0 ;;
        da5)
            run_eval "${PY_V1}" "${NNUNET_DA5_MODELS}/Dataset001_TEM_witness/nnUNetTrainerDA5__nnUNetPlans__2d" \
                da5 checkpoint_best.pth 0 ;;
        da5_multires)
            run_eval "${PY_V1}" "${NNUNET_DA5_MODELS}/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d" \
                da5_multires checkpoint_best.pth 0 ;;
        multires_full)
            run_eval "${PY_V1}" "${MULTIRES_FULL_V1_DIR}" \
                multires_full checkpoint_final.pth all ;;
        multires_v2)
            run_eval "${PY_V2}" "${NNUNET_RESULTS_V2}/Dataset003_TEM_multires_v2/nnUNetTrainer__nnUNetPlans__2d" \
                multires_v2 checkpoint_best.pth 0 ;;
        multires_full_v2)
            run_eval "${PY_V2}" "${NNUNET_RESULTS_V2}/Dataset006_TEM12_multires_v2/nnUNetTrainer__nnUNetPlans__2d" \
                multires_full_v2 checkpoint_final.pth all ;;
        *)
            echo "Unknown model: ${model}" ;;
    esac
done

echo ""
echo "=== Done (GPU ${GPU_ID}) ==="
