#!/bin/bash
# Train one model of the ResInv paper retrain batch (nnUNet 2.8.1).
#
# Usage: bash train_paper.sh <model_key> <fold> [gpu]
#        bash train_paper.sh --list
#
# What makes the batch "fair and square":
#   - one nnUNet version (2.8.1, checked below), one venv
#   - one set of plans for every model: planned once on the multires4 dataset
#     (Dataset102), then transferred to every other dataset, so all models share the
#     same patch size, batch size and network. Without this, nnUNet would plan each
#     dataset from its own image sizes and witness/multires would get different networks.
#   - one subject split (TEM1 subject_split.json, seed 42), reused, never regenerated
#   - one fold assignment (generate_splits_shared.py): fold k validates on the same source
#     images in every dataset
#   - every dataset built by the same script (prepare_dataset_paper.py)
#
# Each run appends a line to ${BASE}/cost_log.jsonl (preprocessing time, training time,
# case count, preprocessed disk size) for the training-cost comparison.
#
# Resumable: a finished fold is skipped, an interrupted one continues with --c.
# Safe to launch several models at once: dataset prep and planning take a per-dataset lock.

set -euo pipefail

BASE="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${RESINV_VENV:-${HOME}/resinv_exp/venv_resinv_v2}"
TEM1_DIR="${HOME}/resinv_exp/data/TEM1"
TEM2_DIR="${HOME}/resinv_exp/data/TEM2"
NNUNET_VERSION_REQUIRED="2.8.1"

export nnUNet_raw="${BASE}/nnUNet_raw"
export nnUNet_preprocessed="${RESINV_PAPER_PREPROCESSED:-/tmp/yolaatar/nnunet_preprocessed_paper}"
export nnUNet_results="${BASE}/nnUNet_results"

# Reference dataset whose plans every model uses.
REF_ID=102
REF_NAME="Dataset102_TEM1_multires4"
PLANS_DEFAULT="nnUNetPlans_ref102"
PLANS_RESENC="nnUNetResEncMPlans_ref102"

# ---------------------------------------------------------------------------
# Model registry: key -> dataset id, dataset name, source, extra pixel sizes (um/px),
# trainer, plans kind, gated (1 = needs ALLOW_GATED=1, design not settled yet).
# ---------------------------------------------------------------------------
lookup_model() {
    GATED=0; PLANS_KIND=default; TRAINER=nnUNetTrainer; SOURCE=TEM1
    case "$1" in
        # Batch 1: core comparison + blur ablation
        witness)        DS_ID=101; DS_NAME=Dataset101_TEM1_witness;   EXTRA="" ;;
        multires)       DS_ID=102; DS_NAME=Dataset102_TEM1_multires4; EXTRA="0.007 0.01 0.016" ;;
        da5)            DS_ID=101; DS_NAME=Dataset101_TEM1_witness;   EXTRA="";  TRAINER=nnUNetTrainerDA5 ;;
        da5_multires)   DS_ID=102; DS_NAME=Dataset102_TEM1_multires4; EXTRA="0.007 0.01 0.016"; TRAINER=nnUNetTrainerDA5 ;;
        # Batch 2: mechanism (E2). Same coverage, different density: multires2/4/8.
        # Same count as multires4, half the coverage: multires_to7.
        scaleaug)       DS_ID=101; DS_NAME=Dataset101_TEM1_witness;   EXTRA="";  TRAINER=nnUNetTrainerScaleAug2p5 ;;
        scaleaug_p05)   DS_ID=101; DS_NAME=Dataset101_TEM1_witness;   EXTRA="";  TRAINER=nnUNetTrainerScaleAug2p5_p05 ;;
        multires2)      DS_ID=103; DS_NAME=Dataset103_TEM1_multires2; EXTRA="0.016" ;;
        multires8)      DS_ID=104; DS_NAME=Dataset104_TEM1_multires8; EXTRA="0.003102 0.004078 0.00536 0.007045 0.00926 0.01217 0.016" ;;
        multires_to7)   DS_ID=105; DS_NAME=Dataset105_TEM1_multires_to7; EXTRA="0.003391 0.004872 0.007" ;;
        # Batch 3: architecture check (ResEnc M), same data as batch 1
        witness_resenc)  DS_ID=101; DS_NAME=Dataset101_TEM1_witness;   EXTRA="";  PLANS_KIND=resenc ;;
        multires_resenc) DS_ID=102; DS_NAME=Dataset102_TEM1_multires4; EXTRA="0.007 0.01 0.016"; PLANS_KIND=resenc ;;
        # Gated: TEM2-only pair. Needs a leakage-free TEM2 split agreed with Armand first
        # (the held-out test set shares 5 subjects with TEM2's annotated images).
        witness_tem2)   DS_ID=111; DS_NAME=Dataset111_TEM2_witness;   EXTRA="";  SOURCE=TEM2; GATED=1 ;;
        multires_tem2)  DS_ID=112; DS_NAME=Dataset112_TEM2_multires4; EXTRA="0.007 0.01 0.016"; SOURCE=TEM2; GATED=1 ;;
        *) return 1 ;;
    esac
    if [ "${SOURCE}" = TEM1 ]; then
        DATA_DIR="${TEM1_DIR}"; ORIGINAL_PX=0.00236; SPLIT_FILE="${TEM1_DIR}/subject_split.json"
    else
        DATA_DIR="${TEM2_DIR}"; ORIGINAL_PX=0.00493; SPLIT_FILE="${TEM2_DIR}/subject_split_paper.json"
    fi
    if [ "${PLANS_KIND}" = resenc ]; then PLANS="${PLANS_RESENC}"; else PLANS="${PLANS_DEFAULT}"; fi
}

ALL_KEYS="witness multires da5 da5_multires scaleaug scaleaug_p05 multires2 multires8 multires_to7 witness_resenc multires_resenc witness_tem2 multires_tem2"

if [ "${1:-}" = "--list" ]; then
    printf "%-16s %-30s %-30s %-20s %s\n" KEY DATASET TRAINER PLANS EXTRA_PX
    for k in ${ALL_KEYS}; do
        lookup_model "$k"
        printf "%-16s %-30s %-30s %-20s %s%s\n" "$k" "${DS_NAME}" "${TRAINER}" "${PLANS}" "${EXTRA:-none}" \
            "$([ "${GATED}" = 1 ] && echo '  [gated]')"
    done
    exit 0
fi

MODEL_KEY="${1:?usage: train_paper.sh <model_key> <fold> [gpu]  (or --list)}"
FOLD="${2:?usage: train_paper.sh <model_key> <fold> [gpu]}"
GPU="${3:-0}"

lookup_model "${MODEL_KEY}" || { echo "Unknown model key '${MODEL_KEY}'. Run with --list."; exit 1; }
if [ "${GATED}" = 1 ] && [ "${ALLOW_GATED:-0}" != 1 ]; then
    echo "${MODEL_KEY} is gated (see comment in the registry). Set ALLOW_GATED=1 once its design is settled."
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
NNUNET_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("nnunetv2"))')"
if [ "${NNUNET_VERSION}" != "${NNUNET_VERSION_REQUIRED}" ]; then
    echo "nnunetv2 ${NNUNET_VERSION} in ${VENV}, expected ${NNUNET_VERSION_REQUIRED}. Refusing to mix versions."
    exit 1
fi

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}" "${BASE}/locks" "${BASE}/logs"
LOG="${BASE}/logs/train_${MODEL_KEY}_fold${FOLD}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

# Run "$@" while holding an exclusive lock named $1 (first arg), so two GPUs never prepare
# or preprocess the same dataset at the same time.
with_lock() {
    local name="$1"; shift
    ( flock -x 9; "$@" ) 9>"${BASE}/locks/${name}.lock"
}

prepare_raw() {  # <ds_id> <ds_name> <data_dir> <original_px> <split_file> <extra...>
    local id="$1" name="$2" data="$3" opx="$4" split="$5"; shift 5
    local extra=()
    [ "$#" -gt 0 ] && extra=(--extra-px "$@")
    python "${SCRIPTS_DIR}/prepare_dataset_paper.py" \
        --data-dir "${data}" --nnunet-raw "${nnUNet_raw}" \
        --dataset-id "${id}" --dataset-name "${name}" \
        --original-px "${opx}" --split-file "${split}" \
        ${extra[@]+"${extra[@]}"} 2>&1 | tee -a "${LOG}"
}

plan_reference() {  # plans the reference dataset once per plans kind
    local marker="${nnUNet_preprocessed}/${REF_NAME}/.done_${PLANS_DEFAULT}"
    if [ ! -f "${marker}" ]; then
        prepare_raw "${REF_ID}" "${REF_NAME}" "${TEM1_DIR}" 0.00236 "${TEM1_DIR}/subject_split.json" 0.007 0.01 0.016
        log "Planning + preprocessing reference ${REF_NAME} -> ${PLANS_DEFAULT}"
        local t0=${SECONDS}
        nnUNetv2_plan_and_preprocess -d "${REF_ID}" -c 2d --verify_dataset_integrity \
            -overwrite_plans_name "${PLANS_DEFAULT}" 2>&1 | tee -a "${LOG}"
        echo $((SECONDS - t0)) > "${marker}"
    fi
    if [ "${PLANS_KIND}" = resenc ]; then
        local rmarker="${nnUNet_preprocessed}/${REF_NAME}/.done_${PLANS_RESENC}"
        if [ ! -f "${rmarker}" ]; then
            log "Planning ResEnc M on reference ${REF_NAME} -> ${PLANS_RESENC}"
            local t0=${SECONDS}
            nnUNetv2_plan_experiment -d "${REF_ID}" -pl nnUNetPlannerResEncM \
                -overwrite_plans_name "${PLANS_RESENC}" 2>&1 | tee -a "${LOG}"
            nnUNetv2_preprocess -d "${REF_ID}" -plans_name "${PLANS_RESENC}" -c 2d 2>&1 | tee -a "${LOG}"
            echo $((SECONDS - t0)) > "${rmarker}"
        fi
    fi
}

prepare_target() {  # raw dataset + transferred plans + preprocessing for the model's dataset
    # Runs inside with_lock's subshell: it communicates only through the marker file,
    # whose content is the preprocessing time in seconds.
    local marker="${nnUNet_preprocessed}/${DS_NAME}/.done_${PLANS}"
    [ -f "${marker}" ] && return
    # shellcheck disable=SC2086
    prepare_raw "${DS_ID}" "${DS_NAME}" "${DATA_DIR}" "${ORIGINAL_PX}" "${SPLIT_FILE}" ${EXTRA}
    local t0=${SECONDS}
    if [ "${DS_ID}" != "${REF_ID}" ]; then
        log "Transferring ${PLANS} from ${REF_NAME} to ${DS_NAME}"
        nnUNetv2_extract_fingerprint -d "${DS_ID}" --verify_dataset_integrity 2>&1 | tee -a "${LOG}"
        nnUNetv2_move_plans_between_datasets -s "${REF_ID}" -t "${DS_ID}" -sp "${PLANS}" -tp "${PLANS}" 2>&1 | tee -a "${LOG}"
        nnUNetv2_preprocess -d "${DS_ID}" -plans_name "${PLANS}" -c 2d 2>&1 | tee -a "${LOG}"
    fi
    echo $((SECONDS - t0)) > "${marker}"
}

make_splits() {
    python "${SCRIPTS_DIR}/generate_splits_shared.py" \
        --nnunet-raw "${nnUNet_raw}" --nnunet-preprocessed "${nnUNet_preprocessed}" \
        --dataset-name "${DS_NAME}" 2>&1 | tee -a "${LOG}"
}

log "=== ${MODEL_KEY} fold ${FOLD} on GPU ${GPU}: ${DS_NAME}, ${TRAINER}, ${PLANS}, nnunetv2 ${NNUNET_VERSION} ==="

# Fail now, not after an hour of preprocessing, if a custom trainer isn't installed.
python - "${TRAINER}" <<'EOF'
import os, sys
import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
root = os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer")
if recursive_find_python_class(root, sys.argv[1], "nnunetv2.training.nnUNetTrainer") is None:
    sys.exit(f"Trainer {sys.argv[1]} not found in nnunetv2. Run install_custom_trainers.sh first.")
EOF

with_lock "${REF_NAME}" plan_reference
with_lock "${DS_NAME}" prepare_target
with_lock "${DS_NAME}" make_splits
PREPROC_SECONDS="$(cat "${nnUNet_preprocessed}/${DS_NAME}/.done_${PLANS}" 2>/dev/null || echo null)"

OUT_DIR="${nnUNet_results}/${DS_NAME}/${TRAINER}__${PLANS}__2d/fold_${FOLD}"
if [ -f "${OUT_DIR}/checkpoint_final.pth" ]; then
    log "Already trained: ${OUT_DIR}/checkpoint_final.pth, skipping"
    exit 0
fi
CONTINUE=""
if [ -f "${OUT_DIR}/checkpoint_latest.pth" ]; then
    log "Found checkpoint_latest.pth, resuming"
    CONTINUE="--c"
fi

START_TS="$(date -Iseconds)"
T0=${SECONDS}
CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_train "${DS_ID}" 2d "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" ${CONTINUE} \
    2>&1 | tee -a "${LOG}"
TRAIN_SECONDS=$((SECONDS - T0))

N_CASES="$(python -c "import json; print(json.load(open('${nnUNet_raw}/${DS_NAME}/dataset.json'))['numTraining'])")"
PREPROC_BYTES="$(du -sb "${nnUNet_preprocessed}/${DS_NAME}/${PLANS}_2d" 2>/dev/null | cut -f1 || echo null)"
printf '{"model": "%s", "fold": "%s", "dataset": "%s", "trainer": "%s", "plans": "%s", "n_cases": %s, "preprocess_seconds": %s, "train_seconds": %s, "resumed": %s, "preprocessed_bytes": %s, "host": "%s", "gpu": "%s", "start": "%s", "end": "%s", "nnunetv2": "%s"}\n' \
    "${MODEL_KEY}" "${FOLD}" "${DS_NAME}" "${TRAINER}" "${PLANS}" "${N_CASES}" "${PREPROC_SECONDS:-null}" \
    "${TRAIN_SECONDS}" "$([ -n "${CONTINUE}" ] && echo true || echo false)" "${PREPROC_BYTES:-null}" \
    "$(hostname)" "${GPU}" "${START_TS}" "$(date -Iseconds)" "${NNUNET_VERSION}" >> "${BASE}/cost_log.jsonl"

log "=== Done: ${OUT_DIR} (${TRAIN_SECONDS}s) ==="
