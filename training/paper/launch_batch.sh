#!/bin/bash
# Launch a batch of paper trainings in tmux, one queue per GPU.
#
# Usage: bash launch_batch.sh <batch> [gpu ...]     (default GPUs: 0 1)
#        bash launch_batch.sh 1          -> batch 1 split over GPUs 0 and 1
#        bash launch_batch.sh 1 1        -> all of batch 1 on GPU 1 only
#        bash launch_batch.sh pilot 0    -> witness fold 0 only (quick fair pair with multires_v2)
#
# Batches (about 6 h per run on tassan, from the v2 training logs):
#   pilot : witness:0                                     (1 run)
#   1     : witness + multires, folds 0-2; da5, da5_multires fold 0   (8 runs)
#   2     : scaleaug, multires2, multires8, multires_to7, fold 0       (4 runs)
#   3     : witness_resenc, multires_resenc, fold 0                    (2 runs)
#
# Jobs are dealt round-robin to the GPUs in the listed order, so put the most important
# first. Watch with: tmux ls; tmux attach -t paper_b<batch>_gpu<N>

set -e
BATCH="${1:?usage: launch_batch.sh <pilot|1|2|3> [gpu ...]}"; shift
if [ "$#" -eq 0 ]; then GPUS=(0 1); else GPUS=("$@"); fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${BATCH}" in
    pilot) JOBS=(witness:0) ;;
    1) JOBS=(witness:0 multires:0 witness:1 multires:1 witness:2 multires:2 da5:0 da5_multires:0) ;;
    2) JOBS=(scaleaug:0 multires_to7:0 multires2:0 multires8:0) ;;
    3) JOBS=(witness_resenc:0 multires_resenc:0) ;;
    *) echo "Unknown batch '${BATCH}'"; exit 1 ;;
esac

if [ "${BATCH}" = 2 ]; then
    bash "${HERE}/install_custom_trainers.sh"
fi

declare -A QUEUE
for i in "${!JOBS[@]}"; do
    g="${GPUS[$((i % ${#GPUS[@]}))]}"
    QUEUE[$g]="${QUEUE[$g]:-} ${JOBS[$i]}"
done

for g in "${GPUS[@]}"; do
    [ -z "${QUEUE[$g]:-}" ] && continue
    session="paper_b${BATCH}_gpu${g}"
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "tmux session ${session} already exists, not relaunching it"; continue
    fi
    # shellcheck disable=SC2086
    tmux new-session -d -s "${session}" "bash ${HERE}/run_queue.sh ${g} ${QUEUE[$g]}; exec bash"
    echo "Started ${session}:${QUEUE[$g]}"
done
