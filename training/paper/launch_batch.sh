#!/bin/bash
# Launch a batch of paper trainings in tmux, one queue per GPU.
#
# Usage: bash launch_batch.sh <batch> [gpu ...]     (default GPU: 0, tassan has one)
#        bash launch_batch.sh 1          -> batch 1 queued on GPU 0
#        bash launch_batch.sh 1 0 1      -> batch 1 split over GPUs 0 and 1, if two are free
#        bash launch_batch.sh pilot 0    -> control fold 0 only (quick fair pair with multires_v2)
#
# Batches (about 6 h per run on tassan, from the v2 training logs):
#   pilot : control:0                                     (1 run)
#   1     : control + multires, folds 0-4 (10 runs, ~57 h on one GPU). DA5 is not retrained:
#           the paper reuses the earlier nnUNet 2.2.1 DA5 results as a self-contained ablation
#   2     : scaleaug, multires2, multires8, multires_to7, fold 0       (4 runs)
#   3     : control_resenc, multires_resenc, fold 0                    (2 runs)
#
# Jobs are dealt round-robin to the GPUs in the listed order, so put the most important
# first. Watch with: tmux ls; tmux attach -t paper_b<batch>_gpu<N>

set -e
BATCH="${1:?usage: launch_batch.sh <pilot|1|2|3> [gpu ...]}"; shift
if [ "$#" -eq 0 ]; then GPUS=(0); else GPUS=("$@"); fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${BATCH}" in
    pilot) JOBS=(control:0) ;;
    1) JOBS=(control:0 multires:0 control:1 multires:1 control:2 multires:2 control:3 multires:3 control:4 multires:4) ;;
    2) JOBS=(scaleaug:0 multires_to7:0 multires2:0 multires8:0) ;;
    3) JOBS=(control_resenc:0 multires_resenc:0) ;;
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
