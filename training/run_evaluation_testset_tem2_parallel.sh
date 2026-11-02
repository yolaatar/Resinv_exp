#!/bin/bash
# Launch the held-out TEM2 test set eval on GPU 1 (only GPU available right now).
#
#   tmux session eval_testset_tem2_gpu1 -> all 7 models, sequential
#
# Check on it with:
#   tmux attach -t eval_testset_tem2_gpu1
#
# Usage: bash run_evaluation_testset_tem2_parallel.sh

set -e

SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

tmux new -d -s eval_testset_tem2_gpu1 \
    "bash ${SCRIPTS_DIR}/run_evaluation_testset_tem2.sh 1 witness multires da5 da5_multires multires_full multires_v2 multires_full_v2"

echo "Launched:"
tmux ls
