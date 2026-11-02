#!/usr/bin/env bash
# Filter + count axons for TEM3 (armand_uaxon and armand_control), using
# axondeepseg_filter and axondeepseg_count from axondeepseg's ac/axon-counter
# branch (PR #1005). Requires that branch checked out wherever this runs:
#   cd ~/axondeepseg && git checkout ac/axon-counter && git pull origin ac/axon-counter
#
# Runs on the per-pixel-size .xlsx morphometrics files that run_morphometrics.py
# already generated -- default (non -m/-mask_mode) mode for both tools, so
# --allow-large-images is not needed here (no image I/O in this mode).
#
# Resumable: skips any image folder that already has axon_counts.csv.
#
# Usage:
#   bash run_counting_tem3.sh [RESULTS_ROOT]
# Default RESULTS_ROOT: ~/resinv_exp/results_tem3

set -euo pipefail

RESULTS_ROOT="${1:-$HOME/resinv_exp/results_tem3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MODEL in armand_uaxon armand_control; do
    RESULTS_DIR="${RESULTS_ROOT}/${MODEL}"
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "SKIP: ${RESULTS_DIR} does not exist"
        continue
    fi

    echo "======================================================"
    echo " ${MODEL}: filtering + counting"
    echo "======================================================"

    for d in "${RESULTS_DIR}"/*/predictions; do
        [ -d "$d" ] || continue
        out="${d}/axon_counts.csv"
        if [ -f "$out" ]; then
            echo "skip $(dirname "$d") (already done)"
            continue
        fi
        echo "=== $(basename "$(dirname "$d")") ==="
        axondeepseg_filter -i "$d"
        axondeepseg_count -i "$d" -o "$out"
    done

    echo "Aggregating ${MODEL}..."
    python "${SCRIPT_DIR}/aggregate_axon_counts_tem3.py" \
        --results-dir "${RESULTS_DIR}" \
        --out "${RESULTS_DIR}/axon_counts_all.csv"
done

echo "Done."
