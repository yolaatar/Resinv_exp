#!/bin/bash
# Wait for the hybrid queue (run_hybrid.sh) to finish, then run the pixel-size estimation (CPU only).
# Both need RAM beyond the ~2.5 GB user cap, so launch this from tmux after set_slot <gpu>.
#
# Usage:  bash run_after_hybrid.sh [workers]      (default 8)
# Log:    ~/resinv_exp/nnunet_paper/logs/px_estimation.log

WORKERS="${1:-8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${RESINV_PAPER_BASE:-${HOME}/resinv_exp/nnunet_paper}/logs"
HYBRID_LOG="${LOGS}/hybrid.log"
LOG="${LOGS}/px_estimation.log"

echo "[$(date '+%F %T')] waiting for the hybrid queue (${HYBRID_LOG})" | tee -a "${LOG}"
until grep -q "Done. FAILED" "${HYBRID_LOG}" 2>/dev/null; do
    sleep 300
done
echo "[$(date '+%F %T')] hybrid done: $(tail -1 "${HYBRID_LOG}")" | tee -a "${LOG}"

# shellcheck disable=SC1091
source "${RESINV_VENV:-${HOME}/resinv_exp/venv_resinv_v2}/bin/activate"
python "${HERE}/estimate_pixel_size.py" --workers "${WORKERS}" 2>&1 | tee -a "${LOG}"
echo "[$(date '+%F %T')] pixel-size estimation finished (exit ${PIPESTATUS[0]})" | tee -a "${LOG}"
