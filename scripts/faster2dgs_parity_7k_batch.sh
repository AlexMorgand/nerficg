#!/usr/bin/env bash
# Run 7k parity one scene at a time (single GPU job, flock lock).
#
# Each scene: Faster2DGS train+eval, then official train+eval — never in parallel.
# Avoid starting a second terminal batch while this is running.
#
# Usage:
#   ./faster2dgs_parity_7k_batch.sh room counter
#   ./faster2dgs_parity_7k_batch.sh --all-remaining
#   PRELOADING_LEVEL=0 ./faster2dgs_parity_7k_batch.sh garden   # lighter RAM use
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/opt/anaconda3/envs/nerficg/bin/python}"
LOCK="${LOCK:-/tmp/faster2dgs_parity_7k.lock}"
LOG="${LOG:-/tmp/faster2dgs_parity_7k_batch.log}"
REPO="$(cd .. && pwd)"
DATASET_ROOT="$REPO/dataset/mipnerf360"

ALL_SCENES=(kitchen bonsai room counter garden bicycle stump)

dataset_ready() {
  local scene=$1
  [[ -d "$DATASET_ROOT/$scene/sparse/0" ]]
}

remaining_scenes() {
  local s out ours
  for s in "${ALL_SCENES[@]}"; do
    out="$REPO/output/official_2dgs/${s}_7k/point_cloud/iteration_7000/point_cloud.ply"
    results="$REPO/output/official_2dgs/${s}_7k/results.json"
    ours=$(find "$REPO/output/Faster2DGS" -maxdepth 1 -type d -name "${s}_faster2dgs_parity7k*" 2>/dev/null | sort | tail -1)
    if [[ ! -f "$out" ]] || [[ ! -f "$results" ]] \
        || [[ -z "$ours" ]] || [[ ! -f "$ours/test_7000/metrics_8bit.txt" ]]; then
      echo "$s"
    fi
  done
}

run_scene() {
  local scene=$1
  if ! dataset_ready "$scene"; then
    echo "SKIP $scene: missing $DATASET_ROOT/$scene/sparse/0 (mount dataset?)" | tee -a "$LOG"
    return 0
  fi
  echo "========== parity @7k: $scene $(date -Is) ==========" | tee -a "$LOG"
  if [[ -n "${PRELOADING_LEVEL:-}" ]]; then
    export PRELOADING_LEVEL
  fi
  local -a skip=()
  ours=$(find "$REPO/output/Faster2DGS" -maxdepth 1 -type d -name "${scene}_faster2dgs_parity7k*" 2>/dev/null | sort | tail -1)
  if [[ -n "$ours" && -f "$ours/test_7000/metrics_8bit.txt" ]]; then
    skip+=(--skip-ours)
    echo "  (ours already done: $ours)" | tee -a "$LOG"
  fi
  "$PYTHON" faster2dgs_parity_7k.py --scene "$scene" "${skip[@]}" 2>&1 | tee -a "$LOG"
}

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another parity batch holds $LOCK — not starting." >&2
  exit 1
fi

scenes=()
if [[ "${1:-}" == "--all-remaining" ]]; then
  mapfile -t scenes < <(remaining_scenes)
elif [[ $# -eq 0 ]]; then
  echo "usage: $0 [--all-remaining | scene ...]" >&2
  exit 1
else
  scenes=("$@")
fi

echo "parity batch: ${scenes[*]} (log: $LOG)" | tee -a "$LOG"
for scene in "${scenes[@]}"; do
  run_scene "$scene"
done
echo "========== batch done $(date -Is) ==========" | tee -a "$LOG"
