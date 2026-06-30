#!/usr/bin/env bash
# Convenience wrapper: M360 photometric parity @7k for all scenes (sequential, flock lock).
# Prefer: PRELOADING_LEVEL=0 ./faster2dgs_parity_7k_batch.sh --all-remaining
set -euo pipefail
cd "$(dirname "$0")"
export PRELOADING_LEVEL="${PRELOADING_LEVEL:-0}"
exec ./faster2dgs_parity_7k_batch.sh kitchen bonsai room counter garden bicycle stump
