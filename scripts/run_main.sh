#!/usr/bin/env bash
# Example queue for one dataset and the five paper seeds.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?Usage: run_main.sh DATASET [MODEL ...]}"
shift
SEEDS=(3 13 42 2026 20262027)
for seed in "${SEEDS[@]}"; do
  echo ">>> Main experiment dataset=${DATASET} seed=${seed}"
  bash "${SCRIPT_DIR}/run_seed.sh" --dataset "$DATASET" --seed "$seed" "$@"
done
