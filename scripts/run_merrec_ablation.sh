#!/usr/bin/env bash
# Reproduce the reported single-seed MerRec component/input-feature ablation.
# The paper seed is 20262027. By default this queue uses two GPUs.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${SEED:-20262027}"
GPU_IDS="${GPU_IDS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
KEEP_CHECKPOINT="${KEEP_CHECKPOINT:-0}"

MODELS=(
  RankMixer
  EST
  HSAFStagedStateInput
  HSAFInterleavedStateInput
  HSAFStaged
  HSAFInterleaved
  HSAFStaged_wo_GHS_I
  HSAFStaged_wo_GHS_MT
  HSAFStaged_wo_LHS_SM
  HSAFInterleaved_wo_GHS_I
  HSAFInterleaved_wo_GHS_MT
  HSAFInterleaved_wo_LHS_SM
)

cd "$REPO_DIR"
for model in "${MODELS[@]}"; do
  echo ">>> MerRec ablation: ${model} (seed ${SEED})"
  GPU_IDS="$GPU_IDS" NPROC_PER_NODE="$NPROC_PER_NODE" \
  KEEP_CHECKPOINT="$KEEP_CHECKPOINT" EXPID_TAG="_ablation" \
    bash scripts/run_seed.sh --dataset merrec --seed "$SEED" "$model"
done
