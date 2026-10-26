#!/usr/bin/env bash
# Train one or more models on one dataset with a specified seed.
#
# Example:
#   GPU_IDS=0,1 NPROC_PER_NODE=2 bash scripts/run_seed.sh \
#     --dataset merrec --seed 20262027 HSAFStaged HSAFInterleaved
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DATA_DIR="${DATA_DIR:-}"
CONFIG_DIR="${CONFIG_DIR:-}"
SKIP_TEST="${SKIP_TEST:-0}"
KEEP_CHECKPOINT="${KEEP_CHECKPOINT:-1}"
EXPID_TAG="${EXPID_TAG:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
DATASET=""
SEED=""
MODELS=(HSAFStaged HSAFInterleaved)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_seed.sh --dataset DATASET --seed SEED [MODEL ...]

Datasets:
  taobao, kuairand, merrec, taac25

Models may be any model/expid prefix in config/model_config.yaml, such as
HSAFStaged, HSAFInterleaved, RankMixer, EST, or TokenMixerAligned.

Environment:
  GPU_IDS           GPUs passed to run_expid.py (default: 0,1)
  NPROC_PER_NODE    torchrun process count (default: 2)
  DATA_DIR          Override the processed-data directory
  CONFIG_DIR        Preserve/reuse the generated temporary config directory
  SKIP_TEST=1       Do not open the test split
  KEEP_CHECKPOINT   0/1; controls whether checkpoints are retained
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --config-dir) CONFIG_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) MODELS=("$@"); break ;;
  esac
done

case "${DATASET}" in
  taobao) DATASET_ID=Taobao_Action; ROOT_NAME=Taobao ;;
  kuairand|kuai-rand) DATASET_ID=KuaiRand_Video_Action; ROOT_NAME=KuaiRand ;;
  merrec) DATASET_ID=MerRec_Action; ROOT_NAME=MerRec ;;
  taac25|taac2025|taac) DATASET_ID=TencentGR_10M_Action; ROOT_NAME=TAAC2025 ;;
  *) echo "Missing or unsupported --dataset: ${DATASET}" >&2; usage >&2; exit 2 ;;
esac

DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/${ROOT_NAME}/UniRank-${ROOT_NAME}}"
if [[ -z "$SEED" ]]; then
  echo "--seed is required (paper seeds: 3 13 42 2026 20262027)" >&2
  exit 2
fi
if [[ ! "${EXPID_TAG}" =~ ^[A-Za-z0-9_]*$ ]]; then
  echo "EXPID_TAG contains invalid characters" >&2; exit 2
fi
for split in train valid test; do
  [[ -d "${DATA_DIR}/${split}/data" ]] || {
    echo "Missing ${DATA_DIR}/${split}/data; see README.md" >&2; exit 1;
  }
done

CONFIG_DIR="${CONFIG_DIR:-${REPO_DIR}/runs/config_${DATASET_ID}_seed${SEED}_$$}"
CONFIG_DIR="$(mkdir -p "$CONFIG_DIR" && cd "$CONFIG_DIR" && pwd)"
cleanup() {
  rm -f "${CONFIG_DIR}/model_config.yaml" "${CONFIG_DIR}/dataset_config.yaml"
  rmdir "$CONFIG_DIR" 2>/dev/null || true
}
trap cleanup EXIT

export PYTHONHASHSEED="$SEED"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

MODEL_NAMES="${MODELS[*]}" DATASET_ID="$DATASET_ID" DATA_DIR="$DATA_DIR" \
CONFIG_DIR="$CONFIG_DIR" REPO_DIR="$REPO_DIR" SEED="$SEED" EXPID_TAG="$EXPID_TAG" \
SKIP_TEST="$SKIP_TEST" KEEP_CHECKPOINT="$KEEP_CHECKPOINT" \
"$PYTHON_BIN" - <<'PY'
import copy
import os
from pathlib import Path
import yaml

repo = Path(os.environ["REPO_DIR"]).resolve()
data = Path(os.environ["DATA_DIR"]).resolve()
config_dir = Path(os.environ["CONFIG_DIR"])
dataset_id = os.environ["DATASET_ID"]
models = os.environ["MODEL_NAMES"].split()
seed = int(os.environ["SEED"])
expid_tag = os.environ["EXPID_TAG"]
skip_test = os.environ["SKIP_TEST"].lower() in {"1", "true", "yes", "on"}
keep_checkpoint = os.environ["KEEP_CHECKPOINT"].lower() in {"1", "true", "yes", "on"}

model_config = yaml.safe_load((repo / "config/model_config.yaml").read_text())
dataset_config = yaml.safe_load((repo / "config/dataset_config.yaml").read_text())
if dataset_id not in dataset_config:
    raise RuntimeError(f"Missing dataset config: {dataset_id}")

for model in models:
    base_expid = f"{model}_{dataset_id}"
    if base_expid not in model_config:
        raise RuntimeError(f"Missing model config: {base_expid}")
    seeded = copy.deepcopy(model_config[base_expid])
    seeded["seed"] = seed
    seeded["model_id"] = f"{base_expid}_seed{seed}{expid_tag}"
    model_config[seeded["model_id"]] = seeded

entry = copy.deepcopy(dataset_config[dataset_id])
entry.update({
    "data_root": str(config_dir),
    "dataset_dir": str(data),
    "train_data": str(data / "train/data"),
    "valid_data": str(data / "valid/data"),
    "test_data": str(data / "test/data"),
    "train_user_info": str(data / "train/user_info"),
    "valid_user_info": str(data / "valid/user_info"),
    "test_user_info": str(data / "test/user_info"),
    "train_item_info": str(data / "train/item_info"),
    "valid_item_info": str(data / "valid/item_info"),
    "test_item_info": str(data / "test/item_info"),
    "rebuild_dataset": False,
    "blocked": True,
    "keep_checkpoint": keep_checkpoint,
})
if skip_test:
    entry.update({"test_data": None, "test_user_info": None, "test_item_info": None})

(config_dir / "model_config.yaml").write_text(
    yaml.safe_dump(model_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
(config_dir / "dataset_config.yaml").write_text(
    yaml.safe_dump({dataset_id: entry}, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
PY

cd "$REPO_DIR"
for model in "${MODELS[@]}"; do
  expid="${model}_${DATASET_ID}_seed${SEED}${EXPID_TAG}"
  echo ">>> Running ${expid}"
  if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
      run_expid.py --config "$CONFIG_DIR" --expid "$expid" --gpu "$GPU_IDS" ${EXTRA_RUN_ARGS:-}
  else
    "$PYTHON_BIN" run_expid.py --config "$CONFIG_DIR" --expid "$expid" \
      --gpu "$GPU_IDS" ${EXTRA_RUN_ARGS:-}
  fi
done
