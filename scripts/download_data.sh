#!/usr/bin/env bash
# Download a UniRank-processed dataset into the directory expected by config/.
# Set HF_ENDPOINT=https://huggingface.co if the default mirror is not desired.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?Usage: download_data.sh taobao|kuairand|merrec|taac25}"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$DATASET" in
  taobao)
    "$PYTHON_BIN" "$ROOT/data_preparation/Taobao/download_taobao.py" \
      --local-dir "$ROOT/data/Taobao/UniRank-Taobao" --endpoint "$ENDPOINT" "${@:2}"
    ;;
  kuairand)
    "$PYTHON_BIN" "$ROOT/data_preparation/KuaiRand/download_unirank_kuairand.py" \
      --local-dir "$ROOT/data/KuaiRand/UniRank-KuaiRand" --endpoint "$ENDPOINT" "${@:2}"
    ;;
  merrec)
    "$PYTHON_BIN" "$ROOT/data_preparation/MerRec/download_unirank_merrec.py" \
      --local-dir "$ROOT/data/MerRec/UniRank-MerRec" --endpoint "$ENDPOINT" "${@:2}"
    ;;
  taac25|taac2025|taac)
    "$PYTHON_BIN" "$ROOT/data_preparation/TAAC2025/download_taac25.py" \
      --local-dir "$ROOT/data/TAAC2025/UniRank-TAAC-25" --endpoint "$ENDPOINT" "${@:2}"
    ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac
