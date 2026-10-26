# History-State-Aware Framework for Unified CTR Models

This repository contains the PyTorch implementation and reproduction pipeline for **History-State-Aware Framework for Unified CTR Models (HSAF)**. HSAF adds global and local history-state awareness to staged and interleaved unified CTR architectures. The release includes the two HSAF instances, eight unified baselines, dataset configurations, training/evaluation code, and the MerRec ablation used in the paper.

The code is built on the [UniRank](https://github.com/salmon1802/UniRank) benchmark. See [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) for the attribution and citation.

## Repository layout

```text
config/                  Dataset and model YAML configurations
model_zoo/               HSAF, state-input controls, and unified baselines
unirank/                 UniRank data loading, training, and evaluation framework
data_preparation/        Download and raw-data preprocessing scripts
scripts/                 Download, seeded training, main-run, and ablation launchers
docs/dataset.md          Detailed dataset provenance and preprocessing protocol
run_expid.py             Single-experiment training/evaluation entry point
```

No dataset, checkpoint, or experiment log is distributed in this directory.

## Installation

The code was tested with Python 3.10, PyTorch 2.5.1, and CUDA 12.4.

```bash
cd HSAF
python -m pip install -r requirements.txt
```

The default launcher uses `python` and `torchrun` from the activated environment. Both can be overridden with `PYTHON_BIN` and `TORCHRUN_BIN`.

## Data preparation

The experiments use four UniRank-processed datasets:

| Paper name | Dataset ID | Expected processed path |
|---|---|---|
| Taobao | `Taobao_Action` | `data/Taobao/UniRank-Taobao` |
| KuaiRand | `KuaiRand_Video_Action` | `data/KuaiRand/UniRank-KuaiRand` |
| MerRec | `MerRec_Action` | `data/MerRec/UniRank-MerRec` |
| TAAC-25 | `TencentGR_10M_Action` | `data/TAAC2025/UniRank-TAAC-25` |

### Option A: download processed UniRank data (recommended)

The processed repositories are available on Hugging Face:

- `salmon1802/Taobao`
- `salmon1802/KuaiRand`
- `salmon1802/MerRec`
- `salmon1802/TAAC-25`

A convenience wrapper downloads one dataset into the expected path:

```bash
# The wrapper defaults to hf-mirror.com. Use official Hugging Face as follows.
export HF_ENDPOINT=https://huggingface.co
bash scripts/download_data.sh merrec
```

Replace `merrec` with `taobao`, `kuairand`, or `taac25`. Each resulting directory should contain:

```text
train/{data,user_info,item_info}
valid/{data,user_info,item_info}
test/{data,user_info,item_info}
feature_map.json
meta_data.json
block_manifest.json
```

### Option B: rebuild from raw data

Raw-data scripts are in `data_preparation/<dataset>/`. A typical rebuild has the form:

```bash
python data_preparation/Taobao/preprocess_Taobao_seq_action.py \
  --data_dir /path/to/raw/Taobao \
  --output_dir data/Taobao/UniRank-Taobao

python data_preparation/MerRec/preprocess_MerRec_seq_action.py \
  --data_dir /path/to/raw/MerRec \
  --output_dir data/MerRec/UniRank-MerRec
```

Run each script with `--help` to inspect dataset-specific options. KuaiRand and TAAC-25 preprocessors are `data_preparation/KuaiRand/preprocess_Kuairand_seq_action.py` and `data_preparation/TAAC2025/preprocess_TAAC2025_seq_action.py`, respectively. Source URLs, schema details, chronological split rules, and resource notes are documented in [`docs/dataset.md`](docs/dataset.md).

The processed protocol is point-wise autoregressive: each target request can only use preceding behaviors. Histories are truncated to at most 100 events in the main experiments. The loaders read blocked Parquet files and construct the causal prefix from full user sequence information.

## Main experiments

### One seed and selected models

Train the two HSAF instances on MerRec using two GPUs:

```bash
GPU_IDS=0,1 NPROC_PER_NODE=2 bash scripts/run_seed.sh \
  --dataset merrec --seed 20262027 HSAFStaged HSAFInterleaved
```

The dataset aliases are `taobao`, `kuairand`, `merrec`, and `taac25`. Models can be selected positionally. The paper's architecture groups and implementations are:

| Architecture | Paper baseline | Implementation / model name |
|---|---|---|
| Staged | RankMixer | `RankMixer` |
| Staged | HeMix | `HeMix` |
| Staged | UniMixer | `UniMixer` |
| Staged | TokenMixer | `TokenMixerAligned` |
| Interleaved | OneTrans | `OneTrans` |
| Interleaved | HyFormer | `HyFormer` |
| Interleaved | EST | `EST` |
| Interleaved | UltraHSTU | `UltraHSTU` |
| Staged HSAF | HSAF_stg | `HSAFStaged` |
| Interleaved HSAF | HSAF_int | `HSAFInterleaved` |

For example:

```bash
GPU_IDS=0,1 NPROC_PER_NODE=2 bash scripts/run_seed.sh \
  --dataset taobao --seed 3 \
  RankMixer HeMix UniMixer TokenMixerAligned \
  OneTrans HyFormer EST UltraHSTU \
  HSAFStaged HSAFInterleaved
```

The five seeds used in the main table are `3`, `13`, `42`, `2026`, and `20262027`. To run all five seeds sequentially:

```bash
bash scripts/run_main.sh merrec \
  RankMixer HeMix UniMixer TokenMixerAligned \
  OneTrans HyFormer EST UltraHSTU \
  HSAFStaged HSAFInterleaved
```

Logs and retained checkpoints are written under `checkpoints/<dataset_id>/`, with one `<expid>_seed<seed>.log` per run. Set `KEEP_CHECKPOINT=0` to keep logs only.

### MerRec component and input-feature ablation

Run the single-seed MerRec ablation queue:

```bash
GPU_IDS=0,1 NPROC_PER_NODE=2 bash scripts/run_merrec_ablation.sh
```

The script runs the staged and interleaved backbones, the full HSAF instances, the three reported component removals, and the history-state-as-input-feature control.

**Reproduction note.** The interleaved component-removal numbers in the current paper table were produced with the historical `EST_EviTime` entry point, before the EST implementation was consolidated into `EST_EviTime_gatecalib_use_time`. To reproduce those recorded numbers exactly, the packaged `HSAFInterleaved_wo_*` configurations use `HSAFInterleavedReported`, an alias of `EST_EviTime`. The main `HSAFInterleaved` model and all main-table runs use the current per-layer-gate mainline implementation. The staged component ablations already use the current mainline class. If a purely method-consistent interleaved ablation is desired, change `model: HSAFInterleavedReported` to `model: HSAFInterleaved` in the corresponding YAML entries before training.

The reported input-feature control is the **numeric non-sequence state control**:

| Public control | Exact implementation retained from the reported run |
|---|---|
| `HSAFStagedStateInput` | `RankMixer_StateFeature` |
| `HSAFInterleavedStateInput` | `EST_StateFeature` |

It registers global and summarized local state scalars as ordinary numeric fields and feeds them through the backbones' original feature-embedding path. It does not use bucket embeddings, gates, calibration heads, score biases, or the later exploratory state-sequence-token variant.

## HSAF implementation map

`model_zoo/HSAF.py` exposes the public names. The exact mainline numerical core remains in `model_zoo/EviTime_gatecalib_use_time.py`; retaining the original internal identifier makes the mapping to reported runs explicit and avoids unnecessary state-dict changes. The historical interleaved ablation entry point is retained in `model_zoo/EviTime.py` and aliased as `HSAFInterleavedReported`.

| Paper component | Main implementation points | Configuration switch |
|---|---|---|
| Global state encoding | `_evidence_features`; length, recency, and span bucket embeddings | `evidence_len_edges`, `num_evidence_time_buckets` |
| Global-state injection | `_inject_evidence` / `_inject_evidence_rows` | `use_evidence_token` |
| GHS-I: guided interaction | layer-wise `evidence_gate_layers` and history-read modulation | `use_history_gate` |
| GHS-MT: calibrated multi-task prediction | `evidence_calibration` and EMA-centered task logits | `use_logit_calibration`, `calib_task_specific`, `center_mode` |
| LHS-SM: local sequence modeling | `temporal_bias`, `position_bias`, `_score_biases` | `use_temporal_bias`, `use_position_bias` |
| Gate-spread regularization | batch standard deviation hinge over layer gates | `gate_spread_weight`, `gate_spread_target` |

The reported component removals map directly to:

```text
w/o GHS-I   -> use_history_gate: false
w/o GHS-MT  -> calib_task_specific: false
w/o LHS-SM  -> use_temporal_bias: false and use_position_bias: false
```

A diagnostic `*_wo_global_state_injection` configuration is also included, but it is not one of the three component-removal rows in the reported ablation table.

## Hyperparameters and protocol

All model-specific settings are in `config/model_config.yaml`; dataset fields and vocabularies are in `config/dataset_config.yaml`. Main runs use `max_len: 100`, one training epoch, embedding dimension 16, token dimension 256, three interaction layers, multi-task binary cross-entropy, and task-level AUC/LogLoss. Training uses bf16 autocast and DDP when `torchrun` is selected. Dataset-specific HSAF bucket edges and gate scales are preserved in the YAML files.

## License and acknowledgement

UniRank-derived files retain the original Apache-2.0 headers and this repository includes the Apache 2.0 `LICENSE`. Please see [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) before redistributing or citing this code.
