#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for blocked dataset statistics."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    default_output_dir: Path
    feature_fields: Sequence[str]
    label_columns: Sequence[str]


def list_parquet_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.glob("*.parquet") if p.is_file())


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parquet_num_rows(path: Path) -> int:
    return int(pq.ParquetFile(str(path)).metadata.num_rows)


def parquet_columns(path: Path) -> List[str]:
    return list(pq.ParquetFile(str(path)).schema_arrow.names)


def read_existing_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    existing = set(parquet_columns(path))
    selected = [c for c in columns if c in existing]
    missing = [c for c in columns if c not in existing]
    if missing:
        raise ValueError(f"{path} Missing required column: {missing}")
    return pd.read_parquet(path, columns=selected)


def seq_length(value) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and np.isnan(value):
        return 0
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value)
    try:
        return len(value)
    except TypeError:
        return 0


def update_set_from_series(target: set, series: pd.Series, drop_zero: bool = True):
    values = pd.to_numeric(series, errors="coerce").dropna().astype(np.int64)
    if drop_zero:
        values = values[values != 0]
    target.update(values.tolist())


def data_item_user_stats(output_dir: Path) -> Tuple[Dict[str, int], set, Dict[int, int]]:
    sample_counts = {s: 0 for s in SPLITS}
    users = set()
    user_seq_lens: Dict[int, int] = {}

    for split in SPLITS:
        data_dir = output_dir / split / "data"
        user_info_dir = output_dir / split / "user_info"
        data_files = list_parquet_files(data_dir)
        sample_counts[split] = sum(parquet_num_rows(fp) for fp in data_files)

        user_info_by_name = {fp.name: fp for fp in list_parquet_files(user_info_dir)}
        for data_fp in data_files:
            data_df = read_existing_columns(data_fp, ["user_index", "user_id"])
            update_set_from_series(users, data_df["user_id"], drop_zero=True)

            user_info_fp = user_info_by_name.get(data_fp.name)
            if user_info_fp is None:
                continue

            mapping = data_df.drop_duplicates("user_index")
            user_info_df = read_existing_columns(user_info_fp, ["user_index", "full_item_seq"])
            merged = user_info_df.merge(mapping, on="user_index", how="left")
            merged = merged.dropna(subset=["user_id"])

            for row in merged.itertuples(index=False):
                user_id = int(row.user_id)
                length = seq_length(row.full_item_seq)
                if length > user_seq_lens.get(user_id, 0):
                    user_seq_lens[user_id] = length

    return sample_counts, users, user_seq_lens


def item_stats(output_dir: Path) -> set:
    items = set()
    for split in SPLITS:
        item_info_dir = output_dir / split / "item_info"
        for fp in list_parquet_files(item_info_dir):
            item_df = read_existing_columns(fp, ["item_id"])
            update_set_from_series(items, item_df["item_id"], drop_zero=True)
    return items


def compute_stats(spec: DatasetSpec, output_dir: Path) -> dict:
    output_dir = output_dir.expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"The output directory does not exist: {output_dir}")

    meta = load_json(output_dir / "meta_data.json")
    sample_counts, users, user_seq_lens = data_item_user_stats(output_dir)
    items = item_stats(output_dir)

    total_samples = int(sum(sample_counts.values()))
    seq_lengths = list(user_seq_lens.values())
    labels = meta.get("label") or list(spec.label_columns)

    return {
        "dataset": spec.name,
        "output_dir": str(output_dir),
        "user_count": int(len(users)),
        "item_count": int(len(items)),
        "sample_count": {
            "total": total_samples,
            **{k: int(v) for k, v in sample_counts.items()},
        },
        "avg_sequence_length": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
        "max_sequence_length": int(max(seq_lengths)) if seq_lengths else 0,
        "feature_field_count": int(len(spec.feature_fields)),
        "task_count": int(len(labels)),
        "feature_fields": list(spec.feature_fields),
        "label_columns": list(labels),
        "sequence_user_count": int(len(seq_lengths)),
        "meta_sample_count": meta.get("sample_size", {}),
        "meta_max_len": meta.get("max_len", {}),
    }


def print_stats(stats: dict):
    sample = stats["sample_count"]
    print("=" * 72)
    print(f"Dataset: {stats['dataset']}")
    print(f"Output directory: {stats['output_dir']}")
    print("=" * 72)
    print(f"user quantity: {stats['user_count']:,}")
    print(f"item quantity: {stats['item_count']:,}")
    print(
        "Sample size:"
        f"{sample['total']:,} "
        f"(train={sample['train']:,}, valid={sample['valid']:,}, test={sample['test']:,})"
    )
    print(f"Average sequence length: {stats['avg_sequence_length']:.4f}")
    print(f"Maximum sequence length: {stats['max_sequence_length']:,}")
    print(f"feature field quantity: {stats['feature_field_count']:,}")
    print(f"Number of tasks: {stats['task_count']:,}")
    print("=" * 72)


def run_cli(spec: DatasetSpec):
    parser = argparse.ArgumentParser(description=f"Statistics {spec.name} blocked data set information")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(spec.default_output_dir),
        help="blocked output directory generated by the preprocessing script",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional: Save statistical results as a JSON file",
    )
    args = parser.parse_args()

    stats = compute_stats(spec, Path(args.output_dir))
    print_stats(stats)

    if args.json:
        json_path = Path(args.json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"Statistical results have been saved: {json_path}")
