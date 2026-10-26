#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Statistics Taobao_Action blocked data set information.

In addition to the general blocked data set statistics, additional statistics:
  - The number of positive and negative samples, the rate of positive and negative samples, and the proportion of positive samples for each label in train/valid/test/total
  - action token distribution
  - behavior_log matching statistics in meta_data.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset_stats_utils import (  # noqa: E402
    DatasetSpec,
    SPLITS,
    compute_stats,
    list_parquet_files,
    load_json,
    print_stats,
    read_existing_columns,
)


DEFAULT_OUTPUT_DIR = Path("./data/Taobao/UniRank-Taobao")

USER_STATIC_FEATURES = [
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
]

ITEM_STATIC_FEATURES = [
    "cate_id",
    "campaign_id",
    "customer_id",
    "brand",
    "price_bucket",
]

CONTEXT_FEATURES = ["pid", "day_of_week", "is_weekend", "hour"]

FEATURE_FIELDS = (
    ["user_index", "item_index", "seq_len", "user_id", "item_id"]
    + ITEM_STATIC_FEATURES
    + USER_STATIC_FEATURES
    + CONTEXT_FEATURES
    + ["action"]
)

LABEL_COLUMNS = ["is_click", "cart", "fav", "buy"]

SPEC = DatasetSpec(
    name="Taobao_Action",
    default_output_dir=DEFAULT_OUTPUT_DIR,
    feature_fields=FEATURE_FIELDS,
    label_columns=LABEL_COLUMNS,
)


def compute_label_stats(output_dir: Path, label_columns):
    def _empty_label_stats():
        return {
            "positive": 0,
            "negative": 0,
            "total": 0,
            "positive_rate": 0.0,
            "negative_rate": 0.0,
            "positive_share": 0.0,
        }

    stats = {
        split: {label: _empty_label_stats() for label in label_columns}
        for split in SPLITS
    }
    stats["total"] = {label: _empty_label_stats() for label in label_columns}

    for split in SPLITS:
        data_dir = output_dir / split / "data"
        for fp in list_parquet_files(data_dir):
            df = read_existing_columns(fp, label_columns)
            n = int(len(df))
            for label in label_columns:
                pos = int(pd.to_numeric(df[label], errors="coerce").fillna(0).sum())
                stats[split][label]["positive"] += pos
                stats[split][label]["total"] += n
                stats["total"][label]["positive"] += pos
                stats["total"][label]["total"] += n

    for split in list(SPLITS) + ["total"]:
        split_positive_sum = sum(stats[split][label]["positive"] for label in label_columns)
        for label in label_columns:
            total = stats[split][label]["total"]
            pos = stats[split][label]["positive"]
            neg = int(total - pos)
            stats[split][label]["negative"] = neg
            stats[split][label]["positive_rate"] = float(pos / total) if total > 0 else 0.0
            stats[split][label]["negative_rate"] = float(neg / total) if total > 0 else 0.0
            stats[split][label]["positive_share"] = (
                float(pos / split_positive_sum) if split_positive_sum > 0 else 0.0
            )

    return stats


def build_pattern_to_action_id(action_vocab: dict, label_columns):
    pattern_to_action_id = {}
    for pattern in range(1 << len(label_columns)):
        if pattern == 0:
            action_name = "exposure"
        else:
            action_name = "|".join(
                label for i, label in enumerate(label_columns) if pattern & (1 << i)
            )
        if action_name in action_vocab:
            pattern_to_action_id[pattern] = int(action_vocab[action_name])
    return pattern_to_action_id


def compute_action_stats(output_dir: Path, action_vocab: dict, label_columns):
    """Statistical action distribution.

    Taobao's data parquet does not directly save the `action` column, `action` is only used for
    user_info.full_action_seq and dataloader. So here press multitasking in data
    The label column reconstructs the action pattern consistent with the preprocessing script, and then maps it to meta_data.json
    action_vocab in.
    """
    id_to_name = {int(v): str(k) for k, v in action_vocab.items()}
    pattern_to_action_id = build_pattern_to_action_id(action_vocab, label_columns)
    action_counts = {split: defaultdict(int) for split in SPLITS}
    action_counts["total"] = defaultdict(int)

    for split in SPLITS:
        data_dir = output_dir / split / "data"
        for fp in list_parquet_files(data_dir):
            df = read_existing_columns(fp, label_columns)
            binary = df[label_columns].apply(
                lambda s: pd.to_numeric(s, errors="coerce").fillna(0) > 0
            ).astype("int8")
            pattern = None
            for i, label in enumerate(label_columns):
                part = binary[label].astype("int32") * (1 << i)
                pattern = part if pattern is None else pattern + part
            vc = pattern.map(pattern_to_action_id).dropna().astype(int).value_counts()
            for action_id, cnt in vc.items():
                action_id = int(action_id)
                cnt = int(cnt)
                action_counts[split][action_id] += cnt
                action_counts["total"][action_id] += cnt

    result = {}
    for split in list(SPLITS) + ["total"]:
        total = int(sum(action_counts[split].values()))
        rows = []
        for action_id in sorted(action_counts[split]):
            cnt = int(action_counts[split][action_id])
            rows.append(
                {
                    "action_id": int(action_id),
                    "action_name": id_to_name.get(action_id, str(action_id)),
                    "count": cnt,
                    "rate": float(cnt / total) if total > 0 else 0.0,
                }
            )
        result[split] = {"total": total, "distribution": rows}
    return result


def print_label_stats(label_stats: dict, label_columns):
    print("\nLabel ratio statistics:")
    header = (
        f"{'split':<8} {'label':<10} {'positive':>14} {'negative':>14} "
        f"{'total':>14} {'pos_rate':>12} {'neg_rate':>12} {'pos_share':>12}"
    )
    print(header)
    print("-" * len(header))
    for split in list(SPLITS) + ["total"]:
        for label in label_columns:
            row = label_stats[split][label]
            print(
                f"{split:<8} {label:<10} "
                f"{row['positive']:>14,} {row['negative']:>14,} {row['total']:>14,} "
                f"{row['positive_rate']:>12.6f} {row['negative_rate']:>12.6f} "
                f"{row['positive_share']:>12.6f}"
            )


def print_action_stats(action_stats: dict):
    total_dist = action_stats.get("total", {}).get("distribution", [])
    if not total_dist:
        return
    print("\nAction token distribution (total):")
    header = f"{'id':>4} {'action':<32} {'count':>14} {'rate':>12}"
    print(header)
    print("-" * len(header))
    for row in total_dist:
        print(
            f"{row['action_id']:>4} {row['action_name']:<32} "
            f"{row['count']:>14,} {row['rate']:>12.6f}"
        )


def print_behavior_meta(meta: dict):
    behavior = meta.get("behavior_log_usage") or {}
    if not behavior:
        return
    print("\nbehavior_log matching statistics:")
    print(f"used: {behavior.get('used')}")
    stats = behavior.get("stats") or {}
    if stats:
        print(f"raw_rows:  {int(stats.get('raw_rows', 0)):,}")
        print(f"kept_rows: {int(stats.get('kept_rows', 0)):,}")
        label_counts = stats.get("label_counts") or {}
        if label_counts:
            print("label_counts:")
            for label, cnt in label_counts.items():
                print(f"  {label}: {int(cnt):,}")
    if behavior.get("match_key"):
        print("match_key: " + " + ".join(behavior["match_key"]))


def run_cli():
    parser = argparse.ArgumentParser(description="Statistics Taobao_Action blocked data set information")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SPEC.default_output_dir),
        help="blocked output directory generated by the preprocessing script",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional: Save statistical results as a JSON file",
    )
    parser.add_argument(
        "--skip_action_stats",
        action="store_true",
        default=False,
        help="Skip action token distribution statistics",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    meta = load_json(output_dir / "meta_data.json")
    labels = meta.get("label") or LABEL_COLUMNS

    stats = compute_stats(SPEC, output_dir)
    stats["label_positive_stats"] = compute_label_stats(output_dir, labels)
    stats["behavior_log_usage"] = meta.get("behavior_log_usage", {})
    if not args.skip_action_stats:
        stats["action_stats"] = compute_action_stats(
            output_dir, meta.get("action_vocab", {}), labels
        )

    print_stats(stats)
    print_label_stats(stats["label_positive_stats"], labels)
    print_action_stats(stats.get("action_stats", {}))
    print_behavior_meta(meta)

    if args.json:
        json_path = Path(args.json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"\n statistical results have been saved: {json_path}")


if __name__ == "__main__":
    run_cli()
