#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_TencentGR_blocked_seq_action.py
==========================================

Combine TencentGR / TAAC2025 Seq-Action preprocessing and block segmentation to directly generate:

output_dir/
  train/
    data/part-00000.parquet
    user_info/part-00000.parquet
    item_info/part-00000.parquet
    ...
  valid/
    data/part-00000.parquet
    user_info/part-00000.parquet
    item_info/part-00000.parquet
    ...
  test/
    data/part-00000.parquet
    user_info/part-00000.parquet
    item_info/part-00000.parquet
    ...
  meta_data.json
  block_manifest.json

Core design
--------
1. Partition seq.parquet by user_id hash to bound memory usage.
2. Process each partition and split it chronologically with positive-label constraints.
3. Divide each split into multiple blocks.
4. Write separate files for each block:
   - data
   - user_info
   - item_info
5. Remap user_index and item_index to contiguous block-local IDs.
   - Keep user_id and item_id global for consistent embeddings.

Additional rules
--------
1. Remove users whose complete sequence has no click or conversion.
2. Use the shortest trailing interval containing a positive click or conversion for testing.
3. Use the shortest preceding positive interval for validation.
4. Use all remaining events for training.
5. Include item features when calculating vocab_size.
6. Enforce the funnel relation conversion => click.
"""

import argparse
import gc
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ================================================================
# 1. Constant definition
# ================================================================

BEIJING_TZ = "Asia/Shanghai"

USER_SCALAR_FEATURES = ["103", "104", "105", "109"]
USER_LIST_FEATURES = ["106", "107", "108", "110"]
USER_LIST_KEEP = 5

ITEM_STATIC_FEATURES = [
    "100", "101", "102", "112", "114", "115", "116",
    "117", "118", "119", "120", "121", "122"
]

LABEL_COLUMNS = ["is_click", "is_conversion"]
CONTEXT_FEATURES = ["day_of_week", "is_weekend", "hour"]

# OOV filtering threshold: feature values occurring fewer than this threshold are mapped to 0 (unknown/padding)
MIN_FEAT_COUNT = 2

# Features that require OOV filtering (non-ID category features)
# USER_SCALAR_FEATURES and ITEM_STATIC_FEATURES both do OOV filtering
# The sub-elements of USER_LIST_FEATURES are also OOV filtered
OOV_FILTER_SCALAR_FEATURES = USER_SCALAR_FEATURES + ITEM_STATIC_FEATURES
OOV_FILTER_LIST_FEATURES = USER_LIST_FEATURES

FINAL_COLUMNS = (
    ["user_index", "item_index", "seq_len", "user_id", "timestamp"]
    + USER_SCALAR_FEATURES
    + USER_LIST_FEATURES
    + CONTEXT_FEATURES
    + LABEL_COLUMNS
)


# ================================================================
# 2. Utilities
# ================================================================

def prepare_output_dir(output_dir: Path, overwrite: bool = False, data_dir: Path = None):
    # Security check: output_dir is prohibited from being equal to or containing data_dir, otherwise the original data will be deleted
    if data_dir is not None:
        data_dir = Path(data_dir).resolve()
        output_dir_resolved = Path(output_dir).resolve()
        if data_dir == output_dir_resolved or data_dir.is_relative_to(output_dir_resolved):
            raise ValueError(
                f"output_dir cannot be equal to or contain data_dir, otherwise the original data will be deleted! \n"
                f"  data_dir:   {data_dir}\n"
                f"  output_dir: {output_dir_resolved}\n"
                f"Please set output_dir to a different directory than data_dir."
            )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                f"Please remove it first, or use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c) for c in df.columns]
    return df


def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def normalize_list_feature(v, keep=5):
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = [safe_int(x, 0) for x in list(v)[:keep]]
    else:
        arr = []
    if len(arr) < keep:
        arr += [0] * (keep - len(arr))
    return arr


def detect_timestamp_unit_from_sample(values) -> str:
    for v in values:
        try:
            x = float(v)
            if not math.isnan(x):
                return "ms" if x >= 1e12 else "s"
        except Exception:
            continue
    return "ms"


def fmt_date_int(d: int) -> str:
    return f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"


def build_action_maps():
    raw2name = {
        0: "exposure",
        1: "click",
        2: "click|conversion",
    }
    name2code = {name: i + 1 for i, name in enumerate(sorted(set(raw2name.values())))}
    return raw2name, name2code


def get_scalar_vocab_size(series: pd.Series) -> int:
    if len(series) == 0:
        return 1
    mx = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int64).max()
    return int(mx) + 1


def get_list_vocab_size(series: pd.Series) -> int:
    mx = 0
    for arr in series:
        if isinstance(arr, (list, tuple, np.ndarray)):
            for x in arr:
                xi = safe_int(x, 0)
                if xi > mx:
                    mx = xi
    return int(mx) + 1


def count_positive_users_from_action_counter(user_action_counter):
    kept = 0
    dropped = 0
    kept_users = set()
    for uid, mp in user_action_counter.items():
        click_pos = int(mp.get(1, 0))
        conv_pos = int(mp.get(2, 0))
        if click_pos > 0 or conv_pos > 0:
            kept += 1
            kept_users.add(int(uid))
        else:
            dropped += 1
    return kept_users, kept, dropped


def has_required_positive(date_list, date_action_counter,
                          require_click_positive=True,
                          require_conversion_positive=True):
    click_pos = 0
    conv_pos = 0
    for d in date_list:
        mp = date_action_counter.get(int(d), {})
        conversion_count = int(mp.get(2, 0))
        click_pos += int(mp.get(1, 0)) + conversion_count
        conv_pos += conversion_count

    ok = True
    if require_click_positive:
        ok = ok and (click_pos > 0)
    if require_conversion_positive:
        ok = ok and (conv_pos > 0)

    return ok, click_pos, conv_pos


def build_minimal_tail_splits(sorted_dates,
                              date_action_counter,
                              require_click_positive=True,
                              require_conversion_positive=True):
    """
    Fixed rules:
    1) test is the last shortest time period that "contains at least positive samples"
    2) valid is the shortest time period next to test that contains at least positive samples.
    3) Give all the rest to train
    """
    n_days = len(sorted_dates)
    if n_days < 3:
        raise ValueError(f"There are only {n_days} different dates in the data, and it takes at least 3 days to split train/valid/test.")

    # -------------------------
    # Find test first: expand forward from the last day until the positive sample constraint is met
    # -------------------------
    test_start_idx = None
    test_click_pos = 0
    test_conv_pos = 0

    for i in range(n_days - 1, -1, -1):
        cand_test_dates = sorted_dates[i:]
        ok, click_pos, conv_pos = has_required_positive(
            cand_test_dates,
            date_action_counter,
            require_click_positive=require_click_positive,
            require_conversion_positive=require_conversion_positive
        )
        if ok:
            test_start_idx = i
            test_click_pos = click_pos
            test_conv_pos = conv_pos
            break

    if test_start_idx is None:
        raise ValueError("Unable to construct test: There is no positive sample that meets the conditions in the last time period.")

    # Leave at least 1 day for train and valid
    if test_start_idx < 2:
        raise ValueError("Unable to construct a legal segment: after test is the shortest legal tail segment, the front segment is not enough to further divide train and valid.")

    # -------------------------
    # Find valid again: it must be next to test and it must be the shortest legal time period.
    # valid = sorted_dates[j:test_start_idx]
    # Expand forward from test_start_idx-1
    # -------------------------
    valid_start_idx = None
    valid_click_pos = 0
    valid_conv_pos = 0

    for j in range(test_start_idx - 1, -1, -1):
        cand_valid_dates = sorted_dates[j:test_start_idx]
        ok, click_pos, conv_pos = has_required_positive(
            cand_valid_dates,
            date_action_counter,
            require_click_positive=require_click_positive,
            require_conversion_positive=require_conversion_positive
        )
        if ok:
            valid_start_idx = j
            valid_click_pos = click_pos
            valid_conv_pos = conv_pos
            break

    if valid_start_idx is None:
        raise ValueError("Unable to construct valid: There is no positive sample that satisfies the condition in the immediately preceding time period of test.")

    # train stay at least 1 day
    if valid_start_idx < 1:
        raise ValueError("Unable to construct a legal segment: after valid is the shortest legal segment, there is not enough in front for train.")

    train_dates = sorted_dates[:valid_start_idx]
    valid_dates = sorted_dates[valid_start_idx:test_start_idx]
    test_dates = sorted_dates[test_start_idx:]

    return {
        "n_days": n_days,
        "n_train": len(train_dates),
        "n_valid": len(valid_dates),
        "n_test": len(test_dates),
        "train_dates": train_dates,
        "valid_dates": valid_dates,
        "test_dates": test_dates,
        "valid_start_date": valid_dates[0],
        "test_start_date": test_dates[0],
        "valid_click_pos": int(valid_click_pos),
        "valid_conv_pos": int(valid_conv_pos),
        "test_click_pos": int(test_click_pos),
        "test_conv_pos": int(test_conv_pos),
    }


# ================================================================
# 3. Load user/item features
# ================================================================

def load_user_features(data_dir: Path) -> pd.DataFrame:
    fp = data_dir / "user_feat.parquet"
    print(f"  [Load] {fp.name}")
    uf = pd.read_parquet(fp)
    uf = ensure_str_columns(uf)

    if "user_id" not in uf.columns:
        raise ValueError("user_feat.parquet is missing user_id column")

    out = pd.DataFrame()
    out["user_id"] = pd.to_numeric(uf["user_id"], errors="coerce").fillna(-1).astype(np.int64)

    for col in USER_SCALAR_FEATURES:
        if col in uf.columns:
            out[col] = pd.to_numeric(uf[col], errors="coerce").fillna(0).astype(np.int32)
        else:
            out[col] = np.zeros(len(uf), dtype=np.int32)

    for col in USER_LIST_FEATURES:
        if col in uf.columns:
            out[col] = uf[col].map(lambda x: normalize_list_feature(x, USER_LIST_KEEP))
        else:
            out[col] = [[0] * USER_LIST_KEEP for _ in range(len(uf))]

    out = out[out["user_id"] >= 0].drop_duplicates(subset=["user_id"], keep="last").reset_index(drop=True)
    print(f"         {len(out):,} users")
    return out


def load_item_features(data_dir: Path) -> pd.DataFrame:
    fp = data_dir / "item_feat.parquet"
    print(f"  [Load] {fp.name}")
    itf = pd.read_parquet(fp)
    itf = ensure_str_columns(itf)

    if "item_id" not in itf.columns:
        raise ValueError("item_feat.parquet is missing item_id column")

    out = pd.DataFrame()
    out["item_id"] = pd.to_numeric(itf["item_id"], errors="coerce").fillna(-1).astype(np.int64)

    for col in ITEM_STATIC_FEATURES:
        if col in itf.columns:
            out[col] = pd.to_numeric(itf[col], errors="coerce").fillna(0).astype(np.int32)
        else:
            out[col] = np.zeros(len(itf), dtype=np.int32)

    out = out[out["item_id"] >= 0].drop_duplicates(subset=["item_id"], keep="last").reset_index(drop=True)
    print(f"         {len(out):,} items")
    return out


# ================================================================
# 4. feature size / prepare user features
# ================================================================

def build_feature_size_meta(user_feat_df: pd.DataFrame, item_feat_df: pd.DataFrame) -> dict:
    feat_size = {}

    for col in USER_SCALAR_FEATURES:
        feat_size[col] = get_scalar_vocab_size(user_feat_df[col])

    for col in USER_LIST_FEATURES:
        feat_size[col] = get_list_vocab_size(user_feat_df[col])

    # New: Explicitly add item static features to vocab_size statistics
    for col in ITEM_STATIC_FEATURES:
        feat_size[col] = get_scalar_vocab_size(item_feat_df[col])

    feat_size["day_of_week"] = 8
    feat_size["is_weekend"] = 3
    feat_size["hour"] = 25
    return feat_size


def build_feature_size_meta_with_oov(
    user_feat_df: pd.DataFrame,
    item_feat_df: pd.DataFrame,
    scalar_feat_value_counters: dict,
    list_feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """Build feature_size based on frequency statistics and filter feature values occurring fewer than min_feat_count times.

    Args:
        user_feat_df: User feature DataFrame
        item_feat_df: item feature DataFrame
        scalar_feat_value_counters: {feature_name: {value_int: count}}
            Count the frequency of scalar feature values from logs
        list_feat_value_counters: {feature_name: {value_int: count}}
            Count the frequency of list feature sub-elements from logs
        min_feat_count: frequency threshold

    Returns:
        feat_size: {feature_name: vocab_size}
        - vocab_size = maximum value retained + 1
        - Filtered values are mapped to 0 in process_partition
    """
    feat_size = {}

    # ---- USER_SCALAR_FEATURES: OOV based filtering ----
    for col in USER_SCALAR_FEATURES:
        if col in scalar_feat_value_counters:
            counter = scalar_feat_value_counters[col]
            max_kept = 0
            for v, cnt in counter.items():
                v_int = int(v)
                # Values with frequency >= min_feat_count are retained
                if cnt >= min_feat_count:
                    if v_int > max_kept:
                        max_kept = v_int
            # Also consider the value in user_feat_df (even if the log frequency < min_feat_count, the value in the feature table may still need to be retained)
            # However, according to OOV filtering rules, values with log frequency < min_feat_count are uniformly mapped to 0
            # So vocab_size = max_kept + 1 (0 is reserved for padding/filtered values)
            feat_size[col] = max_kept + 1
        else:
            feat_size[col] = get_scalar_vocab_size(user_feat_df[col])

    # ---- USER_LIST_FEATURES: OOV based filtering ----
    for col in USER_LIST_FEATURES:
        if col in list_feat_value_counters:
            counter = list_feat_value_counters[col]
            max_kept = 0
            for v, cnt in counter.items():
                v_int = int(v)
                if cnt >= min_feat_count:
                    if v_int > max_kept:
                        max_kept = v_int
            feat_size[col] = max_kept + 1
        else:
            feat_size[col] = get_list_vocab_size(user_feat_df[col])

    # ---- ITEM_STATIC_FEATURES: OOV based filtering ----
    for col in ITEM_STATIC_FEATURES:
        if col in scalar_feat_value_counters:
            counter = scalar_feat_value_counters[col]
            max_kept = 0
            for v, cnt in counter.items():
                v_int = int(v)
                if cnt >= min_feat_count:
                    if v_int > max_kept:
                        max_kept = v_int
            feat_size[col] = max_kept + 1
        else:
            feat_size[col] = get_scalar_vocab_size(item_feat_df[col])

    feat_size["day_of_week"] = 8
    feat_size["is_weekend"] = 3
    feat_size["hour"] = 25
    return feat_size


def build_oov_filtered_vocab(
    feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """Build the OOV filtered vocab mapping table.

    Args:
        feat_value_counters: {feature_name: {value_int: count}}
        min_feat_count: frequency threshold

    Returns:
        oov_vocab: {feature_name: {value_int: filtered_value_int}}
        - Filtered values (frequency < min_feat_count) are mapped to 0
        - The retained value is mapped to itself
        - When used: df[col] = df[col].map(oov_vocab[col]).fillna(0)
        - Note: Values not in counter will be processed as 0 by fillna(0)
    """
    oov_vocab = {}
    for col, counter in feat_value_counters.items():
        mp = {}
        for v, cnt in counter.items():
            v_int = int(v)
            if cnt >= min_feat_count:
                mp[v_int] = v_int
            else:
                mp[v_int] = 0  # Filtered values map to 0
        oov_vocab[col] = mp
    return oov_vocab


def prepare_user_features(user_feat_df: pd.DataFrame) -> pd.DataFrame:
    out = user_feat_df[["user_id"]].copy()

    for col in USER_SCALAR_FEATURES:
        out[col] = pd.to_numeric(user_feat_df[col], errors="coerce").fillna(0).astype(np.int32)

    for col in USER_LIST_FEATURES:
        out[col] = user_feat_df[col].map(lambda x: normalize_list_feature(x, USER_LIST_KEEP))

    return out


# ================================================================
# 5. seq.parquet batch reading/preprocessing
# ================================================================

def iter_seq_batches(seq_fp: Path, batch_rows: int = 50000):
    pf = pq.ParquetFile(seq_fp)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        df = ensure_str_columns(df)

        n = len(df)
        if n <= batch_rows:
            yield df
        else:
            for start in range(0, n, batch_rows):
                yield df.iloc[start:start + batch_rows].copy()


def preprocess_seq_batch(batch_df: pd.DataFrame, timestamp_unit: str) -> pd.DataFrame:
    if "user_id" not in batch_df.columns or "seq" not in batch_df.columns:
        raise ValueError("seq.parquet needs to contain two columns: user_id and seq")

    batch_df = batch_df[["user_id", "seq"]].copy()
    batch_df["user_id"] = pd.to_numeric(batch_df["user_id"], errors="coerce")
    batch_df = batch_df.dropna(subset=["user_id"])
    batch_df["user_id"] = batch_df["user_id"].astype(np.int64)

    exploded = batch_df.explode("seq", ignore_index=True)
    exploded = exploded[exploded["seq"].notna()].copy()

    if len(exploded) == 0:
        return pd.DataFrame(columns=[
            "user_id", "item_id", "action_type", "timestamp",
            "date", "day_of_week", "is_weekend", "hour"
        ])

    event_df = pd.json_normalize(exploded["seq"])
    event_df.columns = [str(c) for c in event_df.columns]

    df = pd.concat(
        [
            exploded[["user_id"]].reset_index(drop=True),
            event_df.reset_index(drop=True),
        ],
        axis=1,
    )

    required = ["item_id", "action_type", "timestamp"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Expanded seq event is missing field: {c}")

    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["action_type"] = pd.to_numeric(df["action_type"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df.dropna(subset=["user_id", "item_id", "action_type", "timestamp"], inplace=True)

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "user_id", "item_id", "action_type", "timestamp",
            "date", "day_of_week", "is_weekend", "hour"
        ])

    df["user_id"] = df["user_id"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)
    df["action_type"] = df["action_type"].astype(np.int8)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    dt = pd.to_datetime(df["timestamp"], unit=timestamp_unit, utc=True, errors="coerce").dt.tz_convert(BEIJING_TZ)
    valid_mask = dt.notna()
    df = df.loc[valid_mask].copy()
    dt = dt.loc[valid_mask]

    raw_day_of_week = dt.dt.dayofweek.astype(np.int8)
    raw_is_weekend = (dt.dt.dayofweek >= 5).astype(np.int8)
    raw_hour = dt.dt.hour.astype(np.int8)

    df["date"] = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).astype(np.int32)
    df["day_of_week"] = (raw_day_of_week + 1).astype(np.int32)
    df["is_weekend"] = (raw_is_weekend + 1).astype(np.int32)
    df["hour"] = (raw_hour + 1).astype(np.int32)

    return df[[
        "user_id", "item_id", "action_type", "timestamp",
        "date", "day_of_week", "is_weekend", "hour"
    ]]


# ================================================================
# 6. Phase 1: First partition by user_id hash
# ================================================================

def phase1_partition_seq_to_parquet(
    data_dir: Path,
    tmp_dir: Path,
    n_parts: int,
    seq_batch_rows: int,
    buffer_flush_size: int,
    user_feat_df: pd.DataFrame = None,
    item_feat_df: pd.DataFrame = None,
):
    seq_fp = data_dir / "seq.parquet"
    if not seq_fp.exists():
        raise FileNotFoundError(f"File not found: {seq_fp}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_dir / f"part_{p:03d}").mkdir(parents=True, exist_ok=True)

    sample_user_rows = next(iter_seq_batches(seq_fp, batch_rows=1000))
    sample_exp = sample_user_rows[["seq"]].explode("seq", ignore_index=True)
    sample_exp = sample_exp[sample_exp["seq"].notna()]
    sample_evt = pd.json_normalize(sample_exp["seq"]) if len(sample_exp) > 0 else pd.DataFrame()
    sample_timestamps = sample_evt["timestamp"].tolist()[:100] if "timestamp" in sample_evt.columns else []
    timestamp_unit = detect_timestamp_unit_from_sample(sample_timestamps)

    print(f"  [Phase1] Detect timestamp unit: {timestamp_unit}")
    print(f"  [Phase1] Time features are extracted according to Beijing time {BEIJING_TZ} and encoded according to KuaiRand style")

    # ---- Construct user_id -> user feature value mapping (used for OOV frequency statistics) ----
    user_feat_map = None
    if user_feat_df is not None:
        user_feat_map = {}
        for row in user_feat_df.itertuples(index=False):
            uid = int(row.user_id)
            user_feat_map[uid] = {
                col: int(getattr(row, col)) for col in USER_SCALAR_FEATURES
            }
            for col in USER_LIST_FEATURES:
                val = getattr(row, col)
                if isinstance(val, (list, tuple, np.ndarray)):
                    user_feat_map[uid][col] = [safe_int(x, 0) for x in list(val)[:USER_LIST_KEEP]]
                else:
                    user_feat_map[uid][col] = []

    # ---- Construct item_id -> item feature value mapping (used for OOV frequency statistics) ----
    item_feat_map = None
    if item_feat_df is not None:
        item_feat_map = {}
        for row in item_feat_df.itertuples(index=False):
            iid = int(row.item_id)
            item_feat_map[iid] = {
                col: int(getattr(row, col)) for col in ITEM_STATIC_FEATURES
            }

    user_counts = defaultdict(int)
    item_counts = defaultdict(int)
    item_ids = set()
    all_dates = set()
    raw_action_type_counter = defaultdict(int)
    date_action_counter = defaultdict(lambda: defaultdict(int))
    user_action_counter = defaultdict(lambda: defaultdict(int))

    # ---- OOV frequency statistics container ----
    # scalar feature: {feature_name: {value_int: count}}
    scalar_feat_value_counters = {col: defaultdict(int) for col in OOV_FILTER_SCALAR_FEATURES}
    # list feature: {feature_name: {value_int: count}}
    list_feat_value_counters = {col: defaultdict(int) for col in OOV_FILTER_LIST_FEATURES}

    buffers = {p: [] for p in range(n_parts)}
    buf_sizes = {p: 0 for p in range(n_parts)}
    file_counts = {p: 0 for p in range(n_parts)}

    def flush(pid):
        if not buffers[pid]:
            return
        out = pd.concat(buffers[pid], ignore_index=True)
        fp = tmp_dir / f"part_{pid:03d}" / f"c_{file_counts[pid]:04d}.parquet"
        out.to_parquet(fp, index=False, engine="pyarrow")
        file_counts[pid] += 1
        buffers[pid] = []
        buf_sizes[pid] = 0
        del out

    total_rows = 0
    total_users = 0

    for batch_idx, batch_df in enumerate(iter_seq_batches(seq_fp, batch_rows=seq_batch_rows), start=1):
        total_users += len(batch_df)
        chunk = preprocess_seq_batch(batch_df, timestamp_unit=timestamp_unit)

        if len(chunk) == 0:
            del batch_df, chunk
            gc.collect()
            continue

        vc = chunk["user_id"].value_counts(sort=False)
        for uid, cnt in zip(vc.index, vc.values):
            user_counts[int(uid)] += int(cnt)

        # ---- Statistics item_counts ----
        ic = chunk["item_id"].value_counts(sort=False)
        for iid, cnt in zip(ic.index, ic.values):
            item_counts[int(iid)] += int(cnt)

        item_ids.update(chunk["item_id"].unique().tolist())
        all_dates.update(chunk["date"].unique().tolist())

        ac = chunk["action_type"].value_counts(sort=False)
        for a, cnt in zip(ac.index, ac.values):
            raw_action_type_counter[int(a)] += int(cnt)

        dac = chunk.groupby(["date", "action_type"]).size()
        for (d, a), cnt in dac.items():
            date_action_counter[int(d)][int(a)] += int(cnt)

        uac = chunk.groupby(["user_id", "action_type"]).size()
        for (uid, a), cnt in uac.items():
            user_action_counter[int(uid)][int(a)] += int(cnt)

        # ---- OOV frequency statistics: based on user_counts and item_counts ----
        # It is not counted here, but calculated uniformly after the completion of Phase1 (to avoid duplication)

        part_arr = chunk["user_id"].values.astype(np.int64) % n_parts
        for pid in range(n_parts):
            mask = part_arr == pid
            n_match = int(mask.sum())
            if n_match > 0:
                buffers[pid].append(chunk.loc[mask].copy())
                buf_sizes[pid] += n_match
                if buf_sizes[pid] >= buffer_flush_size:
                    flush(pid)

        total_rows += len(chunk)

        if batch_idx % 10 == 0:
            print(f"  [Phase1] batch={batch_idx:5d}  users={total_users:,}  interactions={total_rows:,}")

        del batch_df, chunk, vc, ac, dac, uac, part_arr
        gc.collect()

    for pid in range(n_parts):
        flush(pid)

    positive_users, kept_users_n, dropped_all_negative_users_n = count_positive_users_from_action_counter(user_action_counter)

    print(
        f"  [Phase1] Completed: {total_rows:,} interactions,"
        f"{len(user_counts):,} users, {len(item_ids):,} items, "
        f"{len(all_dates)} dates"
    )
    print(
        f"  [Phase1] All negative sample user filtering statistics:"
        f"kept_positive_users={kept_users_n:,}, "
        f"dropped_all_negative_users={dropped_all_negative_users_n:,}"
    )

    # ---- Calculate OOV frequency statistics based on user_counts and item_counts ----
    if user_feat_map is not None:
        print(f"  [Phase1] Calculate USER_SCALAR_FEATURES / USER_LIST_FEATURES frequency (for OOV filtering)")
        for uid, cnt in user_counts.items():
            if uid in user_feat_map:
                feats = user_feat_map[uid]
                for col in USER_SCALAR_FEATURES:
                    val = feats[col]
                    scalar_feat_value_counters[col][val] += cnt
                for col in USER_LIST_FEATURES:
                    for sub_val in feats[col]:
                        list_feat_value_counters[col][sub_val] += cnt

        # Print OOV filter statistics
        for col in USER_SCALAR_FEATURES:
            n_total = len(scalar_feat_value_counters[col])
            n_kept = sum(1 for v, c in scalar_feat_value_counters[col].items() if c >= MIN_FEAT_COUNT)
            print(f"  [Phase1] OOV {col}: total_values={n_total}, kept_after_filter={n_kept}")
        for col in USER_LIST_FEATURES:
            n_total = len(list_feat_value_counters[col])
            n_kept = sum(1 for v, c in list_feat_value_counters[col].items() if c >= MIN_FEAT_COUNT)
            print(f"  [Phase1] OOV {col}: total_values={n_total}, kept_after_filter={n_kept}")

    if item_feat_map is not None:
        print(f"  [Phase1] Calculate ITEM_STATIC_FEATURES frequency (for OOV filtering)")
        for iid, cnt in item_counts.items():
            if iid in item_feat_map:
                feats = item_feat_map[iid]
                for col in ITEM_STATIC_FEATURES:
                    val = feats[col]
                    scalar_feat_value_counters[col][val] += cnt

        for col in ITEM_STATIC_FEATURES:
            n_total = len(scalar_feat_value_counters[col])
            n_kept = sum(1 for v, c in scalar_feat_value_counters[col].items() if c >= MIN_FEAT_COUNT)
            print(f"  [Phase1] OOV {col}: total_values={n_total}, kept_after_filter={n_kept}")

    # Convert to plain dict
    scalar_feat_value_counters_plain = {
        k: dict(v) for k, v in scalar_feat_value_counters.items()
    }
    list_feat_value_counters_plain = {
        k: dict(v) for k, v in list_feat_value_counters.items()
    }

    return {
        "user_counts": dict(user_counts),
        "item_ids": item_ids,
        "all_dates": all_dates,
        "raw_action_type_counter": dict(raw_action_type_counter),
        "date_action_counter": {int(d): {int(a): int(c) for a, c in mp.items()} for d, mp in date_action_counter.items()},
        "positive_users": positive_users,
        "timestamp_unit": timestamp_unit,
        "total_rows": total_rows,
        "dropped_all_negative_users": int(dropped_all_negative_users_n),
        "scalar_feat_value_counters": scalar_feat_value_counters_plain,
        "list_feat_value_counters": list_feat_value_counters_plain,
    }


# ================================================================
# 7. Process a single partition and obtain the split data of "global id version"
# ================================================================

def process_partition(
    tmp_dir: Path,
    pid: int,
    valid_users: set,
    global_user_index_map: dict,
    global_item_id_map: dict,
    raw2action_name: dict,
    action_name2code: dict,
    user_feat_ready: pd.DataFrame,
    valid_start_date: int,
    test_start_date: int,
    oov_scalar_vocab: dict = None,
    oov_list_vocab: dict = None,
):
    part_dir = tmp_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None

    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        d = d[d["user_id"].isin(valid_users)]
        if len(d) > 0:
            dfs.append(d)
        del d

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    df = df.merge(user_feat_ready, on="user_id", how="left")

    for col in USER_SCALAR_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)
        # ---- Apply OOV filtering ----
        if oov_scalar_vocab is not None and col in oov_scalar_vocab:
            df[col] = df[col].map(oov_scalar_vocab[col]).fillna(0).astype(np.int32)

    for col in USER_LIST_FEATURES:
        if col not in df.columns:
            df[col] = [[0] * USER_LIST_KEEP for _ in range(len(df))]
        df[col] = df[col].map(lambda x: normalize_list_feature(x, USER_LIST_KEEP))
        # ---- Apply OOV filtering (list feature: filter sub-elements) ----
        if oov_list_vocab is not None and col in oov_list_vocab:
            mp = oov_list_vocab[col]
            df[col] = df[col].map(
                lambda lst: [mp.get(int(x), 0) for x in lst]
            )

    df.sort_values(["user_id", "timestamp", "item_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["global_user_index"] = df["user_id"].map(global_user_index_map).astype(np.int32)
    df["global_item_id"] = df["item_id"].map(global_item_id_map).fillna(0).astype(np.int32)

    df["user_id"] = (df["global_user_index"] + 1).astype(np.int32)

    # Funnel-consistent multi-hot labels: every conversion also implies a click.
    df["is_click"] = df["action_type"].isin([1, 2]).astype(np.float32)
    df["is_conversion"] = (df["action_type"] == 2).astype(np.float32)

    df["action_name"] = df["action_type"].map(raw2action_name)
    df["action"] = df["action_name"].map(action_name2code).fillna(0).astype(np.int32)

    for col in CONTEXT_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    df["seq_len"] = df.groupby("global_user_index", sort=False).cumcount().astype(np.int32)

    user_info_rows = []
    for g_uidx, gdf in df.groupby("global_user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(g_uidx),
                "full_item_seq": gdf["global_item_id"].astype(int).tolist(),
                "full_action_seq": gdf["action"].astype(int).tolist(),
                "full_timestamp_seq": gdf["timestamp"].astype(np.int64).tolist(),
                "full_day_of_week_seq": gdf["day_of_week"].astype(int).tolist(),
                "full_is_weekend_seq": gdf["is_weekend"].astype(int).tolist(),
                "full_hour_seq": gdf["hour"].astype(int).tolist(),
            }
        )
    user_info_df = pd.DataFrame(user_info_rows)

    date_col = df["date"]
    train_df = df[date_col < valid_start_date].copy()
    valid_df = df[(date_col >= valid_start_date) & (date_col < test_start_date)].copy()
    test_df = df[date_col >= test_start_date].copy()

    def _select_final(sdf):
        if len(sdf) == 0:
            return pd.DataFrame(columns=FINAL_COLUMNS)

        out = sdf.copy()
        out["user_index"] = out["global_user_index"].astype(np.int32)
        out["item_index"] = out["global_item_id"].astype(np.int32)

        present = [c for c in FINAL_COLUMNS if c in out.columns]
        return out[present].reset_index(drop=True)

    result = {
        "train": _select_final(train_df),
        "valid": _select_final(valid_df),
        "test": _select_final(test_df),
        "user_info": user_info_df,
    }

    del df, train_df, valid_df, test_df, user_info_df
    gc.collect()
    return result


# ================================================================
# 8. split block manager
# ================================================================

class SplitBlockManager:
    def __init__(self, split_name: str, root_dir: Path, num_blocks: int):
        self.split_name = split_name
        self.root_dir = root_dir / split_name
        self.data_dir = self.root_dir / "data"
        self.user_info_dir = self.root_dir / "user_info"
        self.item_info_dir = self.root_dir / "item_info"

        self.num_blocks = int(num_blocks)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_info_dir.mkdir(parents=True, exist_ok=True)
        self.item_info_dir.mkdir(parents=True, exist_ok=True)

        self.data_writers = [None] * self.num_blocks
        self.user_writers = [None] * self.num_blocks

        self.block_rows = [0] * self.num_blocks
        self.block_user_rows = [0] * self.num_blocks

        self.partition_to_block = {}

        self.user_maps = [dict() for _ in range(self.num_blocks)]
        self.item_maps = [{0: 0} for _ in range(self.num_blocks)]

        self.next_user_local = [0] * self.num_blocks
        self.next_item_local = [1] * self.num_blocks

    def choose_block(self, n_rows: int) -> int:
        return int(np.argmin(self.block_rows))

    def _get_or_add_user_local(self, bid: int, global_user_index: int) -> int:
        mp = self.user_maps[bid]
        if global_user_index not in mp:
            mp[global_user_index] = self.next_user_local[bid]
            self.next_user_local[bid] += 1
        return mp[global_user_index]

    def _get_or_add_item_local(self, bid: int, global_item_id: int) -> int:
        mp = self.item_maps[bid]
        if global_item_id not in mp:
            mp[global_item_id] = self.next_item_local[bid]
            self.next_item_local[bid] += 1
        return mp[global_item_id]

    def _write_table(self, writer_list, bid: int, out_fp: Path, df: pd.DataFrame):
        if len(df) == 0:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer_list[bid] is None:
            writer_list[bid] = pq.ParquetWriter(str(out_fp), table.schema)
        writer_list[bid].write_table(table)

    def append_partition(self, pid: int, split_df: pd.DataFrame, user_info_df: pd.DataFrame):
        if len(split_df) == 0:
            return None

        bid = self.choose_block(len(split_df))
        self.partition_to_block[int(pid)] = int(bid)

        active_users = pd.unique(split_df["user_index"].astype(np.int64))
        active_users_set = set(active_users.tolist())

        sub_ui = user_info_df[user_info_df["user_index"].isin(active_users_set)].copy()
        if len(sub_ui) == 0:
            raise ValueError(
                f"[{self.split_name}] partition={pid} There are samples, but the corresponding user_info cannot be found, and the logic is abnormal."
            )

        out_ui_rows = []
        for row in sub_ui.itertuples(index=False):
            g_user = int(row.user_index)
            l_user = self._get_or_add_user_local(bid, g_user)

            g_item_seq = row.full_item_seq if isinstance(row.full_item_seq, (list, tuple, np.ndarray)) else []
            l_item_seq = [self._get_or_add_item_local(bid, safe_int(x, 0)) for x in g_item_seq]

            g_action_seq = row.full_action_seq if isinstance(row.full_action_seq, (list, tuple, np.ndarray)) else []
            g_time_seq = row.full_timestamp_seq if isinstance(row.full_timestamp_seq, (list, tuple, np.ndarray)) else []

            out_ui_rows.append(
                {
                    "user_index": np.int32(l_user),
                    "full_item_seq": [int(x) for x in l_item_seq],
                    "full_action_seq": [int(x) for x in g_action_seq],
                    "full_timestamp_seq": [int(x) for x in g_time_seq],
                    "full_day_of_week_seq": [int(x) for x in getattr(row, "full_day_of_week_seq", [])],
                    "full_is_weekend_seq": [int(x) for x in getattr(row, "full_is_weekend_seq", [])],
                    "full_hour_seq": [int(x) for x in getattr(row, "full_hour_seq", [])],
                }
            )

        out_ui_df = pd.DataFrame(out_ui_rows)

        user_map = self.user_maps[bid]
        item_map = self.item_maps[bid]

        out_df = split_df.copy()
        out_df["user_index"] = out_df["user_index"].map(user_map).astype(np.int32)
        out_df["item_index"] = out_df["item_index"].map(item_map).astype(np.int32)

        data_fp = self.data_dir / f"part-{bid:05d}.parquet"
        ui_fp = self.user_info_dir / f"part-{bid:05d}.parquet"

        self._write_table(self.data_writers, bid, data_fp, out_df)
        self._write_table(self.user_writers, bid, ui_fp, out_ui_df)

        self.block_rows[bid] += len(out_df)
        self.block_user_rows[bid] += len(out_ui_df)

        del out_df, out_ui_df, sub_ui, out_ui_rows
        gc.collect()
        return bid

    def close_writers(self):
        for w in self.data_writers:
            if w is not None:
                w.close()
        for w in self.user_writers:
            if w is not None:
                w.close()

    def write_item_info_blocks(self, global_item_lookup: pd.DataFrame):
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue

            item_map = self.item_maps[bid]
            local_size = max(item_map.values()) if len(item_map) > 0 else 0

            inv_global_item_id = np.zeros(local_size + 1, dtype=np.int32)
            for g_item_id, l_item_idx in item_map.items():
                inv_global_item_id[l_item_idx] = np.int32(g_item_id)

            out = {
                "item_index": np.arange(local_size + 1, dtype=np.int32),
                "item_id": inv_global_item_id.astype(np.int32),
            }

            if local_size > 0:
                gids = inv_global_item_id[1:].astype(np.int32)
                feat_df = global_item_lookup.reindex(gids).fillna(0)

                for col in ITEM_STATIC_FEATURES:
                    arr = np.zeros(local_size + 1, dtype=np.int32)
                    arr[1:] = feat_df[col].to_numpy(dtype=np.int32, copy=False)
                    out[col] = arr
            else:
                for col in ITEM_STATIC_FEATURES:
                    out[col] = np.zeros(1, dtype=np.int32)

            item_info_df = pd.DataFrame(out)
            out_fp = self.item_info_dir / f"part-{bid:05d}.parquet"
            item_info_df.to_parquet(out_fp, index=False, engine="pyarrow")

            del item_info_df
            gc.collect()

    def build_manifest(self):
        blocks = []
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue
            blocks.append(
                {
                    "block_id": bid,
                    "rows": int(self.block_rows[bid]),
                    "users": int(len(self.user_maps[bid])),
                    "items": int(len(self.item_maps[bid]) - 1),
                    "data_file": str(self.data_dir / f"part-{bid:05d}.parquet"),
                    "user_info_file": str(self.user_info_dir / f"part-{bid:05d}.parquet"),
                    "item_info_file": str(self.item_info_dir / f"part-{bid:05d}.parquet"),
                    "source_partitions": [
                        int(pid) for pid, b in self.partition_to_block.items() if b == bid
                    ],
                }
            )
        return {
            "split": self.split_name,
            "num_blocks_configured": int(self.num_blocks),
            "num_blocks_written": int(len(blocks)),
            "blocks": blocks,
        }


# ================================================================
# 9. Build a global item lookup (for use by each block item_info)
# ================================================================

def build_global_item_lookup(item_feat_df: pd.DataFrame, global_item_id_map: dict) -> pd.DataFrame:
    vf = item_feat_df[item_feat_df["item_id"].isin(global_item_id_map)].copy()
    vf["global_item_id"] = vf["item_id"].map(global_item_id_map).astype(np.int32)
    vf = vf[["global_item_id"] + ITEM_STATIC_FEATURES].drop_duplicates(subset=["global_item_id"], keep="last")
    vf = vf.set_index("global_item_id").sort_index()

    for col in ITEM_STATIC_FEATURES:
        vf[col] = pd.to_numeric(vf[col], errors="coerce").fillna(0).astype(np.int32)

    return vf


def build_global_item_lookup_with_oov(
    item_feat_df: pd.DataFrame,
    global_item_id_map: dict,
    oov_scalar_vocab: dict = None,
) -> pd.DataFrame:
    """Build a global item lookup table and apply OOV filtering.

    Args:
        item_feat_df: item feature DataFrame
        global_item_id_map: {item_id: global_item_index}
        oov_scalar_vocab: {feature_name: {value_int: filtered_value_int}}
            Built by build_oov_filtered_vocab

    Returns:
        DataFrame indexed by global_item_id, listed as ITEM_STATIC_FEATURES
    """
    vf = item_feat_df[item_feat_df["item_id"].isin(global_item_id_map)].copy()
    vf["global_item_id"] = vf["item_id"].map(global_item_id_map).astype(np.int32)
    vf = vf[["global_item_id"] + ITEM_STATIC_FEATURES].drop_duplicates(subset=["global_item_id"], keep="last")
    vf = vf.set_index("global_item_id").sort_index()

    for col in ITEM_STATIC_FEATURES:
        vf[col] = pd.to_numeric(vf[col], errors="coerce").fillna(0).astype(np.int32)
        # ---- Apply OOV filtering ----
        if oov_scalar_vocab is not None and col in oov_scalar_vocab:
            mp = oov_scalar_vocab[col]
            vf[col] = vf[col].map(mp).fillna(0).astype(np.int32)

    return vf


# ================================================================
# 10. Main process
# ================================================================

def preprocess_and_split_blocked(
    data_dir: str = "./",
    output_dir: str = "../TencentGR_10M_Action_Blocked",
    min_user_interactions: int = 10,
    n_user_parts: int = 32,
    seq_batch_rows: int = 4_000_000,
    buffer_flush_size: int = 1_000_000,
    train_blocks: int = 32,
    valid_blocks: int = 8,
    test_blocks: int = 8,
    overwrite: bool = False,
    min_feat_count: int = MIN_FEAT_COUNT,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    block_counts = (int(train_blocks), int(valid_blocks), int(test_blocks))
    if min(block_counts) <= 0:
        raise ValueError("train_blocks / valid_blocks / test_blocks must be positive integers.")
    target_blocks = max(block_counts)
    if n_user_parts < target_blocks:
        print(
            f"[Config] n_user_parts={n_user_parts} is less than the maximum target block number {target_blocks},"
            f"Automatically adjust to {target_blocks} to prevent all target blocks from being generated."
        )
        n_user_parts = target_blocks

    prepare_output_dir(output_dir, overwrite=overwrite, data_dir=data_dir)
    tmp_dir = output_dir / "_tmp_partitions"

    print("\n[Step 1/8] Load user/item features & build feature size")
    user_feat_df_raw = load_user_features(data_dir)
    item_feat_df = load_item_features(data_dir)

    raw2action_name, action_name2code = build_action_maps()
    user_feat_ready = prepare_user_features(user_feat_df_raw)

    print(f"  action type: {len(action_name2code)}")
    print("  User features are ready: the scalar retains the original id, and the list retains a single column list[int]\n")

    print("[Step 2/8] Phase 1: seq.parquet -> partition parquet by user_id hash (including OOV frequency statistics)")
    phase1_stat = phase1_partition_seq_to_parquet(
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        n_parts=n_user_parts,
        seq_batch_rows=seq_batch_rows,
        buffer_flush_size=buffer_flush_size,
        user_feat_df=user_feat_df_raw,
        item_feat_df=item_feat_df,
    )

    user_counts = phase1_stat["user_counts"]
    item_ids = phase1_stat["item_ids"]
    all_dates = phase1_stat["all_dates"]
    raw_action_type_counter = phase1_stat["raw_action_type_counter"]
    total_rows_phase1 = phase1_stat["total_rows"]
    timestamp_unit = phase1_stat["timestamp_unit"]
    date_action_counter = phase1_stat["date_action_counter"]
    positive_users = phase1_stat["positive_users"]
    dropped_all_negative_users = phase1_stat["dropped_all_negative_users"]
    scalar_feat_value_counters = phase1_stat.get("scalar_feat_value_counters", {})
    list_feat_value_counters = phase1_stat.get("list_feat_value_counters", {})
    print("")

    # ---- Build OOV filtered vocab ----
    print("[Step 2.5/8] Build OOV filtered vocab")
    print(f"  OOV filter threshold (min_feat_count): {min_feat_count}")

    # Construct OOV filtering mapping table of scalar features
    oov_scalar_vocab = build_oov_filtered_vocab(
        scalar_feat_value_counters, min_feat_count=min_feat_count
    )
    # Build an OOV filter mapping table for list features
    oov_list_vocab = build_oov_filtered_vocab(
        list_feat_value_counters, min_feat_count=min_feat_count
    )

    # Build OOV filtered feature_size_meta
    feature_size_meta = build_feature_size_meta_with_oov(
        user_feat_df_raw,
        item_feat_df,
        scalar_feat_value_counters,
        list_feat_value_counters,
        min_feat_count=min_feat_count,
    )

    # Print OOV filter statistics
    for col in USER_SCALAR_FEATURES + ITEM_STATIC_FEATURES:
        if col in scalar_feat_value_counters:
            counter = scalar_feat_value_counters[col]
            n_total = len(counter)
            n_kept = sum(1 for v, c in counter.items() if c >= min_feat_count)
            n_filtered = n_total - n_kept
            print(f"  [OOV] {col}: total_values={n_total}, kept={n_kept}, filtered={n_filtered}")
    for col in USER_LIST_FEATURES:
        if col in list_feat_value_counters:
            counter = list_feat_value_counters[col]
            n_total = len(counter)
            n_kept = sum(1 for v, c in counter.items() if c >= min_feat_count)
            n_filtered = n_total - n_kept
            print(f"  [OOV] {col}: total_values={n_total}, kept={n_kept}, filtered={n_filtered}")
    print("")

    print("[Step 2.6/8] Split according to fixed rules: shortest legal test + shortest legal valid + remaining train")
    sorted_dates = sorted(all_dates)
    split_info = build_minimal_tail_splits(
        sorted_dates=sorted_dates,
        date_action_counter=date_action_counter,
        require_click_positive=True,
        require_conversion_positive=True,
    )

    train_dates = split_info["train_dates"]
    valid_dates = split_info["valid_dates"]
    test_dates = split_info["test_dates"]
    valid_start_date = split_info["valid_start_date"]
    test_start_date = split_info["test_start_date"]

    print(f"  Total days: {split_info['n_days']} days ({fmt_date_int(sorted_dates[0])} ~ {fmt_date_int(sorted_dates[-1])})")
    print(f"  Training set: {split_info['n_train']} days {fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}")
    print(f"  Validation set: {split_info['n_valid']} days {fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}")
    print(f"  Test set: {split_info['n_test']} day {fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}")
    print(f"  valid positive sample: click={split_info['valid_click_pos']:,}, conversion={split_info['valid_conv_pos']:,}")
    print(f"  test positive sample: click={split_info['test_click_pos']:,}, conversion={split_info['test_conv_pos']:,}")
    print("")

    print("[Step 3/8] Filter low-frequency users + filter all negative sample users + build global ID")
    valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    n_dropped_low_freq = len(user_counts) - len(valid_users)

    valid_users = valid_users.intersection(positive_users)
    n_after_positive_filter = len(valid_users)

    print(f"  Valid users after low frequency filtering: {len(user_counts) - n_dropped_low_freq:,}")
    print(f"  Number of users filtering all negative samples: {dropped_all_negative_users:,}")
    print(f"  Final number of retained users: {n_after_positive_filter:,}")

    sorted_users = sorted(valid_users)
    global_user_index_map = {u: i for i, u in enumerate(sorted_users)}
    sorted_items = sorted(item_ids)
    global_item_id_map = {it: i + 1 for i, it in enumerate(sorted_items)}

    print(f"  Global item count: {len(global_item_id_map):,}")

    vocab_size = {
        "user_id": len(global_user_index_map) + 1,
        "item_id": len(global_item_id_map) + 1,
        "timestamp": 0,
        "action": len(action_name2code) + 1,
    }
    for col in USER_SCALAR_FEATURES:
        vocab_size[col] = int(feature_size_meta[col])
    for col in USER_LIST_FEATURES:
        vocab_size[col] = int(feature_size_meta[col])
    for col in ITEM_STATIC_FEATURES:
        vocab_size[col] = int(feature_size_meta[col])
    for col in CONTEXT_FEATURES:
        vocab_size[col] = int(feature_size_meta[col])

    print("")

    print("[Step 4/8] Encoding + segmentation + writing block data/user_info directly partition by partition")

    managers = {
        "train": SplitBlockManager("train", output_dir, train_blocks),
        "valid": SplitBlockManager("valid", output_dir, valid_blocks),
        "test": SplitBlockManager("test", output_dir, test_blocks),
    }

    sample_counts = {"train": 0, "valid": 0, "test": 0}
    max_seq = 0

    for pid in range(n_user_parts):
        result = process_partition(
            tmp_dir=tmp_dir,
            pid=pid,
            valid_users=valid_users,
            global_user_index_map=global_user_index_map,
            global_item_id_map=global_item_id_map,
            raw2action_name=raw2action_name,
            action_name2code=action_name2code,
            user_feat_ready=user_feat_ready,
            valid_start_date=valid_start_date,
            test_start_date=test_start_date,
            oov_scalar_vocab=oov_scalar_vocab,
            oov_list_vocab=oov_list_vocab,
        )
        if result is None:
            continue

        user_info_df = result["user_info"]
        if len(user_info_df) > 0:
            part_max_seq = user_info_df["full_item_seq"].map(len).max()
            if int(part_max_seq) > max_seq:
                max_seq = int(part_max_seq)

        for split_name in ["train", "valid", "test"]:
            sdf = result[split_name]
            if len(sdf) == 0:
                continue
            managers[split_name].append_partition(pid=pid, split_df=sdf, user_info_df=user_info_df)
            sample_counts[split_name] += len(sdf)

        done = sum(sample_counts.values())
        print(f"  partition {pid + 1:3d}/{n_user_parts}: Write out {done:,} rows cumulatively")

        del result, user_info_df
        gc.collect()

    for mgr in managers.values():
        mgr.close_writers()

    total = sum(sample_counts.values())
    print(
        f"\n  train={sample_counts['train']:,}  "
        f"valid={sample_counts['valid']:,}  "
        f"test={sample_counts['test']:,}"
    )
    print(f"  Total={total:,}\n")

    if total == 0:
        raise ValueError("The data is empty after processing, please check the input data or reduce min_user_interactions.")

    print("[Step 5/8] Build item_info for each block (including OOV filtering)")
    global_item_lookup = build_global_item_lookup_with_oov(
        item_feat_df, global_item_id_map, oov_scalar_vocab=oov_scalar_vocab
    )

    for split_name in ["train", "valid", "test"]:
        print(f"  [Build item_info] {split_name}")
        managers[split_name].write_item_info_blocks(global_item_lookup)
        gc.collect()

    print("\n[Step 6/8] Save block_manifest.json")
    block_manifest = {
        "train": managers["train"].build_manifest(),
        "valid": managers["valid"].build_manifest(),
        "test": managers["test"].build_manifest(),
    }
    with open(output_dir / "block_manifest.json", "w", encoding="utf-8") as f:
        json.dump(block_manifest, f, ensure_ascii=False, indent=4)
    print("  [Saved] block_manifest.json")

    print("\n[Step 7/8] Save meta_data.json")
    meta = {
        "sample_size": {
            "total": int(total),
            "train": int(sample_counts["train"]),
            "valid": int(sample_counts["valid"]),
            "test": int(sample_counts["test"]),
            "phase1_total_interactions": int(total_rows_phase1),
        },
        "split_by_minimal_tail_with_positive_constraints": {
            "timezone": BEIJING_TZ,
            "timestamp_unit": timestamp_unit,
            "train_days": split_info["n_train"],
            "train_range": f"{fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}",
            "valid_days": split_info["n_valid"],
            "valid_range": f"{fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}",
            "test_days": split_info["n_test"],
            "test_range": f"{fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}",
            "valid_click_pos": int(split_info["valid_click_pos"]),
            "valid_conversion_pos": int(split_info["valid_conv_pos"]),
            "test_click_pos": int(split_info["test_click_pos"]),
            "test_conversion_pos": int(split_info["test_conv_pos"]),
            "rule": "test=the last shortest legal positive sample time period; valid=the shortest legal positive sample time period immediately adjacent to test; the rest=train",
        },
        "user_filtering": {
            "min_user_interactions": int(min_user_interactions),
            "dropped_all_negative_users": int(dropped_all_negative_users),
            "rule": "If the user's complete behavior sequence has neither click nor conversion, filter",
        },
        "oov_filter": {
            "min_feat_count": int(min_feat_count),
            "applied_to": {
                "scalar_features": USER_SCALAR_FEATURES + ITEM_STATIC_FEATURES,
                "list_features": USER_LIST_FEATURES,
            },
            "rule": "The uniform mapping of feature value occurrences < min_feat_count is 0 (unknown/padding)",
            "freq_source": "Calculated based on the number of user/item occurrences in the log * feature value",
        },
        "blocked_layout": {
            "n_user_parts": int(n_user_parts),
            "train_blocks": int(train_blocks),
            "valid_blocks": int(valid_blocks),
            "test_blocks": int(test_blocks),
            "train": {
                "data_dir": "train/data",
                "user_info_dir": "train/user_info",
                "item_info_dir": "train/item_info",
            },
            "valid": {
                "data_dir": "valid/data",
                "user_info_dir": "valid/user_info",
                "item_info_dir": "valid/item_info",
            },
            "test": {
                "data_dir": "test/data",
                "user_info_dir": "test/user_info",
                "item_info_dir": "test/item_info",
            },
            "block_pair_rule": (
                "Under the same split, data/user_info/item_info uses the same part-xxxxx number for paired reading."
            ),
            "local_index_rule": {
                "user_index": "block-local dense index, starts from 0",
                "item_index": "block-local dense index, 0 reserved for padding",
                "user_id": "global feature id, consistent across blocks",
                "item_id": "global feature id, consistent across blocks",
            },
        },
        "vocab_size": {k: int(v) for k, v in vocab_size.items()},
        "label": LABEL_COLUMNS,
        "label_semantics": {
            "type": "multi_hot",
            "rule": "action_type=0 -> [0,0]; action_type=1 -> [1,0]; action_type=2 -> [1,1]",
            "funnel_constraint": "is_conversion=1 implies is_click=1",
        },
        "action_vocab": {k: int(v) for k, v in action_name2code.items()},
        "action_vocab_desc": (
            "Encoded action vocabulary for dataloader based on full_action_seq"
            "Construct task-specific token masks."
        ),
        "action_mapping": {
            "raw_action_type_to_name": {str(k): v for k, v in raw2action_name.items()},
            "action_name_to_code": {k: int(v) for k, v in action_name2code.items()},
        },
        "user_info_schema": {
            "fields": [
                "user_index",
                "full_item_seq",
                "full_action_seq",
                "full_timestamp_seq",
            ],
            "desc": (
                "The item index in user_index / full_item_seq here is all block-local index;"
                "full_action_seq / full_timestamp_seq is the global time sequence sequence."
            ),
        },
        "item_info_schema": {
            "fields": [
                "item_index",
                "item_id",
            ] + ITEM_STATIC_FEATURES,
            "desc": (
                "item_index is block-local index; item_id is global item feature id,"
                "Used for embedding consistency."
            ),
        },
        "feature_schema": {
            "user_scalar_features": USER_SCALAR_FEATURES,
            "user_list_features": USER_LIST_FEATURES,
            "user_list_keep": USER_LIST_KEEP,
            "user_list_storage": "single-column list[int] with fixed length",
            "user_scalar_encoding": "keep raw integer ids; vocab_size=max+1; OOV filtered",
            "user_list_encoding": "keep raw list[int]; vocab_size=max_sub_feature_id+1; OOV filtered",
            "item_static_features": ITEM_STATIC_FEATURES,
            "item_static_encoding": "keep raw integer ids; vocab_size=max+1; OOV filtered",
            "context_features": CONTEXT_FEATURES,
            "time_feature_encoding": {
                "day_of_week": "0~6 -> 1~7, 0 reserved for padding/unknown",
                "is_weekend": "0/1 -> 1~2, 0 reserved for padding/unknown",
                "hour": "0~23 -> 1~24, 0 reserved for padding/unknown",
            },
            "timestamp_kept_raw": True,
            "context_time_timezone": BEIJING_TZ,
        },
        "max_len": {
            "full_item_seq": int(max_seq),
            "full_action_seq": int(max_seq),
            "full_timestamp_seq": int(max_seq),
        },
        "raw_action_type_counter": {
            str(k): int(v) for k, v in sorted(raw_action_type_counter.items())
        },
    }

    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    print("  [Saved] meta_data.json")

    print("\n[Step 8/8] Clean up temporary partition files")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  Done.")

    print("\n" + "=" * 68)
    print("  TencentGR / TAAC2025 Blocked Seq-Action Preprocess Done (including OOV filtering)")
    print("=" * 68)
    print(f"Output directory: {output_dir}\n")
    print("Directory structure example:")
    print(f"  {output_dir / 'train' / 'data'}")
    print(f"  {output_dir / 'train' / 'user_info'}")
    print(f"  {output_dir / 'train' / 'item_info'}")
    print(f"  {output_dir / 'valid' / 'data'}")
    print(f"  {output_dir / 'test' / 'data'}")
    print(f"  {output_dir / 'meta_data.json'}")
    print(f"  {output_dir / 'block_manifest.json'}")


# ================================================================
# 11. CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess TencentGR/TAAC2025 seq-action data directly into blocked train/valid/test with block-local user_info/item_info."
    )
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--output_dir", type=str, default="../TencentGR_10M_Action")

    parser.add_argument("--min_user_interactions", type=int, default=10)

    parser.add_argument("--n_user_parts", type=int, default=20,
                        help="Phase1 Number of temporary partitions based on user hash, it is recommended >= max(train_blocks, valid_blocks, test_blocks)")
    parser.add_argument("--seq_batch_rows", type=int, default=4_000_000,
                        help="How many user-record rows are processed at a time when reading seq.parquet")
    parser.add_argument("--buffer_flush_size", type=int, default=1_000_000,
                        help="How much interaction is cached in the temporary partition and then dropped to disk?")

    parser.add_argument("--train_blocks", type=int, default=32)
    parser.add_argument("--valid_blocks", type=int, default=8)
    parser.add_argument("--test_blocks", type=int, default=8)

    parser.add_argument("--overwrite", action="store_true",  default=False,
                        help="If the output directory already exists, delete it and recreate it.")
    parser.add_argument("--min_feat_count", type=int, default=MIN_FEAT_COUNT,
                        help="OOV filtering threshold: The number of occurrences of the feature value < the unified mapping of this value is 0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preprocess_and_split_blocked(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_user_interactions=args.min_user_interactions,
        n_user_parts=args.n_user_parts,
        seq_batch_rows=args.seq_batch_rows,
        buffer_flush_size=args.buffer_flush_size,
        train_blocks=args.train_blocks,
        valid_blocks=args.valid_blocks,
        test_blocks=args.test_blocks,
        overwrite=args.overwrite,
        min_feat_count=args.min_feat_count,
    )
