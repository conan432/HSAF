#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""
preprocess_MerRec_seq_action.py — Mercari MerRec sequential action preprocessing.

Input:
  - data_dir/20231001/*.parquet or data_dir/*.parquet

Output:
  output_dir/
    train/{data,user_info,item_info}/part-xxxxx.parquet
    valid/{data,user_info,item_info}/part-xxxxx.parquet
    test/{data,user_info,item_info}/part-xxxxx.parquet
    meta_data.json
    block_manifest.json

Samples, labels and actions:
  - For each (user_id, item_id), only the earliest item_view is retained as the exposure sample.
  - Backfill later Like/Cart/Offer/Checkout/Purchase events into the retained view as multi-hot labels.
  - Use non-view events only for label backfilling, not as samples or history tokens.
  - Encode each multi-hot label combination as a history action, consistent with KuaiRand.

Dependencies:
    pip install pandas numpy pyarrow
"""

import argparse
import gc
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


MIN_FEAT_COUNT = 2

EVENT_ID_MAP = {
    "item_view": 0,
    "item_like": 1,
    "item_add_to_cart_tap": 2,
    "offer_make": 3,
    "buy_start": 4,
    "buy_comp": 5,
}
EVENT_NAMES = [name for name, _ in sorted(EVENT_ID_MAP.items(), key=lambda kv: kv[1])]
EVENT_TO_ACTION_NAME = {
    "item_view": "exposure",
    "item_like": "Like",
    "item_add_to_cart_tap": "Cart",
    "offer_make": "Offer",
    "buy_start": "Checkout",
    "buy_comp": "Purchase",
}
ACTION_NAMES = [EVENT_TO_ACTION_NAME[name] for name in EVENT_NAMES]
# The original event type is retained in Phase 1 for subsequent backfilling of non-view events into view samples.
RAW_ACTION_VOCAB = {name: idx + 1 for idx, name in enumerate(ACTION_NAMES)}
RAW_VIEW_ACTION_ID = RAW_ACTION_VOCAB["exposure"]

LABEL_COLUMNS = [
    "Like",
    "Cart",
    "Offer",
    "Checkout",
    "Purchase",
]


def build_action_maps():
    """Encode every multi-task label combination as one history action token."""
    pattern_to_name = {}
    for pattern in range(1 << len(LABEL_COLUMNS)):
        if pattern == 0:
            pattern_to_name[pattern] = "exposure"
        else:
            pattern_to_name[pattern] = "|".join(
                label for idx, label in enumerate(LABEL_COLUMNS) if pattern & (1 << idx)
            )
    action_vocab = {
        name: idx + 1 for idx, name in enumerate(sorted(set(pattern_to_name.values())))
    }
    return pattern_to_name, action_vocab

CONTEXT_FEATURES = ["session_id", "day_of_week", "is_weekend", "hour"]
ITEM_STATIC_FEATURES = [
    "product_id",
    "price_bucket",
    "c0_id",
    "c1_id",
    "c2_id",
    "brand_id",
    "item_condition_id",
    "size_id",
    "shipper_id",
    "color",
]

RAW_COLUMNS = [
    "user_id",
    "stime",
    "session_id",
    "event_id",
    "item_id",
    "price",
    "product_id",
    "c0_id",
    "c1_id",
    "c2_id",
    "brand_id",
    "item_condition_id",
    "size_id",
    "shipper_id",
    "color",
]

FINAL_COLUMNS = (
    ["user_index", "item_index", "seq_len", "user_id"]
    + CONTEXT_FEATURES
    + LABEL_COLUMNS
)

TMP_COLUMNS = (
    ["user_id_raw", "item_id_raw", "stime_ms", "date", "_row_id"]
    + CONTEXT_FEATURES
    + ITEM_STATIC_FEATURES
    + ["action"]
    + LABEL_COLUMNS
)

PRICE_BINS = [-np.inf, 0, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf]
PRICE_LABELS = [
    "<=0",
    "0-5",
    "5-10",
    "10-20",
    "20-50",
    "50-100",
    "100-200",
    "200-500",
    "500-1000",
    "1000+",
]


def _sort_key(v):
    s = str(v)
    try:
        return 0, int(s)
    except Exception:
        return 1, s


def clean_categorical(s: pd.Series) -> pd.Series:
    original_na = s.isna()
    out = s.astype(str).str.strip()
    bad = original_na | (out == "") | (out.str.lower().isin(["nan", "none", "null"]))
    out = out.mask(bad, "__MISSING__")
    return out


def parse_stime(s: pd.Series):
    # Parquet's timestamp[us] will be read as datetime64[us] by pandas; if you convert the value first,
    # You get a microsecond integer, and parsing in nanoseconds will incorrectly map the year 2023 to the same hour in 1970.
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, utc=True, errors="coerce")
    else:
        numeric = pd.to_numeric(s, errors="coerce")
        numeric_ratio = float(numeric.notna().mean()) if len(s) > 0 else 0.0
        if numeric_ratio > 0.8:
            median_abs = float(numeric.dropna().abs().median()) if numeric.notna().any() else 0.0
            if median_abs >= 1e17:
                unit = "ns"
            elif median_abs >= 1e14:
                unit = "us"
            elif median_abs >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            dt = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(s, utc=True, errors="coerce")

    valid = dt.notna()
    ts_ms = pd.Series(np.zeros(len(s), dtype=np.int64), index=s.index)
    if valid.any():
        ts_ms.loc[valid] = (dt.loc[valid].astype("int64") // 1_000_000).astype(np.int64)

    date = pd.Series(np.zeros(len(s), dtype=np.int32), index=s.index)
    day_of_week = pd.Series(np.zeros(len(s), dtype=np.int16), index=s.index)
    hour = pd.Series(np.zeros(len(s), dtype=np.int16), index=s.index)
    if valid.any():
        date.loc[valid] = dt.loc[valid].dt.strftime("%Y%m%d").astype(np.int32)
        day_of_week.loc[valid] = dt.loc[valid].dt.dayofweek.astype(np.int16)
        hour.loc[valid] = dt.loc[valid].dt.hour.astype(np.int16)
    is_weekend = day_of_week.isin([5, 6]).astype(np.int16)
    return valid, ts_ms, date, day_of_week.astype(str), is_weekend.astype(str), hour.astype(str)


def bucket_price(s: pd.Series) -> pd.Series:
    price = pd.to_numeric(s, errors="coerce").fillna(-1)
    return pd.cut(price, bins=PRICE_BINS, labels=PRICE_LABELS).astype(str)


def normalize_event_id(s: pd.Series) -> pd.Series:
    raw = clean_categorical(s)
    numeric = pd.to_numeric(raw, errors="coerce")
    out = raw.where(raw.isin(EVENT_ID_MAP), None)
    for idx, name in enumerate(EVENT_NAMES):
        out = out.mask(numeric == idx, name)
    return out


def build_date_split(sorted_dates, train_ratio, valid_ratio, test_ratio):
    n_days = len(sorted_dates)
    if n_days < 3:
        return None
    ratios = np.array([train_ratio, valid_ratio, test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()
    raw = ratios * n_days
    counts = np.floor(raw).astype(int)
    rem = n_days - counts.sum()
    if rem > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in range(rem):
            counts[order[i % 3]] += 1
    for idx in range(3):
        while counts[idx] == 0:
            donors = np.where(counts > 1)[0]
            if len(donors) == 0:
                break
            donor = donors[np.argmax(counts[donors])]
            counts[donor] -= 1
            counts[idx] += 1
    n_train, n_valid = int(counts[0]), int(counts[1])
    return {
        "train_dates": sorted_dates[:n_train],
        "valid_dates": sorted_dates[n_train:n_train + n_valid],
        "test_dates": sorted_dates[n_train + n_valid:],
        "valid_start_date": sorted_dates[n_train],
        "test_start_date": sorted_dates[n_train + n_valid],
        "n_train": int(counts[0]),
        "n_valid": int(counts[1]),
        "n_test": int(counts[2]),
    }


def allocate_split_counts(n, train_ratio, valid_ratio, test_ratio):
    if n < 3:
        return n, 0, 0
    ratios = np.array([train_ratio, valid_ratio, test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()
    raw = ratios * n
    counts = np.floor(raw).astype(int)
    rem = n - counts.sum()
    if rem > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in range(rem):
            counts[order[i % 3]] += 1
    for idx in range(3):
        while counts[idx] == 0:
            donors = np.where(counts > 1)[0]
            if len(donors) == 0:
                break
            donor = donors[np.argmax(counts[donors])]
            counts[donor] -= 1
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def iter_parquet_batches(data_dir: Path, batch_size: int):
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    for fp in files:
        pf = pq.ParquetFile(fp)
        schema_cols = set(pf.schema_arrow.names)
        missing = [c for c in RAW_COLUMNS if c not in schema_cols]
        if missing:
            raise ValueError(f"{fp} missing required columns: {missing}")
        for batch in pf.iter_batches(batch_size=batch_size, columns=RAW_COLUMNS):
            yield fp, batch.to_pandas()


def clean_chunk(chunk: pd.DataFrame, row_offset: int):
    out = pd.DataFrame(index=chunk.index)
    out["user_id_raw"] = clean_categorical(chunk["user_id"])
    out["item_id_raw"] = clean_categorical(chunk["item_id"])

    valid_time, stime_ms, date, day_of_week, is_weekend, hour = parse_stime(chunk["stime"])
    event_name = normalize_event_id(chunk["event_id"])
    valid = (
        (out["user_id_raw"] != "__MISSING__")
        & (out["item_id_raw"] != "__MISSING__")
        & valid_time
        & event_name.notna()
    )
    out = out.loc[valid].copy()
    if len(out) == 0:
        return out, row_offset

    out["stime_ms"] = stime_ms.loc[valid].astype(np.int64)
    out["date"] = date.loc[valid].astype(np.int32)
    out["session_id"] = clean_categorical(chunk.loc[valid, "session_id"])
    out["day_of_week"] = day_of_week.loc[valid]
    out["is_weekend"] = is_weekend.loc[valid]
    out["hour"] = hour.loc[valid]

    out["product_id"] = clean_categorical(chunk.loc[valid, "product_id"])
    out["price_bucket"] = bucket_price(chunk.loc[valid, "price"])
    for col in [
        "c0_id",
        "c1_id",
        "c2_id",
        "brand_id",
        "item_condition_id",
        "size_id",
        "shipper_id",
        "color",
    ]:
        out[col] = clean_categorical(chunk.loc[valid, col])

    action_name = event_name.loc[valid].map(EVENT_TO_ACTION_NAME)
    out["action"] = action_name.map(RAW_ACTION_VOCAB).astype(np.int32)
    for label in LABEL_COLUMNS:
        out[label] = (action_name == label).astype(np.float32)

    n = len(out)
    out["_row_id"] = np.arange(row_offset, row_offset + n, dtype=np.int64)
    out = out[TMP_COLUMNS]
    return out.reset_index(drop=True), row_offset + n


class PartitionWriter:
    def __init__(self, tmp_dir: Path, n_parts: int, flush_size: int):
        self.tmp_dir = tmp_dir
        self.n_parts = int(n_parts)
        self.flush_size = int(flush_size)
        self.part_dirs = [tmp_dir / f"part_{pid:03d}" for pid in range(self.n_parts)]
        for part_dir in self.part_dirs:
            part_dir.mkdir(parents=True, exist_ok=True)
        self.buffers = [[] for _ in range(self.n_parts)]
        self.buffer_rows = [0 for _ in range(self.n_parts)]
        self.file_counts = [0 for _ in range(self.n_parts)]

    def append(self, df: pd.DataFrame):
        if len(df) == 0:
            return
        hashes = pd.util.hash_pandas_object(df["user_id_raw"], index=False).to_numpy(np.uint64)
        part_ids = (hashes % self.n_parts).astype(np.int32)
        for pid in np.unique(part_ids):
            sub = df.iloc[np.where(part_ids == pid)[0]].copy()
            self.buffers[int(pid)].append(sub)
            self.buffer_rows[int(pid)] += len(sub)
            if self.buffer_rows[int(pid)] >= self.flush_size:
                self.flush(int(pid))

    def flush(self, pid: int):
        if not self.buffers[pid]:
            return
        out = pd.concat(self.buffers[pid], ignore_index=True)
        fp = self.part_dirs[pid] / f"chunk-{self.file_counts[pid]:05d}.parquet"
        out.to_parquet(fp, index=False, engine="pyarrow")
        self.file_counts[pid] += 1
        self.buffers[pid] = []
        self.buffer_rows[pid] = 0
        del out
        gc.collect()

    def close(self):
        for pid in range(self.n_parts):
            self.flush(pid)


def phase1_partition_to_parquet(data_dir: Path, tmp_dir: Path, n_parts: int,
                                batch_size: int, buffer_flush_size: int):
    writer = PartitionWriter(tmp_dir, n_parts=n_parts, flush_size=buffer_flush_size)
    user_counts = defaultdict(int)
    item_ids = set()
    all_dates = set()
    feat_counters = {col: Counter() for col in CONTEXT_FEATURES + ITEM_STATIC_FEATURES}
    row_offset = 0
    total_rows = 0

    for fp, chunk in iter_parquet_batches(data_dir, batch_size=batch_size):
        clean, row_offset = clean_chunk(chunk, row_offset)
        if len(clean) == 0:
            continue
        vc = clean["user_id_raw"].value_counts()
        for user_id, cnt in vc.items():
            user_counts[user_id] += int(cnt)
        item_ids.update(clean["item_id_raw"].unique().tolist())
        all_dates.update(int(x) for x in clean["date"].unique().tolist() if int(x) > 0)
        for col in CONTEXT_FEATURES + ITEM_STATIC_FEATURES:
            feat_counters[col].update(clean[col].astype(str).tolist())
        writer.append(clean)
        total_rows += len(clean)
        print(f"  [Phase1] {fp.name}: kept={len(clean):,}, total={total_rows:,}")
        del clean, chunk
        gc.collect()

    writer.close()
    return dict(user_counts), item_ids, sorted(all_dates), feat_counters, total_rows


def build_feature_vocabs(feat_counters: dict, min_feat_count: int):
    vocabs = {}
    for col, counter in feat_counters.items():
        values = [
            v for v, cnt in counter.items()
            if cnt >= min_feat_count and v != "__MISSING__"
        ]
        values = sorted(values, key=_sort_key)
        vocabs[col] = {v: idx + 1 for idx, v in enumerate(values)}
    return vocabs


def encode_series(s: pd.Series, vocab: dict):
    return s.astype(str).map(vocab).fillna(0).astype(np.int32)


def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


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

    def append_partition(self, pid: int, split_df: pd.DataFrame, user_info_rows: list):
        if len(split_df) == 0:
            return None
        bid = self.choose_block(len(split_df))
        self.partition_to_block[int(pid)] = int(bid)
        ui_lookup = {r["user_index"]: r for r in user_info_rows}
        active_users = set(pd.unique(split_df["user_index"].astype(np.int64)).tolist())

        out_ui_rows = []
        for g_user in active_users:
            row = ui_lookup[g_user]
            l_user = self._get_or_add_user_local(bid, int(g_user))
            g_item_seq = row.get("full_item_seq", [])
            g_action_seq = row.get("full_action_seq", [])
            g_time_seq = row.get("full_timestamp_seq", [])
            l_item_seq = [self._get_or_add_item_local(bid, safe_int(x, 0)) for x in g_item_seq]
            out_ui_rows.append(
                {
                    "user_index": np.int32(l_user),
                    "full_item_seq": [int(x) for x in l_item_seq],
                    "full_action_seq": [int(x) for x in g_action_seq],
                    "full_timestamp_seq": [int(x) for x in g_time_seq],
                }
            )

        out_df = split_df.copy()
        out_df["user_index"] = out_df["user_index"].map(self.user_maps[bid]).astype(np.int32)
        out_df["item_index"] = out_df["item_index"].map(self.item_maps[bid]).astype(np.int32)

        self._write_table(self.data_writers, bid, self.data_dir / f"part-{bid:05d}.parquet", out_df)
        self._write_table(
            self.user_writers,
            bid,
            self.user_info_dir / f"part-{bid:05d}.parquet",
            pd.DataFrame(out_ui_rows),
        )
        self.block_rows[bid] += len(out_df)
        self.block_user_rows[bid] += len(out_ui_rows)
        del out_df, out_ui_rows
        gc.collect()
        return bid

    def close_writers(self):
        for writer in self.data_writers + self.user_writers:
            if writer is not None:
                writer.close()

    def write_item_info_blocks(self, global_item_lookup: pd.DataFrame):
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue
            item_map = self.item_maps[bid]
            local_size = max(item_map.values()) if item_map else 0
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

            pd.DataFrame(out).to_parquet(
                self.item_info_dir / f"part-{bid:05d}.parquet",
                index=False,
                engine="pyarrow",
            )

    def buildmanifest(self):
        blocks = []
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue
            blocks.append(
                {
                    "block_id": int(bid),
                    "rows": int(self.block_rows[bid]),
                    "users": int(len(self.user_maps[bid])),
                    "items": int(len(self.item_maps[bid]) - 1),
                    "data_file": str(self.data_dir / f"part-{bid:05d}.parquet"),
                    "user_info_file": str(self.user_info_dir / f"part-{bid:05d}.parquet"),
                    "item_info_file": str(self.item_info_dir / f"part-{bid:05d}.parquet"),
                    "source_partitions": [
                        int(pid) for pid, block_id in self.partition_to_block.items() if block_id == bid
                    ],
                }
            )
        return {"split": self.split_name, "num_blocks": len(blocks), "blocks": blocks}


def split_by_user_ratio(df, train_ratio, valid_ratio, test_ratio):
    split = np.full(len(df), "train", dtype=object)
    for _, idx in df.groupby("user_index", sort=False).indices.items():
        pos = np.asarray(idx)
        n_train, n_valid, _ = allocate_split_counts(len(pos), train_ratio, valid_ratio, test_ratio)
        split[pos[n_train:n_train + n_valid]] = "valid"
        split[pos[n_train + n_valid:]] = "test"
    return split


def build_view_anchor_samples(df: pd.DataFrame, pattern_to_name: dict,
                              action_vocab: dict) -> pd.DataFrame:
    """Collapse an event stream into one multi-label sample per user-item view.

    For each (user_id_raw, item_id_raw), the first item_view is the sole sample
    anchor. All supported non-view events strictly after that view are OR-ed into
    its labels. Events before the first view and groups without a view are ignored.
    """
    group_cols = ["user_id_raw", "item_id_raw"]
    views = df[df["action"] == RAW_VIEW_ACTION_ID]
    if len(views) == 0:
        return df.iloc[0:0].copy()

    anchors = views.drop_duplicates(group_cols, keep="first").copy()
    anchor_order = anchors[group_cols + ["_row_id"]].rename(
        columns={"_row_id": "__anchor_row_id"}
    )

    # Join only user-item groups that have an anchor, then retain events occurring
    # after it. _row_id is the deterministic tie-breaker for equal timestamps.
    grouped_events = df.merge(anchor_order, on=group_cols, how="inner", sort=False)
    followups = grouped_events[
        grouped_events["_row_id"] > grouped_events["__anchor_row_id"]
    ]
    if len(followups) > 0:
        label_values = (
            followups.groupby(group_cols, sort=False)[LABEL_COLUMNS]
            .max()
            .reset_index()
        )
    else:
        label_values = pd.DataFrame(columns=group_cols + LABEL_COLUMNS)

    anchors = anchors.drop(columns=LABEL_COLUMNS).merge(
        label_values, on=group_cols, how="left", sort=False
    )
    for label in LABEL_COLUMNS:
        anchors[label] = pd.to_numeric(anchors[label], errors="coerce").fillna(0).astype(np.float32)

    binary = anchors[LABEL_COLUMNS].to_numpy(dtype=np.int8, copy=False)
    pattern = np.zeros(len(anchors), dtype=np.int32)
    for idx in range(len(LABEL_COLUMNS)):
        pattern |= binary[:, idx].astype(np.int32) << idx
    anchors["action"] = (
        pd.Series(pattern, index=anchors.index)
        .map(pattern_to_name)
        .map(action_vocab)
        .astype(np.int32)
    )
    return anchors


def process_partition(tmp_dir: Path, pid: int, valid_users: set, user_idx_map: dict,
                      item_idx_map: dict, vocabs: dict, split_info: dict,
                      split_strategy: str, train_ratio: float,
                      valid_ratio: float, test_ratio: float,
                      pattern_to_name: dict, action_vocab: dict):
    part_dir = tmp_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        df = df[df["user_id_raw"].isin(valid_users)]
        if len(df) > 0:
            dfs.append(df)
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    df.sort_values(["user_id_raw", "stime_ms", "_row_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = build_view_anchor_samples(df, pattern_to_name, action_vocab)
    if len(df) == 0:
        return None
    df.sort_values(["user_id_raw", "stime_ms", "_row_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["user_index"] = df["user_id_raw"].map(user_idx_map).astype(np.int32)
    df["item_index"] = df["item_id_raw"].map(item_idx_map).fillna(0).astype(np.int32)
    df["user_id"] = (df["user_index"] + 1).astype(np.int32)

    for col in CONTEXT_FEATURES + ITEM_STATIC_FEATURES:
        df[col] = encode_series(df[col], vocabs[col])
    for col in LABEL_COLUMNS:
        df[col] = df[col].astype(np.float32)
    df["action"] = df["action"].astype(np.int32)
    df["stime_ms"] = df["stime_ms"].astype(np.int64)

    df["seq_len"] = df.groupby("user_index", sort=False).cumcount().astype(np.int32)

    user_info_rows = []
    for uidx, gdf in df.groupby("user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(uidx),
                "full_item_seq": gdf["item_index"].astype(int).tolist(),
                "full_action_seq": gdf["action"].astype(int).tolist(),
                "full_timestamp_seq": gdf["stime_ms"].astype(np.int64).tolist(),
                "full_day_of_week_seq": gdf["day_of_week"].astype(int).tolist(),
                "full_is_weekend_seq": gdf["is_weekend"].astype(int).tolist(),
                "full_hour_seq": gdf["hour"].astype(int).tolist(),
                "full_session_id_seq": gdf["session_id"].astype(int).tolist(),
            }
        )

    item_features = (
        df[["item_index"] + ITEM_STATIC_FEATURES]
        .drop_duplicates("item_index", keep="last")
        .reset_index(drop=True)
    )

    if split_strategy == "date" and split_info is not None:
        train_df = df[df["date"] < split_info["valid_start_date"]]
        valid_df = df[
            (df["date"] >= split_info["valid_start_date"])
            & (df["date"] < split_info["test_start_date"])
        ]
        test_df = df[df["date"] >= split_info["test_start_date"]]
    else:
        split = split_by_user_ratio(df, train_ratio, valid_ratio, test_ratio)
        train_df = df[split == "train"]
        valid_df = df[split == "valid"]
        test_df = df[split == "test"]

    def select_final(sdf):
        if len(sdf) == 0:
            return pd.DataFrame(columns=FINAL_COLUMNS)
        return sdf[FINAL_COLUMNS].reset_index(drop=True)

    result = {
        "train": select_final(train_df),
        "valid": select_final(valid_df),
        "test": select_final(test_df),
        "user_info": user_info_rows,
        "item_features": item_features,
    }
    del df, train_df, valid_df, test_df
    gc.collect()
    return result


def resolve_raw_data_dir(data_dir: Path):
    preferred = data_dir / "20231001"
    if preferred.exists():
        return preferred
    return data_dir


def preprocess_and_split(
    data_dir: str,
    output_dir: str,
    min_user_interactions: int = 10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_strategy: str = "auto",
    n_user_parts: int = 64,
    batch_size: int = 4_000_000,
    buffer_flush_size: int = 1_000_000,
    train_blocks: int = 32,
    valid_blocks: int = 8,
    test_blocks: int = 8,
    min_feat_count: int = MIN_FEAT_COUNT,
    overwrite: bool = False,
):
    data_dir = Path(data_dir).resolve()
    raw_data_dir = resolve_raw_data_dir(data_dir)
    output_dir = Path(output_dir).resolve()

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

    if raw_data_dir == output_dir or raw_data_dir.is_relative_to(output_dir):
        raise ValueError(f"output_dir must not contain raw data dir: {raw_data_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_partitions"

    print(f"[Step 1/7] Phase1 parquet scan: {raw_data_dir}")
    user_counts, item_ids, all_dates, feat_counters, total_raw = phase1_partition_to_parquet(
        raw_data_dir,
        tmp_dir,
        n_parts=n_user_parts,
        batch_size=batch_size,
        buffer_flush_size=buffer_flush_size,
    )
    print(f"  kept rows={total_raw:,}, users={len(user_counts):,}, items={len(item_ids):,}")

    print("[Step 2/7] Build vocab and split plan")
    valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    if not valid_users:
        raise ValueError("No valid users after min_user_interactions filtering.")
    user_idx_map = {u: i for i, u in enumerate(sorted(valid_users, key=_sort_key))}
    item_idx_map = {it: i + 1 for i, it in enumerate(sorted(item_ids, key=_sort_key))}
    vocabs = build_feature_vocabs(feat_counters, min_feat_count=min_feat_count)

    date_split = build_date_split(all_dates, train_ratio, valid_ratio, test_ratio)
    if split_strategy == "auto":
        actual_split_strategy = "date" if date_split is not None else "user_ratio"
    else:
        actual_split_strategy = split_strategy
    if actual_split_strategy == "date" and date_split is None:
        raise ValueError("split_strategy=date requires at least 3 distinct dates.")
    print(f"  split_strategy={actual_split_strategy}, dates={len(all_dates)}")

    pattern_to_name, action_vocab = build_action_maps()

    vocab_size = {
        "user_index": len(user_idx_map),
        "item_index": len(item_idx_map) + 1,
        "user_id": len(user_idx_map) + 1,
        "item_id": len(item_idx_map) + 1,
        "action": len(action_vocab) + 1,
        "timestamp": 0,
    }
    for col in CONTEXT_FEATURES + ITEM_STATIC_FEATURES:
        vocab_size[col] = len(vocabs[col]) + 1

    managers = {
        "train": SplitBlockManager("train", output_dir, train_blocks),
        "valid": SplitBlockManager("valid", output_dir, valid_blocks),
        "test": SplitBlockManager("test", output_dir, test_blocks),
    }
    sample_counts = {"train": 0, "valid": 0, "test": 0}
    max_seq = 0
    item_feature_parts = []

    print("[Step 3/7] Encode partitions and write blocked data/user_info")
    for pid in range(n_user_parts):
        result = process_partition(
            tmp_dir,
            pid,
            valid_users,
            user_idx_map,
            item_idx_map,
            vocabs,
            date_split,
            actual_split_strategy,
            train_ratio,
            valid_ratio,
            test_ratio,
            pattern_to_name,
            action_vocab,
        )
        if result is None:
            continue
        for row in result["user_info"]:
            max_seq = max(max_seq, len(row["full_item_seq"]))
        item_feature_parts.append(result["item_features"])
        for split_name in ["train", "valid", "test"]:
            sdf = result[split_name]
            if len(sdf) == 0:
                continue
            managers[split_name].append_partition(pid, sdf, result["user_info"])
            sample_counts[split_name] += len(sdf)
        print(
            f"  partition {pid + 1:3d}/{n_user_parts}: "
            f"train={sample_counts['train']:,}, valid={sample_counts['valid']:,}, "
            f"test={sample_counts['test']:,}"
        )
        del result
        gc.collect()

    for manager in managers.values():
        manager.close_writers()

    total = sum(sample_counts.values())
    if total == 0:
        raise ValueError("Processed data is empty.")

    print("[Step 4/7] Build item_info blocks")
    if item_feature_parts:
        global_item_lookup = (
            pd.concat(item_feature_parts, ignore_index=True)
            .drop_duplicates("item_index", keep="last")
            .set_index("item_index")
            .sort_index()
        )
    else:
        global_item_lookup = pd.DataFrame(columns=ITEM_STATIC_FEATURES)
    for manager in managers.values():
        manager.write_item_info_blocks(global_item_lookup)

    print("[Step 5/7] Save meta_data.json and block_manifest.json")
    meta = {
        "sample_size": {
            "total": int(total),
            "train": int(sample_counts["train"]),
            "valid": int(sample_counts["valid"]),
            "test": int(sample_counts["test"]),
        },
        "split_strategy": actual_split_strategy,
        "split_by_date": None if date_split is None else {
            "train_days": int(date_split["n_train"]),
            "valid_days": int(date_split["n_valid"]),
            "test_days": int(date_split["n_test"]),
            "train_range": f"{date_split['train_dates'][0]} ~ {date_split['train_dates'][-1]}",
            "valid_range": f"{date_split['valid_dates'][0]} ~ {date_split['valid_dates'][-1]}",
            "test_range": f"{date_split['test_dates'][0]} ~ {date_split['test_dates'][-1]}",
        },
        "oov_filter": {
            "min_feat_count": int(min_feat_count),
            "applied_to": CONTEXT_FEATURES + ITEM_STATIC_FEATURES,
            "rule": "feature values with count < min_feat_count are mapped to 0",
        },
        "blocked_layout": {
            "n_user_parts": int(n_user_parts),
            "train_blocks": int(train_blocks),
            "valid_blocks": int(valid_blocks),
            "test_blocks": int(test_blocks),
            "train": {"data_dir": "train/data", "user_info_dir": "train/user_info", "item_info_dir": "train/item_info"},
            "valid": {"data_dir": "valid/data", "user_info_dir": "valid/user_info", "item_info_dir": "valid/item_info"},
            "test": {"data_dir": "test/data", "user_info_dir": "test/user_info", "item_info_dir": "test/item_info"},
        },
        "event_id_map": EVENT_ID_MAP,
        "event_to_action_name": EVENT_TO_ACTION_NAME,
        "label": LABEL_COLUMNS,
        "action_vocab": action_vocab,
        "action_vocab_desc": (
            "One action token per view sample; each token encodes the multi-label "
            "outcomes matched from later events for the same user-item."
        ),
        "vocab_size": {k: int(v) for k, v in vocab_size.items()},
        "feature_groups": {
            "context_features": CONTEXT_FEATURES,
            "item_static_features": ITEM_STATIC_FEATURES,
        },
        "sample_construction": {
            "anchor": "first item_view per user_id + item_id",
            "labels": (
                "Like/Cart/Offer/Checkout/Purchase events strictly after the anchor "
                "view are OR-ed into that view's multi-label target."
            ),
            "non_view_events": "used only to fill labels; not retained as samples or sequence tokens",
        },
        "user_info_schema": {
            "fields": ["user_index", "full_item_seq", "full_action_seq", "full_timestamp_seq"],
        },
        "item_info_schema": {
            "fields": ["item_index", "item_id"] + ITEM_STATIC_FEATURES,
        },
        "max_len": {
            "full_item_seq": int(max_seq),
            "full_action_seq": int(max_seq),
            "full_timestamp_seq": int(max_seq),
        },
    }
    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    block_manifest = {
        "train": managers["train"].buildmanifest(),
        "valid": managers["valid"].buildmanifest(),
        "test": managers["test"].buildmanifest(),
    }
    with open(output_dir / "block_manifest.json", "w", encoding="utf-8") as f:
        json.dump(block_manifest, f, ensure_ascii=False, indent=4)

    print("[Step 6/7] Clean temp files")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[Step 7/7] Done")
    print(f"  output_dir={output_dir}")
    print(json.dumps(meta, ensure_ascii=False, indent=4))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess MerRec parquet data into UniRank blocked seq-action format."
    )
    parser.add_argument("--data_dir", type=str, default="./data/MerRec/raw")
    parser.add_argument("--output_dir", type=str, default="./data/MerRec/UniRank-MerRec")
    parser.add_argument("--min_user_interactions", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_strategy", type=str, default="auto", choices=["auto", "date", "user_ratio"])
    parser.add_argument("--n_user_parts", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4_000_000)
    parser.add_argument("--buffer_flush_size", type=int, default=1_000_000)
    parser.add_argument("--train_blocks", type=int, default=32)
    parser.add_argument("--valid_blocks", type=int, default=8)
    parser.add_argument("--test_blocks", type=int, default=8)
    parser.add_argument("--min_feat_count", type=int, default=MIN_FEAT_COUNT)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_and_split(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_user_interactions=args.min_user_interactions,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        split_strategy=args.split_strategy,
        n_user_parts=args.n_user_parts,
        batch_size=args.batch_size,
        buffer_flush_size=args.buffer_flush_size,
        train_blocks=args.train_blocks,
        valid_blocks=args.valid_blocks,
        test_blocks=args.test_blocks,
        min_feat_count=args.min_feat_count,
        overwrite=args.overwrite,
    )
