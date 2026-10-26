#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_Taobao_seq_action.py — Ali_Display_Ad_Click / Taobao preprocessing
====================================================================================

Input file (supports .csv / .csv.gz / .csv.tar.gz):
  - raw_sample.csv[.tar.gz]: display/click sample skeleton
  - ad_feature.csv[.tar.gz]: ad features
  - user_profile.csv[.tar.gz]: user features
  - behavior_log.csv[.tar.gz]: 22-day shopping log for cart/fav/buy labels

Split strategy:
  - Convert raw_sample.time_stamp to dates and sort chronologically.
  - Split dates into train/valid/test with the default 8:1:1 ratio.
  - For the official eight-day sample, use days 1--6 for training, day 7 for validation, and day 8 for testing.

Output directory structure (blocked):
  output_dir/
    train/{data,user_info,item_info}/part-xxxxx.parquet
    valid/{data,user_info,item_info}/part-xxxxx.parquet
    test/{data,user_info,item_info}/part-xxxxx.parquet
    meta_data.json
    block_manifest.json

Notes:
  - Build full_item_seq, full_action_seq, and full_timestamp_seq from ad impressions in raw_sample.
  - Write ad_feature to item_info and user_profile fields to each sample.
  - Derive is_click from raw_sample.clk and cart/fav/buy from behavior_log.btag.
  - Map adgroup_id to cate_id and brand, then match later behavior_log events by user_id, cate_id, and brand within the label window.

Dependencies:
    pip install pandas numpy pyarrow
"""

import argparse
import gc
import json
import shutil
import tarfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ================================================================
# 1. Constants and column definitions
# ================================================================

BEIJING_TZ = "Asia/Shanghai"
MIN_FEAT_COUNT = 2

RAW_SAMPLE_STEM = "raw_sample"
AD_FEATURE_STEM = "ad_feature"
USER_PROFILE_STEM = "user_profile"
BEHAVIOR_LOG_STEM = "behavior_log"

RAW_CANONICAL_COLUMNS = ["user_id", "adgroup_id", "time_stamp", "pid", "noclk", "clk"]
RAW_COLUMN_ALIASES = {
    # The official document says user_id / noclk, but the actual common header of the Ali_Display_Ad_Click file is user / nonclk
    "user_id": ["user_id", "user", "userid"],
    "adgroup_id": ["adgroup_id", "ad_group_id"],
    "time_stamp": ["time_stamp", "timestamp", "time"],
    "pid": ["pid"],
    "noclk": ["noclk", "nonclk", "no_clk", "non_clk"],
    "clk": ["clk", "click"],
}
RAW_DTYPES = {
    "user_id": "string",
    "adgroup_id": "string",
    "time_stamp": "int64",
    "pid": "string",
    "noclk": "float32",
    "clk": "float32",
}

BEHAVIOR_CANONICAL_COLUMNS = ["user_id", "time_stamp", "btag", "cate_id", "brand"]
BEHAVIOR_COLUMN_ALIASES = {
    "user_id": ["user", "user_id", "userid"],
    "time_stamp": ["time_stamp", "timestamp", "time"],
    "btag": ["btag", "behavior", "behavior_type"],
    "cate_id": ["cate", "cate_id", "category", "category_id"],
    "brand": ["brand", "brand_id"],
}
BEHAVIOR_DTYPES = {
    "user_id": "string",
    "time_stamp": "int64",
    "btag": "string",
    "cate_id": "string",
    "brand": "string",
}
BEHAVIOR_LABEL_COLUMNS = ["cart", "fav", "buy"]

USER_ID_COL = "user_id"
ITEM_ID_COL = "adgroup_id"
TIMESTAMP_COL = "time_stamp"

LABEL_COLUMNS = ["is_click"] + BEHAVIOR_LABEL_COLUMNS
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
CONTEXT_FEATURES = ["pid", "day_of_week", "is_weekend", "hour"]
ITEM_STATIC_FEATURES = ["cate_id", "campaign_id", "customer_id", "brand", "price_bucket"]

FINAL_COLUMNS = (
    ["user_index", "item_index", "seq_len", "user_id"]
    + USER_STATIC_FEATURES
    + CONTEXT_FEATURES
    + LABEL_COLUMNS
)

PRICE_BINS = [-np.inf, 0, 10, 30, 50, 100, 200, 500, 1000, 5000, np.inf]
PRICE_LABELS = ["<=0", "0-10", "10-30", "30-50", "50-100", "100-200", "200-500", "500-1000", "1000-5000", "5000+"]



# ================================================================
# 2. General Tools
# ================================================================

def _sort_key(v):
    s = str(v)
    try:
        return 0, int(s)
    except Exception:
        return 1, s


def _is_tar_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar")


@contextmanager
def open_csv_source(path: Path):
    """Open a plain CSV or a CSV in a single file tar/tar.gz for use by pandas.read_csv."""
    if _is_tar_path(path):
        tf = tarfile.open(path, "r:*")
        extracted = None
        try:
            members = [m for m in tf.getmembers() if m.isfile()]
            csv_members = [m for m in members if m.name.lower().endswith(".csv")]
            if len(csv_members) == 0:
                raise FileNotFoundError(f"No CSV file in tarball: {path}")
            extracted = tf.extractfile(csv_members[0])
            if extracted is None:
                raise FileNotFoundError(f"Unable to read from tarball: {csv_members[0].name}")
            yield extracted
        finally:
            if extracted is not None:
                extracted.close()
            tf.close()
    else:
        yield path


def find_csv_file(data_dir: Path, stem: str) -> Path:
    candidates = [
        data_dir / f"{stem}.csv",
        data_dir / f"{stem}.csv.gz",
        data_dir / f"{stem}.csv.tar.gz",
        data_dir / f"{stem}.tar.gz",
        data_dir / f"{stem}.tgz",
    ]
    for fp in candidates:
        if fp.exists():
            return fp
    raise FileNotFoundError(
        f"The corresponding file for {stem} cannot be found. Tried:" + ", ".join(str(x) for x in candidates)
    )


def read_csv_file(path: Path, **kwargs) -> pd.DataFrame:
    with open_csv_source(path) as src:
        return pd.read_csv(src, **kwargs)


def iter_csv_chunks(path: Path, **kwargs):
    with open_csv_source(path) as src:
        reader = pd.read_csv(src, **kwargs)
        for chunk in reader:
            yield chunk


def resolve_columns(fp: Path, aliases: dict, dtypes: dict, source_name: str, optional=None):
    """Map actual CSV headers to script-internal canonical column names."""
    optional = set(optional or [])
    header = read_csv_file(fp, nrows=0)
    columns = [str(c).strip() for c in header.columns]
    lower_to_actual = {c.lower(): c for c in columns}

    actual_cols = []
    rename_map = {}
    missing = []
    for canonical, alias_list in aliases.items():
        found = None
        for alias in alias_list:
            found = lower_to_actual.get(alias.lower())
            if found is not None:
                break
        if found is None:
            if canonical in optional:
                continue
            missing.append(canonical)
            continue
        actual_cols.append(found)
        rename_map[found] = canonical

    if missing:
        raise ValueError(
            f"The necessary column is missing in {source_name}: {missing}; The actual header is: {columns};"
            f"Supported alias: {aliases}"
        )

    dtype_map = {
        actual: dtypes[canonical]
        for actual, canonical in rename_map.items()
        if canonical in dtypes
    }
    print(f"  {source_name} column mapping: {rename_map}")
    return actual_cols, rename_map, dtype_map


def resolve_raw_sample_columns(raw_fp: Path):
    return resolve_columns(
        raw_fp,
        RAW_COLUMN_ALIASES,
        RAW_DTYPES,
        source_name="raw_sample",
        optional={"noclk"},
    )


def resolve_behavior_log_columns(behavior_fp: Path):
    return resolve_columns(
        behavior_fp,
        BEHAVIOR_COLUMN_ALIASES,
        BEHAVIOR_DTYPES,
        source_name="behavior_log",
    )


def clean_categorical_series(s: pd.Series) -> pd.Series:
    original_na = s.isna()
    out = s.astype(str).str.strip()
    # If pandas reads the ID column as float, "123.0" will be generated, which is uniformly restored to "123" here.
    out = out.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
    bad = original_na | (out == "") | (out.str.lower().isin(["nan", "none", "null", "<na>"]))
    return out.mask(bad, "__MISSING__")


def build_ordered_vocab(values, start=1) -> dict:
    uniq = list(dict.fromkeys(str(v) for v in values))
    return {v: i for i, v in enumerate(uniq, start=start)}


def build_oov_vocab(values, counter: dict, min_feat_count: int) -> dict:
    vals = {str(v) for v in values}
    vals.add("__MISSING__")
    ordered = ["__MISSING__"] + sorted(vals - {"__MISSING__"}, key=_sort_key)
    vocab = {}
    idx = 1
    for v in ordered:
        cnt = int(counter.get(v, 0))
        if v != "__MISSING__" and 0 < cnt < min_feat_count:
            continue
        vocab[v] = idx
        idx += 1
    return vocab


def build_action_maps():
    """Enumerate click/cart/fav/buy multi-label combination → action code."""
    pat2name = {}
    for p in range(1 << len(LABEL_COLUMNS)):
        if p == 0:
            pat2name[p] = "exposure"
        else:
            pat2name[p] = "|".join(
                col for i, col in enumerate(LABEL_COLUMNS) if p & (1 << i)
            )
    name2code = {name: i + 1 for i, name in enumerate(sorted(set(pat2name.values())))}
    return pat2name, name2code


def fmt_date_int(d: int) -> str:
    return f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"


def detect_timestamp_unit(values) -> str:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(vals) == 0:
        return "s"
    med = float(vals.abs().median())
    return "ms" if med >= 1e12 else "s"


def timestamp_to_datetime(s: pd.Series, unit: str) -> pd.Series:
    return pd.to_datetime(s, unit=unit, utc=True, errors="coerce").dt.tz_convert(BEIJING_TZ)


def build_date_split(sorted_dates, train_ratio, valid_ratio, test_ratio):
    n_days = len(sorted_dates)
    if n_days < 3:
        raise ValueError(f"There are only {n_days} different dates, it takes at least 3 days to split train/valid/test.")

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

    n_train, n_valid, n_test = map(int, counts)
    train_dates = sorted_dates[:n_train]
    valid_dates = sorted_dates[n_train:n_train + n_valid]
    test_dates = sorted_dates[n_train + n_valid:]
    return {
        "n_days": n_days,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "train_dates": train_dates,
        "valid_dates": valid_dates,
        "test_dates": test_dates,
        "valid_start_date": valid_dates[0],
        "test_start_date": test_dates[0],
    }


def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def prepare_output_dir(output_dir: Path, data_dir: Path, overwrite: bool):
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    if data_dir == output_dir or data_dir.is_relative_to(output_dir):
        raise ValueError(
            "output_dir cannot be equal to or contain data_dir, otherwise overwrite will delete the original data. \n"
            f"  data_dir:   {data_dir}\n"
            f"  output_dir: {output_dir}"
        )
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Please use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


# ================================================================
# 3. User/advertising feature loading and encoding
# ================================================================

def load_user_features(data_dir: Path) -> pd.DataFrame:
    fp = find_csv_file(data_dir, USER_PROFILE_STEM)
    print(f"  [Load] {fp.name}")
    uf = read_csv_file(fp)
    uf.rename(
        columns=lambda col: col.strip() if isinstance(col, str) else col,
        inplace=True,
    )
    if "userid" in uf.columns and "user_id" not in uf.columns:
        uf.rename(columns={"userid": "user_id"}, inplace=True)
    if "user_id" not in uf.columns:
        raise ValueError("userid/user_id columns missing in user_profile")

    keep_cols = ["user_id"] + [c for c in USER_STATIC_FEATURES if c in uf.columns]
    uf = uf[keep_cols].copy()
    uf["user_id"] = clean_categorical_series(uf["user_id"])
    for col in USER_STATIC_FEATURES:
        if col not in uf.columns:
            uf[col] = "__MISSING__"
        else:
            uf[col] = clean_categorical_series(uf[col])
    uf.drop_duplicates(subset=["user_id"], keep="last", inplace=True)
    print(f"         {len(uf):,} users")
    return uf[["user_id"] + USER_STATIC_FEATURES]


def bucket_price(price: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(price, errors="coerce")
    bucket = pd.cut(numeric.fillna(-1), bins=PRICE_BINS, labels=PRICE_LABELS).astype(str)
    return bucket.replace({"nan": "__MISSING__"})


def load_ad_features(data_dir: Path) -> pd.DataFrame:
    fp = find_csv_file(data_dir, AD_FEATURE_STEM)
    print(f"  [Load] {fp.name}")
    af = read_csv_file(fp)
    if "adgroup_id" not in af.columns:
        raise ValueError("adgroup_id column missing in ad_feature")

    # The official ad_feature uses customer, which is uniformly named customer_id in the framework.
    if "customer_id" not in af.columns and "customer" in af.columns:
        af.rename(columns={"customer": "customer_id"}, inplace=True)

    keep_cols = ["adgroup_id"] + [c for c in ["cate_id", "campaign_id", "customer_id", "brand", "price"] if c in af.columns]
    af = af[keep_cols].copy()
    af["adgroup_id"] = clean_categorical_series(af["adgroup_id"])

    for col in ["cate_id", "campaign_id", "customer_id", "brand"]:
        if col not in af.columns:
            af[col] = "__MISSING__"
        else:
            af[col] = clean_categorical_series(af[col])

    if "price" in af.columns:
        af["price_bucket"] = bucket_price(af["price"])
        af.drop(columns=["price"], inplace=True)
    else:
        af["price_bucket"] = "__MISSING__"

    af.drop_duplicates(subset=["adgroup_id"], keep="last", inplace=True)
    print(f"         {len(af):,} ads")
    return af[["adgroup_id"] + ITEM_STATIC_FEATURES]


def compute_weighted_feature_counters(
    feat_df: pd.DataFrame,
    key_col: str,
    count_map: dict,
    feature_cols: list,
) -> dict:
    base = pd.DataFrame({key_col: list(count_map.keys()), "__cnt__": list(count_map.values())})
    df = base.merge(feat_df, on=key_col, how="left")
    for col in feature_cols:
        if col not in df.columns:
            df[col] = "__MISSING__"
        df[col] = clean_categorical_series(df[col])

    counters = {col: defaultdict(int) for col in feature_cols}
    for col in feature_cols:
        grouped = df.groupby(col, dropna=False)["__cnt__"].sum()
        for v, cnt in grouped.items():
            counters[col][str(v)] += int(cnt)
    del base, df
    return counters


def encode_user_features(uf: pd.DataFrame, valid_users: set, vocabs: dict) -> pd.DataFrame:
    base = pd.DataFrame({"user_id": list(valid_users)})
    out = base.merge(uf, on="user_id", how="left")
    for col in USER_STATIC_FEATURES:
        out[col] = clean_categorical_series(out[col])
        out[col] = out[col].map(vocabs[col]).fillna(0).astype(np.int32)
    return out[["user_id"] + USER_STATIC_FEATURES]


def encode_ad_features(af: pd.DataFrame, item_idx_map: dict, vocabs: dict) -> pd.DataFrame:
    base = pd.DataFrame({"adgroup_id": list(item_idx_map.keys())})
    out = base.merge(af, on="adgroup_id", how="left")
    out["global_item_index"] = out["adgroup_id"].map(item_idx_map).astype(np.int32)
    for col in ITEM_STATIC_FEATURES:
        out[col] = clean_categorical_series(out[col])
        out[col] = out[col].map(vocabs[col]).fillna(0).astype(np.int32)
    out = out[["global_item_index"] + ITEM_STATIC_FEATURES].drop_duplicates("global_item_index", keep="last")
    return out.set_index("global_item_index").sort_index()


# ================================================================
# 4. Phase 1: raw_sample CSV → user hash partitioning Parquet
# ================================================================

def _preprocess_raw_chunk(
    chunk: pd.DataFrame,
    timestamp_unit: str,
    global_row_offset: int,
    raw_rename_map: dict,
    ad_key_df: pd.DataFrame,
):
    chunk = chunk.rename(columns=raw_rename_map)
    missing = [c for c in RAW_CANONICAL_COLUMNS if c not in chunk.columns and c != "noclk"]
    if missing:
        raise ValueError(f"Missing column in raw_sample: {missing}")
    if "noclk" not in chunk.columns:
        chunk["noclk"] = np.nan

    chunk = chunk[RAW_CANONICAL_COLUMNS].copy()
    chunk["user_id"] = clean_categorical_series(chunk["user_id"])
    chunk["adgroup_id"] = clean_categorical_series(chunk["adgroup_id"])
    chunk["pid"] = clean_categorical_series(chunk["pid"])

    mask = (chunk["user_id"] != "__MISSING__") & (chunk["adgroup_id"] != "__MISSING__")
    chunk = chunk.loc[mask].reset_index(drop=True)
    if len(chunk) == 0:
        return chunk, global_row_offset

    chunk = chunk.merge(ad_key_df, on="adgroup_id", how="left")
    chunk["cate_id"] = clean_categorical_series(chunk["cate_id"])
    chunk["brand"] = clean_categorical_series(chunk["brand"])

    chunk["time_stamp"] = pd.to_numeric(chunk["time_stamp"], errors="coerce").fillna(0).astype(np.int64)
    dt = timestamp_to_datetime(chunk["time_stamp"], timestamp_unit)
    chunk["date"] = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).fillna(0).astype(np.int32)
    chunk["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(np.int8)
    chunk["is_weekend"] = (dt.dt.dayofweek.fillna(0) >= 5).astype(np.int8)
    chunk["hour"] = dt.dt.hour.fillna(0).astype(np.int8)
    del dt

    clk = pd.to_numeric(chunk["clk"], errors="coerce")
    if clk.isna().all():
        noclk = pd.to_numeric(chunk["noclk"], errors="coerce").fillna(1)
        clk = 1 - noclk
    chunk["is_click"] = (clk.fillna(0) > 0).astype(np.float32)
    chunk.drop(columns=["clk", "noclk"], inplace=True)

    n = len(chunk)
    chunk["_row_id"] = np.arange(global_row_offset, global_row_offset + n, dtype=np.int64)
    return chunk, global_row_offset + n


def phase1_partition_to_parquet(
    raw_fp: Path,
    tmp_dir: Path,
    timestamp_unit: str,
    ad_key_df: pd.DataFrame,
    n_parts: int,
    chunk_size: int,
    buffer_flush_size: int,
):
    print(f"\n[Step 1/9] Phase 1: raw_sample → user partition Parquet")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_dir / f"part_{p:03d}").mkdir(exist_ok=True)

    user_counts = defaultdict(int)
    item_counts = defaultdict(int)
    pid_counter = defaultdict(int)
    all_dates = set()

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

    raw_usecols, raw_rename_map, raw_dtype_map = resolve_raw_sample_columns(raw_fp)

    total_rows = 0
    global_row_offset = 0
    reader = iter_csv_chunks(
        raw_fp,
        usecols=raw_usecols,
        dtype=raw_dtype_map,
        chunksize=chunk_size,
    )

    for chunk_idx, raw_chunk in enumerate(reader):
        chunk, global_row_offset = _preprocess_raw_chunk(
            raw_chunk,
            timestamp_unit,
            global_row_offset,
            raw_rename_map,
            ad_key_df,
        )
        del raw_chunk
        if len(chunk) == 0:
            continue

        vc = chunk["user_id"].value_counts(sort=False)
        for uid, cnt in zip(vc.index, vc.values):
            user_counts[str(uid)] += int(cnt)

        ic = chunk["adgroup_id"].value_counts(sort=False)
        for iid, cnt in zip(ic.index, ic.values):
            item_counts[str(iid)] += int(cnt)

        pc = chunk["pid"].value_counts(sort=False)
        for pid, cnt in zip(pc.index, pc.values):
            pid_counter[str(pid)] += int(cnt)

        all_dates.update(int(x) for x in chunk["date"].unique().tolist() if int(x) > 0)

        part_arr = pd.util.hash_pandas_object(chunk["user_id"], index=False).to_numpy() % n_parts
        for pid in range(n_parts):
            mask = part_arr == pid
            n_match = int(mask.sum())
            if n_match > 0:
                buffers[pid].append(chunk.loc[mask].copy())
                buf_sizes[pid] += n_match
                if buf_sizes[pid] >= buffer_flush_size:
                    flush(pid)

        total_rows += len(chunk)
        if (chunk_idx + 1) % 10 == 0:
            print(f"  chunk {chunk_idx + 1}: accumulated {total_rows:,} rows")
        del chunk, part_arr
        gc.collect()

    for pid in range(n_parts):
        flush(pid)

    print(
        f"  Done: {total_rows:,} row, {len(user_counts):,} user,"
        f"{len(item_counts):,} ads, {len(all_dates)} dates"
    )
    return dict(user_counts), dict(item_counts), dict(pid_counter), all_dates, total_rows


# ================================================================
# 5. behavior_log → user hash partition Parquet (for multi-task labels)
# ================================================================

def _preprocess_behavior_chunk(chunk: pd.DataFrame, behavior_rename_map: dict):
    chunk = chunk.rename(columns=behavior_rename_map)
    missing = [c for c in BEHAVIOR_CANONICAL_COLUMNS if c not in chunk.columns]
    if missing:
        raise ValueError(f"Missing column in behavior_log: {missing}")

    chunk = chunk[BEHAVIOR_CANONICAL_COLUMNS].copy()
    chunk["user_id"] = clean_categorical_series(chunk["user_id"])
    chunk["btag"] = clean_categorical_series(chunk["btag"]).str.lower()
    chunk["cate_id"] = clean_categorical_series(chunk["cate_id"])
    chunk["brand"] = clean_categorical_series(chunk["brand"])
    chunk["time_stamp"] = pd.to_numeric(chunk["time_stamp"], errors="coerce").fillna(0).astype(np.int64)

    chunk = chunk[
        (chunk["user_id"] != "__MISSING__")
        & (chunk["cate_id"] != "__MISSING__")
        & (chunk["brand"] != "__MISSING__")
        & (chunk["btag"].isin(BEHAVIOR_LABEL_COLUMNS))
    ].reset_index(drop=True)
    if len(chunk) == 0:
        return chunk

    for col in BEHAVIOR_LABEL_COLUMNS:
        chunk[col] = (chunk["btag"] == col).astype(np.float32)

    key_cols = ["user_id", "time_stamp", "cate_id", "brand"]
    chunk = chunk.groupby(key_cols, as_index=False, sort=False)[BEHAVIOR_LABEL_COLUMNS].max()
    return chunk


def phase_behavior_partition_to_parquet(
    behavior_fp: Path,
    tmp_behavior_dir: Path,
    raw_users: set,
    n_parts: int,
    chunk_size: int,
    buffer_flush_size: int,
):
    print("\n[Step 2/9] behavior_log → User partition Parquet (generate cart/fav/buy tag)")
    tmp_behavior_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_behavior_dir / f"part_{p:03d}").mkdir(exist_ok=True)

    behavior_usecols, behavior_rename_map, behavior_dtype_map = resolve_behavior_log_columns(behavior_fp)

    buffers = {p: [] for p in range(n_parts)}
    buf_sizes = {p: 0 for p in range(n_parts)}
    file_counts = {p: 0 for p in range(n_parts)}
    total_rows = 0
    kept_rows = 0
    label_counts = defaultdict(int)

    def flush(pid):
        if not buffers[pid]:
            return
        out = pd.concat(buffers[pid], ignore_index=True)
        fp = tmp_behavior_dir / f"part_{pid:03d}" / f"c_{file_counts[pid]:04d}.parquet"
        out.to_parquet(fp, index=False, engine="pyarrow")
        file_counts[pid] += 1
        buffers[pid] = []
        buf_sizes[pid] = 0
        del out

    reader = iter_csv_chunks(
        behavior_fp,
        usecols=behavior_usecols,
        dtype=behavior_dtype_map,
        chunksize=chunk_size,
    )
    for chunk_idx, raw_chunk in enumerate(reader):
        total_rows += len(raw_chunk)
        chunk = _preprocess_behavior_chunk(raw_chunk, behavior_rename_map)
        del raw_chunk
        if len(chunk) == 0:
            continue

        chunk = chunk[chunk["user_id"].isin(raw_users)].reset_index(drop=True)
        if len(chunk) == 0:
            continue

        kept_rows += len(chunk)
        for col in BEHAVIOR_LABEL_COLUMNS:
            label_counts[col] += int(chunk[col].sum())

        part_arr = pd.util.hash_pandas_object(chunk["user_id"], index=False).to_numpy() % n_parts
        for pid in range(n_parts):
            mask = part_arr == pid
            n_match = int(mask.sum())
            if n_match > 0:
                buffers[pid].append(chunk.loc[mask].copy())
                buf_sizes[pid] += n_match
                if buf_sizes[pid] >= buffer_flush_size:
                    flush(pid)

        if (chunk_idx + 1) % 20 == 0:
            print(f"  behavior chunk {chunk_idx + 1}: raw={total_rows:,}, kept={kept_rows:,}")
        del chunk, part_arr
        gc.collect()

    for pid in range(n_parts):
        flush(pid)

    print(f"  behavior_log completed: raw={total_rows:,}, kept={kept_rows:,}, labels={dict(label_counts)}")
    return {
        "raw_rows": int(total_rows),
        "kept_rows": int(kept_rows),
        "label_counts": {k: int(v) for k, v in label_counts.items()},
    }


def load_behavior_partition(tmp_behavior_dir: Path, pid: int):
    part_dir = tmp_behavior_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(fp) for fp in files]
    if not dfs:
        return None
    behavior = pd.concat(dfs, ignore_index=True)
    del dfs
    if len(behavior) == 0:
        return None
    key_cols = ["user_id", "time_stamp", "cate_id", "brand"]
    behavior = behavior.groupby(key_cols, as_index=False, sort=False)[BEHAVIOR_LABEL_COLUMNS].max()
    return behavior


def attach_behavior_labels_by_forward_window(
    df: pd.DataFrame,
    behavior: pd.DataFrame,
    behavior_label_window: int,
) -> pd.DataFrame:
    """Press user/cate/brand + future time window to convert behavior_log behavior into exposure sample label.

    Previously, user_id + time_stamp + cate_id + brand was used for exact matching, which has a great impact on ad exposure and
    Two independent logs of shopping behavior are too strict and will rarely hit the target in practice. Here it is changed to: for each exposure sample,
    If the same user is under the same cate/brand, it will occur within behavior_label_window after the exposure time.
    cart/fav/buy, corresponding to label=1.
    """
    for col in BEHAVIOR_LABEL_COLUMNS:
        df[col] = 0.0

    if behavior is None or len(behavior) == 0 or behavior_label_window <= 0:
        return df

    key_cols = ["user_id", "cate_id", "brand"]
    df["__label_row_id__"] = np.arange(len(df), dtype=np.int64)
    raw_groups = df.groupby(key_cols, sort=False).indices

    for label in BEHAVIOR_LABEL_COLUMNS:
        b = behavior[pd.to_numeric(behavior[label], errors="coerce").fillna(0) > 0]
        if len(b) == 0:
            continue

        for key, g in b.groupby(key_cols, sort=False):
            raw_idx = raw_groups.get(key)
            if raw_idx is None or len(raw_idx) == 0:
                continue

            behavior_times = np.sort(
                pd.to_numeric(g["time_stamp"], errors="coerce")
                .dropna()
                .astype(np.int64)
                .unique()
            )
            if len(behavior_times) == 0:
                continue

            raw_times = df.iloc[raw_idx]["time_stamp"].to_numpy(dtype=np.int64, copy=False)
            pos = np.searchsorted(behavior_times, raw_times, side="left")
            valid = pos < len(behavior_times)
            if not valid.any():
                continue

            matched_pos = pos[valid]
            matched = (behavior_times[matched_pos] - raw_times[valid]) <= behavior_label_window
            if matched.any():
                df.loc[df.index[raw_idx[valid][matched]], label] = 1.0

        del b
        gc.collect()

    df.drop(columns=["__label_row_id__"], inplace=True)
    return df


# ================================================================
# 6. Phase 2: Partition encoding + segmentation by date
# ================================================================

def process_partition(
    tmp_dir: Path,
    tmp_behavior_dir: Path,
    pid: int,
    valid_users: set,
    user_idx_map: dict,
    item_idx_map: dict,
    uf_enc: pd.DataFrame,
    vocabs: dict,
    pat2name: dict,
    action_name2code: dict,
    valid_start_date: int,
    test_start_date: int,
    use_behavior_labels: bool = True,
    behavior_label_window: int = 86400,
):
    part_dir = tmp_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None

    dfs = []
    for fp in files:
        d = pd.read_parquet(fp)
        d = d[d["user_id"].isin(valid_users)]
        if len(d) > 0:
            dfs.append(d)
        del d
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    if use_behavior_labels:
        behavior = load_behavior_partition(tmp_behavior_dir, pid)
        df = attach_behavior_labels_by_forward_window(
            df,
            behavior,
            behavior_label_window=behavior_label_window,
        )
        del behavior
    else:
        for col in BEHAVIOR_LABEL_COLUMNS:
            df[col] = 0.0

    df = df.merge(uf_enc, on="user_id", how="left")
    for col in USER_STATIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)

    df.sort_values(["user_id", "time_stamp", "_row_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["user_index"] = df["user_id"].map(user_idx_map).astype(np.int32)
    df["item_index"] = df["adgroup_id"].map(item_idx_map).fillna(0).astype(np.int32)
    df["user_id"] = (df["user_index"] + 1).astype(np.int32)

    df["pid"] = df["pid"].map(vocabs["pid"]).fillna(0).astype(np.int32)
    df["day_of_week"] = (df["day_of_week"].astype(np.int32) + 1).astype(np.int32)
    df["is_weekend"] = (df["is_weekend"].astype(np.int32) + 1).astype(np.int32)
    df["hour"] = (df["hour"].astype(np.int32) + 1).astype(np.int32)
    for col in LABEL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)

    binary = df[LABEL_COLUMNS].values.astype(np.int8)
    pattern = np.zeros(len(df), dtype=np.int32)
    for i in range(len(LABEL_COLUMNS)):
        pattern += binary[:, i].astype(np.int32) << i
    df["action"] = pd.Series(pattern, index=df.index).map(pat2name).map(action_name2code).astype(np.int32)
    del binary, pattern
    df["time_stamp"] = pd.to_numeric(df["time_stamp"], errors="coerce").fillna(0).astype(np.int64)
    df["seq_len"] = df.groupby("user_index", sort=False).cumcount().astype(np.int32)

    user_info_rows = []
    for uidx, gdf in df.groupby("user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(uidx),
                "full_item_seq": gdf["item_index"].astype(int).tolist(),
                "full_action_seq": gdf["action"].astype(int).tolist(),
                "full_timestamp_seq": gdf["time_stamp"].astype(np.int64).tolist(),
                "full_day_of_week_seq": gdf["day_of_week"].astype(int).tolist(),
                "full_is_weekend_seq": gdf["is_weekend"].astype(int).tolist(),
                "full_hour_seq": gdf["hour"].astype(int).tolist(),
                "full_pid_seq": gdf["pid"].astype(int).tolist(),
            }
        )

    date_col = df["date"]
    train_df = df[date_col < valid_start_date]
    valid_df = df[(date_col >= valid_start_date) & (date_col < test_start_date)]
    test_df = df[date_col >= test_start_date]

    def _select_final(sdf):
        if len(sdf) == 0:
            return pd.DataFrame(columns=FINAL_COLUMNS)
        return sdf[FINAL_COLUMNS].reset_index(drop=True)

    result = {
        "train": _select_final(train_df),
        "valid": _select_final(valid_df),
        "test": _select_final(test_df),
        "user_info": user_info_rows,
    }
    del df, train_df, valid_df, test_df
    gc.collect()
    return result


# ================================================================
# 6. blocked output management
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

    def append_partition(self, pid: int, split_df: pd.DataFrame, user_info_rows: list):
        if len(split_df) == 0:
            return None
        bid = self.choose_block(len(split_df))
        self.partition_to_block[int(pid)] = int(bid)

        ui_lookup = {r["user_index"]: r for r in user_info_rows}
        active_users = set(pd.unique(split_df["user_index"].astype(np.int64)).tolist())

        out_ui_rows = []
        for g_user in active_users:
            if g_user not in ui_lookup:
                raise ValueError(f"[{self.split_name}] partition={pid} user_index={g_user} missing user_info")
            row = ui_lookup[g_user]
            l_user = self._get_or_add_user_local(bid, int(g_user))

            g_item_seq = row.get("full_item_seq", [])
            if not isinstance(g_item_seq, (list, tuple, np.ndarray)):
                g_item_seq = []
            l_item_seq = [self._get_or_add_item_local(bid, safe_int(x, 0)) for x in g_item_seq]

            g_action_seq = row.get("full_action_seq", [])
            if not isinstance(g_action_seq, (list, tuple, np.ndarray)):
                g_action_seq = []

            g_time_seq = row.get("full_timestamp_seq", [])
            if not isinstance(g_time_seq, (list, tuple, np.ndarray)):
                g_time_seq = []

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
        out_ui_df = pd.DataFrame(out_ui_rows)

        data_fp = self.data_dir / f"part-{bid:05d}.parquet"
        ui_fp = self.user_info_dir / f"part-{bid:05d}.parquet"
        self._write_table(self.data_writers, bid, data_fp, out_df)
        self._write_table(self.user_writers, bid, ui_fp, out_ui_df)

        self.block_rows[bid] += len(out_df)
        self.block_user_rows[bid] += len(out_ui_df)
        del out_df, out_ui_df, out_ui_rows
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

            item_info_df = pd.DataFrame(out)
            item_info_df.to_parquet(self.item_info_dir / f"part-{bid:05d}.parquet", index=False, engine="pyarrow")
            del item_info_df
            gc.collect()

    def build_manifest(self):
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
                    "source_partitions": [int(pid) for pid, b in self.partition_to_block.items() if b == bid],
                }
            )
        return {
            "split": self.split_name,
            "num_blocks_configured": int(self.num_blocks),
            "num_blocks_written": int(len(blocks)),
            "blocks": blocks,
        }


# ================================================================
# 7. Main process
# ================================================================

def preprocess_and_split(
    data_dir: str = "./Taobao",
    output_dir: str = "./data/Taobao_Ad_Click_Action",
    min_user_interactions: int = 10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_user_parts: int = 64,
    chunk_size: int = 4_000_000,
    buffer_flush_size: int = 1_000_000,
    train_blocks: int = 32,
    valid_blocks: int = 8,
    test_blocks: int = 8,
    min_feat_count: int = MIN_FEAT_COUNT,
    timestamp_unit: str = "auto",
    use_behavior_labels: bool = True,
    behavior_label_window_seconds: int = 86400,
    overwrite: bool = False,
):
    data_dir = Path(data_dir).resolve()
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

    raw_fp = find_csv_file(data_dir, RAW_SAMPLE_STEM)
    behavior_fp = find_csv_file(data_dir, BEHAVIOR_LOG_STEM) if use_behavior_labels else None

    prepare_output_dir(output_dir, data_dir, overwrite)
    tmp_dir = output_dir / "_tmp_partitions"
    tmp_behavior_dir = output_dir / "_tmp_behavior_partitions"

    _, raw_rename_for_ts, _ = resolve_raw_sample_columns(raw_fp)
    time_actual_col = next(k for k, v in raw_rename_for_ts.items() if v == "time_stamp")
    if timestamp_unit == "auto":
        sample = read_csv_file(raw_fp, usecols=[time_actual_col], nrows=1000)
        timestamp_unit = detect_timestamp_unit(sample[time_actual_col])
        del sample
    if timestamp_unit not in {"s", "ms"}:
        raise ValueError("timestamp_unit can only be auto/s/ms")
    behavior_label_window = int(behavior_label_window_seconds) * (1000 if timestamp_unit == "ms" else 1)
    print(
        f"[Config] timestamp_unit={timestamp_unit}, timezone={BEIJING_TZ}, "
        f"use_behavior_labels={use_behavior_labels}, "
        f"behavior_label_window_seconds={behavior_label_window_seconds}"
    )

    print("\n[Step 0/9] Load ad_feature (for raw_sample and behavior_log alignment)")
    af = load_ad_features(data_dir)
    ad_key_df = af[["adgroup_id", "cate_id", "brand"]].copy()

    user_counts, item_counts, pid_counter, all_dates, total_rows_phase1 = phase1_partition_to_parquet(
        raw_fp=raw_fp,
        tmp_dir=tmp_dir,
        timestamp_unit=timestamp_unit,
        ad_key_df=ad_key_df,
        n_parts=n_user_parts,
        chunk_size=chunk_size,
        buffer_flush_size=buffer_flush_size,
    )

    behavior_stats = None
    if use_behavior_labels:
        behavior_stats = phase_behavior_partition_to_parquet(
            behavior_fp=behavior_fp,
            tmp_behavior_dir=tmp_behavior_dir,
            raw_users=set(user_counts.keys()),
            n_parts=n_user_parts,
            chunk_size=chunk_size,
            buffer_flush_size=buffer_flush_size,
        )

    print("\n[Step 3/9] Load user_profile and build OOV vocab")
    uf = load_user_features(data_dir)

    valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    if len(valid_users) == 0:
        raise ValueError("The number of effective users after filtering is 0, please adjust --min_user_interactions to a smaller value.")
    print(f"  Valid users: {len(valid_users):,} / {len(user_counts):,}")

    user_feat_counters = compute_weighted_feature_counters(uf, "user_id", user_counts, USER_STATIC_FEATURES)
    item_feat_counters = compute_weighted_feature_counters(af, "adgroup_id", item_counts, ITEM_STATIC_FEATURES)

    vocabs = {}
    for col in USER_STATIC_FEATURES:
        vocabs[col] = build_oov_vocab(uf[col].unique(), user_feat_counters[col], min_feat_count)
    for col in ITEM_STATIC_FEATURES:
        vocabs[col] = build_oov_vocab(af[col].unique(), item_feat_counters[col], min_feat_count)
    vocabs["pid"] = build_ordered_vocab(["__MISSING__"] + sorted(pid_counter.keys(), key=_sort_key), start=1)

    print(f"  OOV filter threshold: {min_feat_count}")
    for col in USER_STATIC_FEATURES + ITEM_STATIC_FEATURES + ["pid"]:
        print(f"  {col}: vocab_size={len(vocabs[col]) + 1}")

    print("\n[Step 4/9] Determine train/valid/test based on date ratio")
    sorted_dates = sorted(all_dates)
    split_info = build_date_split(sorted_dates, train_ratio, valid_ratio, test_ratio)
    train_dates = split_info["train_dates"]
    valid_dates = split_info["valid_dates"]
    test_dates = split_info["test_dates"]
    valid_start_date = split_info["valid_start_date"]
    test_start_date = split_info["test_start_date"]
    print(f"  Total days: {split_info['n_days']} ({fmt_date_int(sorted_dates[0])} ~ {fmt_date_int(sorted_dates[-1])})")
    print(f"  train: {split_info['n_train']} day {fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}")
    print(f"  valid: {split_info['n_valid']} days {fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}")
    print(f"  test: {split_info['n_test']} days {fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}")

    print("\n[Step 5/9] Construct global ID mapping and encoded static features")
    sorted_users = sorted(valid_users, key=_sort_key)
    user_idx_map = {u: i for i, u in enumerate(sorted_users)}
    sorted_items = sorted(item_counts.keys(), key=_sort_key)
    item_idx_map = {it: i + 1 for i, it in enumerate(sorted_items)}
    print(f"  users={len(user_idx_map):,}, ads={len(item_idx_map):,}")

    uf_enc = encode_user_features(uf, valid_users, vocabs)
    global_item_lookup = encode_ad_features(af, item_idx_map, vocabs)
    pat2name, action_name2code = build_action_maps()

    vocab_size = {
        "user_index": len(user_idx_map),
        "item_index": len(item_idx_map) + 1,
        "user_id": len(user_idx_map) + 1,
        "item_id": len(item_idx_map) + 1,
        "action": len(action_name2code) + 1,
        "timestamp": 0,
        "day_of_week": 8,
        "is_weekend": 3,
        "hour": 25,
    }
    for col in USER_STATIC_FEATURES + ITEM_STATIC_FEATURES + ["pid"]:
        vocab_size[col] = len(vocabs[col]) + 1

    del uf, af, sorted_users, sorted_items, user_feat_counters, item_feat_counters
    gc.collect()

    print("\n[Step 6/9] Encode partition by partition + split by date + write block data/user_info")
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
            tmp_behavior_dir=tmp_behavior_dir,
            pid=pid,
            valid_users=valid_users,
            user_idx_map=user_idx_map,
            item_idx_map=item_idx_map,
            uf_enc=uf_enc,
            vocabs=vocabs,
            pat2name=pat2name,
            action_name2code=action_name2code,
            valid_start_date=valid_start_date,
            test_start_date=test_start_date,
            use_behavior_labels=use_behavior_labels,
            behavior_label_window=behavior_label_window,
        )
        if result is None:
            continue

        user_info_rows = result["user_info"]
        for r in user_info_rows:
            max_seq = max(max_seq, len(r["full_item_seq"]))

        for split_name in ["train", "valid", "test"]:
            sdf = result[split_name]
            if len(sdf) == 0:
                continue
            managers[split_name].append_partition(pid, sdf, user_info_rows)
            sample_counts[split_name] += len(sdf)

        done = sum(sample_counts.values())
        print(f"  partition {pid + 1:3d}/{n_user_parts}: accumulated {done:,} rows")
        del result
        gc.collect()

    for mgr in managers.values():
        mgr.close_writers()

    total = sum(sample_counts.values())
    if total == 0:
        raise ValueError("The number of samples after processing is 0, please check the original data or filter threshold.")
    print(f"  train={sample_counts['train']:,} valid={sample_counts['valid']:,} test={sample_counts['test']:,} total={total:,}")

    print("\n[Step 7/9] Build item_info for each block")
    for split_name in ["train", "valid", "test"]:
        print(f"  [Build item_info] {split_name}")
        managers[split_name].write_item_info_blocks(global_item_lookup)
        gc.collect()

    print("\n[Step 8/9] Save meta_data + block_manifest")
    meta = {
        "dataset": "Ali_Display_Ad_Click / Taobao",
        "sample_size": {
            "total": int(total),
            "train": int(sample_counts["train"]),
            "valid": int(sample_counts["valid"]),
            "test": int(sample_counts["test"]),
            "phase1_total_interactions": int(total_rows_phase1),
        },
        "split_by_date": {
            "timezone": BEIJING_TZ,
            "timestamp_unit": timestamp_unit,
            "train_days": split_info["n_train"],
            "train_range": f"{fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}",
            "valid_days": split_info["n_valid"],
            "valid_range": f"{fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}",
            "test_days": split_info["n_test"],
            "test_range": f"{fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}",
            "rule": "Sort by raw_sample date and split according to train/valid/test ratio; the official 8-day data defaults to about 6/1/1.",
        },
        "behavior_log_usage": {
            "used": bool(use_behavior_labels),
            "label_columns": BEHAVIOR_LABEL_COLUMNS,
            "stats": behavior_stats,
            "match_key": ["user_id", "cate_id", "brand"],
            "behavior_label_window_seconds": int(behavior_label_window_seconds),
            "rule": "raw_sample is first mapped to ad_feature.cate_id/brand through adgroup_id; for each exposure sample, if the same user_id + cate_id + brand appears within behavior_label_window_seconds seconds after exposure time_stamp, the corresponding btag will be set to 1, otherwise it will be 0.",
            "note": "behavior_log does not contain adgroup_id, so this is category/brand granularity, behavioral supervision within the post-exposure time window, not a direct label at the advertising ID granularity.",
        },
        "user_filtering": {
            "min_user_interactions": int(min_user_interactions),
            "valid_users": int(len(valid_users)),
            "dropped_users": int(len(user_counts) - len(valid_users)),
        },
        "oov_filter": {
            "min_feat_count": int(min_feat_count),
            "applied_to": USER_STATIC_FEATURES + ITEM_STATIC_FEATURES,
            "rule": "The uniform mapping of feature value occurrences < min_feat_count is 0 (unknown/padding)",
            "freq_source": "Based on the number of occurrences of user/item in raw_sample * static feature value statistics",
        },
        "blocked_layout": {
            "n_user_parts": int(n_user_parts),
            "train_blocks": int(train_blocks),
            "valid_blocks": int(valid_blocks),
            "test_blocks": int(test_blocks),
            "train": {"data_dir": "train/data", "user_info_dir": "train/user_info", "item_info_dir": "train/item_info"},
            "valid": {"data_dir": "valid/data", "user_info_dir": "valid/user_info", "item_info_dir": "valid/item_info"},
            "test": {"data_dir": "test/data", "user_info_dir": "test/user_info", "item_info_dir": "test/item_info"},
            "block_pair_rule": "Under the same split, data/user_info/item_info uses the same part-xxxxx number for paired reading.",
            "local_index_rule": {
                "user_index": "block-local dense index, starts from 0",
                "item_index": "block-local dense index, 0 reserved for padding",
                "user_id": "global feature id, consistent across blocks",
                "item_id": "global feature id, consistent across blocks",
            },
        },
        "vocab_size": {k: int(v) for k, v in vocab_size.items()},
        "label": LABEL_COLUMNS,
        "action_vocab": {k: int(v) for k, v in action_name2code.items()},
        "action_vocab_desc": "The encoded multi-task action vocabulary is used by the dataloader to construct task-specific token masks based on full_action_seq.",
        "user_info_schema": {
            "fields": [
                "user_index",
                "full_item_seq",
                "full_action_seq",
                "full_timestamp_seq",
                "full_day_of_week_seq",
                "full_is_weekend_seq",
                "full_hour_seq",
            ],
            "full_timestamp_seq_desc": "chronological sequence of raw_sample time_stamp",
            "desc": "The item index in user_index / full_item_seq is a block-local index; full_action_seq / full_timestamp_seq / full_*_seq context columns are global time order sequences.",
        },
        "item_info_schema": {
            "fields": ["item_index", "item_id"] + ITEM_STATIC_FEATURES,
            "desc": "item_index is block-local index; item_id is global item feature id.",
        },
        "feature_schema": {
            "user_static_features": USER_STATIC_FEATURES,
            "context_features": CONTEXT_FEATURES,
            "item_static_features": ITEM_STATIC_FEATURES,
            "label_columns": LABEL_COLUMNS,
        },
        "max_len": {
            "full_item_seq": int(max_seq),
            "full_action_seq": int(max_seq),
            "full_timestamp_seq": int(max_seq),
            "full_day_of_week_seq": int(max_seq),
            "full_is_weekend_seq": int(max_seq),
            "full_hour_seq": int(max_seq),
        },
    }

    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    print("  [Saved] meta_data.json")

    block_manifest = {
        "train": managers["train"].build_manifest(),
        "valid": managers["valid"].build_manifest(),
        "test": managers["test"].build_manifest(),
    }
    with open(output_dir / "block_manifest.json", "w", encoding="utf-8") as f:
        json.dump(block_manifest, f, ensure_ascii=False, indent=4)
    print("  [Saved] block_manifest.json")

    print("\n[Step 9/9] Clean up temporary files")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(tmp_behavior_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("  Taobao Preprocess Done (multi-task label + split by date ratio + blocked output + OOV filtering)")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(json.dumps(meta, ensure_ascii=False, indent=4))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Taobao Ali_Display_Ad_Click into blocked seq-action parquet data."
    )
    parser.add_argument("--data_dir", type=str, default="./Taobao")
    parser.add_argument("--output_dir", type=str, default="./Taobao_Action")
    parser.add_argument("--min_user_interactions", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--n_user_parts", type=int, default=32, help="The number of temporary partitions according to user hash; behavior_log is very large and can be adjusted to 128/256 if there is insufficient memory.")
    parser.add_argument("--chunk_size", type=int, default=4_000_000, help="Number of rows in a single chunk when reading raw_sample")
    parser.add_argument("--buffer_flush_size", type=int, default=1_000_000, help="How many rows are cached in the temporary partition before being flushed to disk?")
    parser.add_argument("--train_blocks", type=int, default=32)
    parser.add_argument("--valid_blocks", type=int, default=8)
    parser.add_argument("--test_blocks", type=int, default=8)
    parser.add_argument("--min_feat_count", type=int, default=MIN_FEAT_COUNT)
    parser.add_argument("--timestamp_unit", type=str, default="auto", choices=["auto", "s", "ms"])
    parser.add_argument("--behavior_label_window_seconds", type=int, default=86400,
                        help="How many seconds after exposure the behavior of the same user/cate/brand will be marked as cart/fav/buy positive samples")
    parser.add_argument("--disable_behavior_labels", action="store_true", default=False,
                        help="Disable behavior_log to generate cart/fav/buy tags, leaving only is_click")
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
        n_user_parts=args.n_user_parts,
        chunk_size=args.chunk_size,
        buffer_flush_size=args.buffer_flush_size,
        train_blocks=args.train_blocks,
        valid_blocks=args.valid_blocks,
        test_blocks=args.test_blocks,
        min_feat_count=args.min_feat_count,
        timestamp_unit=args.timestamp_unit,
        use_behavior_labels=not args.disable_behavior_labels,
        behavior_label_window_seconds=args.behavior_label_window_seconds,
        overwrite=args.overwrite,
    )
