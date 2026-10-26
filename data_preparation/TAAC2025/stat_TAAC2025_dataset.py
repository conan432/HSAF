#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Statistics TAAC2025 / TencentGR blocked data set information."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset_stats_utils import DatasetSpec, run_cli


ROOT = Path(__file__).resolve().parents[1]

USER_SCALAR_FEATURES = ["103", "104", "105", "109"]
USER_LIST_FEATURES = ["106", "107", "108", "110"]
ITEM_STATIC_FEATURES = [
    "100", "101", "102", "112", "114", "115", "116",
    "117", "118", "119", "120", "121", "122",
]
CONTEXT_FEATURES = ["day_of_week", "is_weekend", "hour"]

FEATURE_FIELDS = (
    ["user_index", "item_index", "seq_len", "user_id", "item_id"]
    + ITEM_STATIC_FEATURES
    + USER_SCALAR_FEATURES
    + USER_LIST_FEATURES
    + CONTEXT_FEATURES
    + ["action"]
)

LABEL_COLUMNS = ["is_click", "is_conversion"]


if __name__ == "__main__":
    run_cli(
        DatasetSpec(
            name="TencentGR_10M_Action",
            default_output_dir=ROOT / "TencentGR_10M_Action_Blocked",
            feature_fields=FEATURE_FIELDS,
            label_columns=LABEL_COLUMNS,
        )
    )
