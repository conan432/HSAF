#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Statistics MerRec_Action blocked data set information."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset_stats_utils import DatasetSpec, run_cli


ROOT = Path(__file__).resolve().parents[1]

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

CONTEXT_FEATURES = ["session_id", "day_of_week", "is_weekend", "hour"]

FEATURE_FIELDS = (
    ["user_index", "item_index", "seq_len", "user_id", "item_id"]
    + ITEM_STATIC_FEATURES
    + CONTEXT_FEATURES
    + ["action"]
)

LABEL_COLUMNS = ["Like", "Cart", "Offer", "Checkout", "Purchase"]


if __name__ == "__main__":
    run_cli(
        DatasetSpec(
            name="MerRec_Action",
            default_output_dir=ROOT / "MerRec_Action",
            feature_fields=FEATURE_FIELDS,
            label_columns=LABEL_COLUMNS,
        )
    )
