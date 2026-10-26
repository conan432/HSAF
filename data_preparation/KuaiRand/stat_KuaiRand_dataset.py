#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Statistics KuaiRand-27K blocked data set information."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset_stats_utils import DatasetSpec, run_cli


ROOT = Path(__file__).resolve().parents[1]

USER_STATIC_FEATURES = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]

ITEM_STATIC_FEATURES = ["video_type", "primary_tag", "music_type", "duration_bucket"]
CONTEXT_FEATURES = ["tab", "day_of_week", "is_weekend", "hour"]

FEATURE_FIELDS = (
    ["user_index", "item_index", "seq_len", "user_id", "item_id"]
    + ITEM_STATIC_FEATURES
    + USER_STATIC_FEATURES
    + CONTEXT_FEATURES
    + ["action"]
)

LABEL_COLUMNS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view",
]


if __name__ == "__main__":
    run_cli(
        DatasetSpec(
            name="KuaiRand_Video_Action",
            default_output_dir=ROOT / "KuaiRand_Video_Action",
            feature_fields=FEATURE_FIELDS,
            label_columns=LABEL_COLUMNS,
        )
    )
