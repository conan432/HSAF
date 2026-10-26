# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the LICENSE file at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import logging

from .unirank_dataloader import SpawnUniRankDataloader, UniRankDataloader


_DATA_LOADER_REGISTRY = {
    "default": UniRankDataloader,
    "unirank": UniRankDataloader,
    "unirank_dataloader": UniRankDataloader,
    "spawn": SpawnUniRankDataloader,
    "spawn_unirank": SpawnUniRankDataloader,
    "spawn_unirank_dataloader": SpawnUniRankDataloader,
}


def _resolve_data_loader(loader):
    """Resolve a loader specified either as a Python class or YAML string."""
    if isinstance(loader, str):
        key = loader.strip().lower().replace(" ", "_")
        if key not in _DATA_LOADER_REGISTRY:
            raise ValueError(
                f"Unknown data_loader={loader!r}; choose from "
                f"{sorted(_DATA_LOADER_REGISTRY)}"
            )
        return _DATA_LOADER_REGISTRY[key]
    return loader


class RankDataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None,
                 valid_data=None, test_data=None, batch_size=32,
                 shuffle=True, **kwargs):
        logging.info("Loading datasets...")
        train_gen = None
        valid_gen = None
        test_gen = None

        data_loader = kwargs.pop("data_loader", UniRankDataloader)
        train_data_loader = kwargs.pop("train_data_loader", data_loader)
        eval_data_loader = kwargs.pop("eval_data_loader", data_loader)
        self.stage = stage

        if stage in ["both", "train"]:
            train_gen = _resolve_data_loader(train_data_loader)(
                feature_map,
                train_data,
                split="train",
                batch_size=batch_size,
                shuffle=shuffle,
                **kwargs,
            )
            logging.info(
                "Train samples: total/{:d}, blocks/{:d}".format(
                    train_gen.num_samples, train_gen.num_blocks
                )
            )
            if valid_data:
                valid_gen = _resolve_data_loader(eval_data_loader)(
                    feature_map,
                    valid_data,
                    split="valid",
                    batch_size=batch_size,
                    shuffle=False,
                    **kwargs,
                )
                logging.info(
                    "Validation samples: total/{:d}, blocks/{:d}".format(
                        valid_gen.num_samples, valid_gen.num_blocks
                    )
                )

        if stage in ["both", "test"]:
            if test_data:
                test_gen = _resolve_data_loader(eval_data_loader)(
                    feature_map,
                    test_data,
                    split="test",
                    batch_size=batch_size,
                    shuffle=False,
                    **kwargs,
                )
                logging.info(
                    "Test samples: total/{:d}, blocks/{:d}".format(
                        test_gen.num_samples, test_gen.num_blocks
                    )
                )

        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.test_gen = test_gen

    def make_iterator(self):
        if self.stage == "train":
            logging.info("Loading train and validation data done.")
            return self.train_gen, self.valid_gen
        elif self.stage == "test":
            logging.info("Loading test data done.")
            return self.test_gen
        else:
            logging.info("Loading data done.")
            return self.train_gen, self.valid_gen, self.test_gen
