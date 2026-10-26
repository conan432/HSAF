# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2025. FuxiCTR Authors. All rights reserved.
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

import json
import math
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from unirank.pytorch.torch_utils import build_batch_dict_from_tensor
from unirank.utils import (
    build_part_file_map,
    dataframe_to_darray,
    estimate_parquet_block_cost,
    find_meta_data_json,
    get_parquet_schema_names,
    resolve_side_info_path,
)


def _single_threaded_worker_init(_worker_id):
    """Avoid PyTorch thread oversubscription inside parquet workers."""
    torch.set_num_threads(1)


# ================================================================
# BlockedParquetBatchDataset
# ================================================================

class BlockedParquetBatchDataset(IterableDataset):
    """
    IterableDataset in blocked mode.

    Features:
    - One block corresponds to a group:
        data/part-xxxxx.parquet
        user_info/part-xxxxx.parquet
        item_info/part-xxxxx.parquet
    - dataset reads data files in blocks
    - Yield an "already batched" payload each time to avoid mixing different blocks into the batch
    - The collator processes it based on the user/item side-info file in the payload.

    DDP improvements:
    - Simple rank::world_size round-robin allocation is no longer used
    - Change to greedy balanced distribution based on block estimated load (cost) to reduce serious imbalances between ranks
    """
    def __init__(self,
                 data_path,
                 user_info_path,
                 item_info_path,
                 columns=None,
                 batch_size=32,
                 shuffle=False,
                 distributed=False,
                 rank=0,
                 world_size=1,
                 drop_last=True,
                 seed=2026):
        super().__init__()
        self.columns = columns
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

        self.data_part_map = build_part_file_map(data_path)
        self.user_part_map = build_part_file_map(user_info_path)
        self.item_part_map = build_part_file_map(item_info_path)

        common_part_ids = sorted(
            set(self.data_part_map.keys())
            & set(self.user_part_map.keys())
            & set(self.item_part_map.keys())
        )
        if len(common_part_ids) == 0:
            raise ValueError(
                "No matched blocked parquet part ids across data/user_info/item_info.\n"
                f"data parts={sorted(self.data_part_map.keys())}\n"
                f"user parts={sorted(self.user_part_map.keys())}\n"
                f"item parts={sorted(self.item_part_map.keys())}"
            )

        self.blocks = []
        for pid in common_part_ids:
            data_file = self.data_part_map[pid]
            load_stat = estimate_parquet_block_cost(
                data_file, seq_len_col="seq_len", sample_rows=4096
            )

            self.blocks.append(
                {
                    "part_id": pid,
                    "data_file": data_file,
                    "user_info_file": self.user_part_map[pid],
                    "item_info_file": self.item_part_map[pid],
                    "num_rows": int(load_stat["num_rows"]),
                    "avg_seq_len": load_stat["avg_seq_len"],
                    "cost": float(load_stat["cost"]),
                }
            )

        if self.distributed:
            self.rank_blocks, self.rank_loads = self._assign_blocks_greedily(self.blocks, self.world_size)
            self.rank_blocks = self.rank_blocks[self.rank]
            self.my_estimated_load = float(self.rank_loads[self.rank])
        else:
            self.rank_blocks = self.blocks
            self.rank_loads = [sum(float(blk["cost"]) for blk in self.blocks)]
            self.my_estimated_load = float(self.rank_loads[0])

        if len(self.rank_blocks) == 0:
            raise ValueError(
                f"No blocked parquet files assigned to rank={rank}. "
                f"total_blocks={len(self.blocks)}, world_size={world_size}"
            )

        self.column_index = self._infer_column_index()
        self.block_row_counts = self._count_rows(self.rank_blocks)
        self.num_blocks = len(self.rank_blocks)
        self.num_samples = int(sum(self.block_row_counts.values()))
        self.num_batches = self._count_batches()

        # Expose per-rank statistics for balance diagnostics.
        self.rank_num_rows = int(sum(int(blk["num_rows"]) for blk in self.rank_blocks))
        self.rank_num_cost = float(sum(float(blk["cost"]) for blk in self.rank_blocks))

    def _assign_blocks_greedily(self, blocks, world_size):
        """
        Sort by block cost from large to small, and then greedily assign it to the rank with the smallest current total load.
        """
        sorted_blocks = sorted(blocks, key=lambda x: (x["cost"], x["num_rows"]), reverse=True)

        rank_buckets = [[] for _ in range(world_size)]
        rank_loads = [0.0 for _ in range(world_size)]

        for blk in sorted_blocks:
            target_rank = int(np.argmin(rank_loads))
            rank_buckets[target_rank].append(blk)
            rank_loads[target_rank] += float(blk["cost"])

        # Each rank is internally sorted by part_id to ensure stability.
        for r in range(world_size):
            rank_buckets[r] = sorted(rank_buckets[r], key=lambda x: x["part_id"])

        return rank_buckets, rank_loads

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def _count_rows(self, blocks):
        out = {}
        for blk in blocks:
            out[int(blk["part_id"])] = int(blk["num_rows"])
        return out

    def _count_batches(self):
        total = 0
        for blk in self.rank_blocks:
            n = self.block_row_counts[int(blk["part_id"])]
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += int(np.ceil(n / self.batch_size))
        return int(total)

    def _infer_column_index(self):
        for blk in self.rank_blocks:
            pf = pq.ParquetFile(blk["data_file"])
            for record_batch in pf.iter_batches(batch_size=32, columns=self.columns):
                df = record_batch.to_pandas()
                if len(df) == 0:
                    continue
                _, column_index = dataframe_to_darray(df)
                return column_index
        raise ValueError("Failed to infer column_index from blocked parquet files.")

    def __iter__(self):
        worker_info = get_worker_info()

        blocks = list(self.rank_blocks)
        seed_seq = np.random.SeedSequence([self.seed, self.epoch, self.rank])
        py_rng = random.Random(int(seed_seq.generate_state(1)[0]))

        if self.shuffle:
            py_rng.shuffle(blocks)

        if worker_info is not None:
            blocks = blocks[worker_info.id::worker_info.num_workers]
            worker_id = worker_info.id
        else:
            worker_id = 0
        np_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, self.rank, worker_id])
        )

        for blk in blocks:
            df = pd.read_parquet(blk["data_file"], columns=self.columns)
            if len(df) == 0:
                continue

            file_array, _ = dataframe_to_darray(df)

            if self.shuffle and len(file_array) > 1:
                perm = np_rng.permutation(len(file_array))
                file_array = file_array[perm]

            n = len(file_array)
            start = 0
            while start < n:
                end = min(start + self.batch_size, n)
                if self.drop_last and (end - start) < self.batch_size:
                    break

                rows = np.ascontiguousarray(file_array[start:end])

                yield {
                    "rows": rows,
                    "part_id": int(blk["part_id"]),
                    "data_file": blk["data_file"],
                    "user_info_file": blk["user_info_file"],
                    "item_info_file": blk["item_info_file"],
                }
                start = end


# ================================================================
# UniRankDataloader
# ================================================================

class UniRankDataloader(DataLoader):
    """
    DataLoader in blocked mode.

    Features:
    - Applicable to all datasets (TAAC2025 / KuaiRand / QK_Video)
    - Automatically read in blocks according to part-xxxxx.parquet
    - Each block corresponds to a set of data / user_info / item_info

    side-info path writing:
        train_user_info / train_item_info
        valid_user_info / valid_item_info
        test_user_info  / test_item_info
    """
    def __init__(self, feature_map, data_path, user_info=None, item_info=None,
                 batch_size=32, shuffle=False, num_workers=4, max_len=50, padding="pre",
                 distributed=False, rank=0, world_size=1, drop_last=True, split=None, **kwargs):

        self.feature_map = feature_map
        self.split = split

        self.block_cache_size = kwargs.pop("block_cache_size", 2)
        seed = int(kwargs.get("seed", 2026))
        pin_memory = bool(kwargs.pop("pin_memory", True))
        prefetch_factor = kwargs.pop("prefetch_factor", None)
        persistent_workers = kwargs.pop("persistent_workers", None)
        # Persistent workers retain a pickled copy of an IterableDataset.  In
        # a multi-epoch run, calling dataset.set_epoch() in the parent does
        # not update that copy.  Keep the speed benefit for the common one-pass
        # UniRank config, but recreate workers by default when epochs>1 so the
        # epoch-dependent block/sample shuffle actually changes.
        configured_epochs = int(kwargs.get("epochs", 1))
        multiprocessing_context = kwargs.pop("multiprocessing_context", None)
        self.return_timestamp_seq = bool(kwargs.pop("return_timestamp_seq", False))
        self.use_queryformerest_dataloader = bool(kwargs.pop("use_queryformerest_dataloader", False))
        self.return_timestamp_context_seq = bool(kwargs.pop("return_timestamp_context_seq", False))
        self.use_timestamp_context_aware_dataloader = bool(kwargs.pop("use_timestamp_context_aware_dataloader", False))

        user_info = resolve_side_info_path(
            split=self.split,
            key="user_info",
            explicit_path=user_info,
            config=kwargs
        )
        item_info = resolve_side_info_path(
            split=self.split,
            key="item_info",
            explicit_path=item_info,
            config=kwargs
        )

        self.dataset = BlockedParquetBatchDataset(
            data_path=data_path,
            user_info_path=user_info,
            item_info_path=item_info,
            columns=None,
            batch_size=batch_size,
            shuffle=shuffle,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            drop_last=drop_last,
            seed=seed
        )
        if distributed:
            print(
                f"[Rank {rank}] blocked load balanced: "
                f"blocks={self.dataset.num_blocks}, "
                f"rows={self.dataset.rank_num_rows}, "
                f"batches={self.dataset.num_batches}, "
                f"est_cost={self.dataset.rank_num_cost:.2f}"
            )

        collator_cls = TimestampContextBatchCollator if (self.return_timestamp_context_seq or self.use_timestamp_context_aware_dataloader) else TimestampBatchCollator if (self.return_timestamp_seq or self.use_queryformerest_dataloader) else BlockedBatchCollator
        collate_fn = collator_cls(
            feature_map=feature_map,
            max_len=max_len,
            column_index=self.dataset.column_index,
            user_info=user_info,
            item_info=item_info,
            padding=padding,
            cache_size=self.block_cache_size,
        )

        # In blocked mode, the dataset has been split according to rank, and sampler is not used.
        self.sampler_ref = None
        num_workers = max(0, int(num_workers))
        if prefetch_factor is None:
            prefetch_factor = 4 if num_workers > 0 else None
        elif num_workers == 0:
            prefetch_factor = None
        if persistent_workers is None:
            persistent_workers = num_workers > 0 and configured_epochs <= 1
        else:
            persistent_workers = bool(persistent_workers) and num_workers > 0
        if num_workers == 0:
            multiprocessing_context = None

        super().__init__(
            dataset=self.dataset,
            batch_size=None,
            shuffle=False,
            sampler=None,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn,
            worker_init_fn=(
                _single_threaded_worker_init if num_workers > 0 else None
            ),
            multiprocessing_context=multiprocessing_context,
        )

        self._configured_batch_size = int(batch_size)
        self.num_blocks = getattr(self.dataset, "num_blocks", 1)

        # Global sample number, available for display
        self.global_num_samples = getattr(self.dataset, "num_samples", len(self.dataset))

        # Actual number of samples in current rank/number of batches
        self.num_samples = getattr(self.dataset, "num_samples", len(self.dataset))
        self.num_batches = len(self.dataset)

        # Marked as blocked mode, used for DDP + blocked training logic in rank_model.py
        self.blocked = True

    def __len__(self):
        return super().__len__()

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


class SpawnUniRankDataloader(UniRankDataloader):
    """Blocked parquet loader using fresh spawn workers.

    Validation/test loaders are commonly created in a process that has already
    initialized CUDA, TorchInductor, pandas, and Arrow.  Forking such a
    multi-threaded native process is not always safe and can make Arrow readers
    die with SIGSEGV inside DataLoader workers.  A spawned worker starts with a
    clean interpreter state, at the cost of a small startup latency.  Training
    can continue to use the ordinary fork-based ``UniRankDataloader``.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("multiprocessing_context", "spawn")
        # Validation commonly has fewer batches than training; two windows per
        # worker are enough to overlap I/O without amplifying parquet memory.
        kwargs.setdefault("prefetch_factor", 2)
        # Evaluation loaders should release their parquet caches when an epoch's
        # iterator is exhausted instead of staying alive beside training workers.
        kwargs.setdefault("persistent_workers", False)
        super().__init__(*args, **kwargs)


# ================================================================
# BlockedBatchCollator
# ================================================================

class BlockedBatchCollator(object):
    """
    blocked version:
    Only the user_info/item_info of the current block is cached each time to avoid reading the entire side-info at once.
    """
    def __init__(self, feature_map, max_len, column_index, user_info, item_info,
                 padding="pre", cache_size=2):
        self.feature_map = feature_map
        self.max_len = max_len
        self.padding = padding
        self.cache_size = int(max(1, cache_size))

        self.all_cols = set(list(feature_map.features.keys()) + feature_map.labels)
        self.batch_cols = [(col, idx) for col, idx in column_index.items() if col in self.all_cols]
        self.task_labels = (
            [feature_map.target_label]
            if getattr(feature_map, "target_label", None)
            else list(feature_map.labels)
        )

        meta_fp = find_meta_data_json(user_info)
        try:
            with open(meta_fp, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read meta_data.json: {meta_fp}, error={e}")

        action_vocab = meta_data.get("action_vocab", None)
        if not action_vocab:
            raise ValueError(
                "action_vocab is missing in meta_data.json; unable to construct "
                "the single CTR behavior mask."
            )

        self.action_task_table = self._build_action_task_table(action_vocab)
        self.side_cache = OrderedDict()

    def _build_action_task_table(self, action_vocab):
        max_action_id = max(int(v) for v in action_vocab.values()) if len(action_vocab) > 0 else 0
        action_task_table = np.zeros(
            (max_action_id + 1, len(self.task_labels)), dtype=np.float32
        )

        def _task_aliases(task_name):
            aliases = {task_name}
            if task_name.startswith("is_"):
                aliases.add(task_name[3:])
            return aliases

        task_alias_sets = [_task_aliases(t) for t in self.task_labels]

        for action_name, action_id in action_vocab.items():
            action_id = int(action_id)
            if action_id <= 0:
                continue
            if not action_name or action_name == "exposure":
                continue
            parts = set(str(action_name).split("|"))
            for t_idx, aliases in enumerate(task_alias_sets):
                if len(parts.intersection(aliases)) > 0:
                    action_task_table[action_id, t_idx] = 1.0

        return torch.from_numpy(action_task_table)

    def _load_block_side_info(self, user_info_file, item_info_file):
        cache_key = (str(user_info_file), str(item_info_file))
        if cache_key in self.side_cache:
            value = self.side_cache.pop(cache_key)
            self.side_cache[cache_key] = value
            return value

        need_user_cols = ["user_index", "full_item_seq", "full_action_seq"]
        user_df = pd.read_parquet(user_info_file, columns=need_user_cols)
        user_df = user_df.set_index("user_index").sort_index()

        if len(user_df) == 0:
            raise ValueError(f"Empty blocked user_info file: {user_info_file}")

        user_indices = user_df.index.to_numpy(dtype=np.int64)
        user_index_min = int(user_indices.min())
        user_index_max = int(user_indices.max())

        user_row_lookup = torch.full(
            (user_index_max - user_index_min + 1,),
            -1,
            dtype=torch.long
        )
        user_row_lookup[user_indices - user_index_min] = torch.arange(
            len(user_df), dtype=torch.long
        )

        user_item_seqs = user_df["full_item_seq"].to_numpy()
        user_action_seqs = user_df["full_action_seq"].to_numpy()

        item_schema = get_parquet_schema_names(item_info_file)
        item_cols = ["item_index"] + [col for col in self.all_cols if col not in {"action", "item_index"}]
        item_cols = [c for c in item_cols if c in item_schema]

        if "item_index" not in item_cols:
            raise ValueError(f"item_index column missing in blocked item_info: {item_info_file}")

        item_df = pd.read_parquet(item_info_file, columns=item_cols).set_index("item_index").sort_index()

        if 0 not in item_df.index:
            item_df.loc[0] = 0
            item_df = item_df.sort_index()

        item_indices = item_df.index.to_numpy(dtype=np.int64)
        item_index_min = int(item_indices.min())
        item_index_max = int(item_indices.max())

        item_row_lookup = torch.full(
            (item_index_max - item_index_min + 1,),
            -1,
            dtype=torch.long
        )
        item_row_lookup[item_indices - item_index_min] = torch.arange(
            len(item_df), dtype=torch.long
        )

        item_tensors = {}
        for col in item_df.columns:
            if col in self.all_cols:
                col_array = np.ascontiguousarray(item_df[col].to_numpy(copy=True))
                item_tensors[col] = torch.from_numpy(col_array)

        side_info = {
            "user_index_min": user_index_min,
            "user_index_max": user_index_max,
            "user_row_lookup": user_row_lookup,
            "user_item_seqs": user_item_seqs,
            "user_action_seqs": user_action_seqs,
            "item_index_min": item_index_min,
            "item_index_max": item_index_max,
            "item_row_lookup": item_row_lookup,
            "item_tensors": item_tensors,
        }

        self.side_cache[cache_key] = side_info
        while len(self.side_cache) > self.cache_size:
            self.side_cache.popitem(last=False)

        return side_info

    def __call__(self, payload):
        if not isinstance(payload, dict):
            raise TypeError(
                "BlockedBatchCollator expects dataset payload dict, "
                f"but got type={type(payload)}"
            )

        rows = payload["rows"]
        user_info_file = payload["user_info_file"]
        item_info_file = payload["item_info_file"]

        if not isinstance(rows, np.ndarray):
            rows = np.asarray(rows)

        batch_tensor = torch.from_numpy(rows)
        batch_dict = build_batch_dict_from_tensor(batch_tensor, self.batch_cols)

        side = self._load_block_side_info(user_info_file, item_info_file)

        user_index_tensor = batch_dict["user_index"].long().cpu()
        lookup_pos = user_index_tensor - side["user_index_min"]

        if lookup_pos.min().item() < 0 or lookup_pos.max().item() >= len(side["user_row_lookup"]):
            raise IndexError(
                f"user_index exceeds blocked user_info range:"
                f"min={user_index_tensor.min().item()}, max={user_index_tensor.max().item()}, "
                f"allowed=[{side['user_index_min']}, {side['user_index_max']}] | file={user_info_file}"
            )

        user_row_ids = side["user_row_lookup"][lookup_pos]
        if (user_row_ids < 0).any():
            bad_uid = user_index_tensor[user_row_ids < 0][0].item()
            raise IndexError(f"user_index={bad_uid} does not exist in blocked user_info. file={user_info_file}")

        user_row_ids = user_row_ids.numpy().astype(np.int64, copy=False)
        seq_lens = batch_dict["seq_len"].int().cpu().numpy()

        user_item_seqs = side["user_item_seqs"][user_row_ids]
        user_action_seqs = side["user_action_seqs"][user_row_ids]

        batch_item_seqs = self._fast_pad(user_item_seqs, seq_lens)
        batch_action_seqs = self._fast_pad(user_action_seqs, seq_lens)

        mask = torch.from_numpy((batch_item_seqs > 0).astype(np.float32))

        batch_action_tensor = torch.from_numpy(
            batch_action_seqs.astype(np.int64, copy=False)
        )

        token_task_mask = self.action_task_table[batch_action_tensor]
        token_task_mask = token_task_mask * mask.unsqueeze(-1)
        task_masks = [
            token_task_mask[:, :, t] for t in range(token_task_mask.shape[-1])
        ]

        batch_size = len(user_row_ids)
        seq_total_len = batch_item_seqs.shape[1] + 1

        batch_items = np.empty((batch_size, seq_total_len), dtype=np.int64)
        batch_items[:, :-1] = batch_item_seqs
        batch_items[:, -1] = batch_dict["item_index"].cpu().numpy().astype(np.int64, copy=False)

        batch_actions = np.zeros((batch_size, seq_total_len), dtype=batch_action_seqs.dtype)
        batch_actions[:, :-1] = batch_action_seqs

        flat_items = batch_items.reshape(-1)
        lookup_pos = flat_items - side["item_index_min"]

        if lookup_pos.min() < 0 or lookup_pos.max() >= len(side["item_row_lookup"]):
            raise IndexError(
                f"item_index exceeds blocked item_info range:"
                f"min={flat_items.min()}, max={flat_items.max()}, "
                f"allowed=[{side['item_index_min']}, {side['item_index_max']}] | file={item_info_file}"
            )

        lookup_pos_tensor = torch.from_numpy(lookup_pos.astype(np.int64, copy=False))
        row_ids = side["item_row_lookup"][lookup_pos_tensor]

        if (row_ids < 0).any():
            bad_item = flat_items[(row_ids < 0).numpy()][0]
            raise IndexError(f"item_index={bad_item} does not exist in blocked item_info. file={item_info_file}")

        item_dict = {}
        for col, tensor_data in side["item_tensors"].items():
            item_dict[col] = tensor_data[row_ids].view(batch_size, seq_total_len)

        if "action" in self.all_cols:
            item_dict["action"] = torch.from_numpy(batch_actions)

        return batch_dict, item_dict, mask, task_masks

    def _fast_pad(self, user_seqs, seq_lens):
        max_len = self.max_len
        batch_size = len(user_seqs)
        result = np.zeros((batch_size, max_len), dtype=np.int64)

        for i in range(batch_size):
            seq = user_seqs[i]
            l = int(seq_lens[i])
            if l == 0:
                continue

            if self.padding == "pre":
                if l >= max_len:
                    result[i, :] = seq[l - max_len:l]
                else:
                    result[i, max_len - l:] = seq[:l]
            else:
                actual = min(l, max_len)
                result[i, :actual] = seq[:actual]
        return result


class TimestampBatchCollator(BlockedBatchCollator):
    """Blocked input contract with history and target timestamps."""

    def _load_block_side_info(self, user_info_file, item_info_file):
        side_info = super()._load_block_side_info(user_info_file, item_info_file)
        if "user_timestamp_seqs" in side_info:
            return side_info

        try:
            user_df = pd.read_parquet(user_info_file, columns=["user_index", "full_timestamp_seq"])
        except Exception:
            side_info["user_timestamp_seqs"] = None
            return side_info
        user_df = user_df.set_index("user_index").sort_index()
        side_info["user_timestamp_seqs"] = user_df["full_timestamp_seq"].to_numpy()
        return side_info

    def __call__(self, payload):
        outputs = super().__call__(payload)
        batch_dict, _, mask, _ = outputs

        side = self._load_block_side_info(payload["user_info_file"], payload["item_info_file"])
        timestamp_seqs = side.get("user_timestamp_seqs")
        if timestamp_seqs is None:
            batch_timestamps = torch.zeros_like(mask, dtype=torch.long)
            target_timestamps = torch.zeros(mask.size(0), dtype=torch.long)
            return outputs + (batch_timestamps, target_timestamps)

        user_index_tensor = batch_dict["user_index"].long().cpu()
        lookup_pos = user_index_tensor - side["user_index_min"]
        user_row_ids = side["user_row_lookup"][lookup_pos].numpy().astype(np.int64, copy=False)
        seq_lens = batch_dict["seq_len"].int().cpu().numpy()
        batch_timestamp_seqs = self._fast_pad(timestamp_seqs[user_row_ids], seq_lens)
        batch_timestamps = torch.from_numpy(batch_timestamp_seqs.astype(np.int64, copy=False))
        target_timestamps = np.zeros(len(user_row_ids), dtype=np.int64)
        for i, (seq, seq_len) in enumerate(zip(timestamp_seqs[user_row_ids], seq_lens)):
            seq_len = int(seq_len)
            if isinstance(seq, (list, tuple, np.ndarray)) and 0 <= seq_len < len(seq):
                target_timestamps[i] = int(seq[seq_len])
        target_timestamps = torch.from_numpy(target_timestamps)
        return outputs + (batch_timestamps, target_timestamps)


class TimestampContextBatchCollator(TimestampBatchCollator):
    """Blocked input contract with history timestamps and precomputed context sequences.

    When the precomputed context columns (``full_day_of_week_seq``,
    ``full_is_weekend_seq``, ``full_hour_seq``) are absent from user_info
    (e.g. data generated by an older preprocessing pipeline), they are
    lazily derived from ``full_timestamp_seq`` using Beijing time and the
    same +1 encoding as the preprocessing scripts, so that the history
    context sequences stay value-compatible with the target context scalars
    stored in batch_dict.

    Dataset-specific categorical context sequences (``tab``, ``pid``,
    ``session_id``) follow exactly the same padding contract as the time
    contexts. They are appended after ``hour`` in fixed order and are zero
    filled for datasets where they are absent.
    """

    BEIJING_TZ = "Asia/Shanghai"
    EXTRA_CONTEXT_COLUMNS = (
        ("full_day_of_week_seq", "user_day_of_week_seqs"),
        ("full_is_weekend_seq", "user_is_weekend_seqs"),
        ("full_hour_seq", "user_hour_seqs"),
        ("full_tab_seq", "user_tab_seqs"),
        ("full_pid_seq", "user_pid_seqs"),
        ("full_session_id_seq", "user_session_id_seqs"),
    )

    @staticmethod
    def _detect_timestamp_unit(values):
        # Same heuristic as detect_timestamp_unit_from_sample in the
        # preprocessing scripts: >= 1e12 -> milliseconds, else seconds.
        for v in values:
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(x):
                return "ms" if x >= 1e12 else "s"
        return "ms"

    @classmethod
    def _derive_context_seqs(cls, timestamp_seqs):
        """Lazily derive (day_of_week, is_weekend, hour) sequences from
        per-user timestamp lists. Returns None if timestamps are unavailable.

        The encoding matches the preprocessing scripts:
          day_of_week = dayofweek + 1   # 1..7
          is_weekend  = (dayofweek>=5) + 1  # 1..2
          hour        = hour + 1       # 1..24
        Invalid/missing timestamps map to id 0 (padding slot).
        """
        if timestamp_seqs is None:
            return None, None, None

        # Detect unit from the first non-empty sequence.
        unit = "ms"
        for seq in timestamp_seqs:
            if seq is not None and len(seq) > 0:
                unit = cls._detect_timestamp_unit(seq)
                break

        dow_seqs = np.empty(len(timestamp_seqs), dtype=object)
        weekend_seqs = np.empty(len(timestamp_seqs), dtype=object)
        hour_seqs = np.empty(len(timestamp_seqs), dtype=object)
        for i, seq in enumerate(timestamp_seqs):
            if seq is None or len(seq) == 0:
                dow_seqs[i] = np.zeros(0, dtype=np.int32)
                weekend_seqs[i] = np.zeros(0, dtype=np.int32)
                hour_seqs[i] = np.zeros(0, dtype=np.int32)
                continue
            arr = np.asarray(seq, dtype=np.int64)
            ts = pd.Series(arr)
            dt = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce").dt.tz_convert(cls.BEIJING_TZ)
            raw_dow = dt.dt.dayofweek.fillna(-1).astype(np.int16)
            raw_hour = dt.dt.hour.fillna(-1).astype(np.int16)
            dow = (raw_dow + 1).astype(np.int32)              # -1 -> 0 (padding)
            weekend = ((raw_dow >= 5).astype(np.int32) + 1)   # invalid -> 1, but masked later
            hour = (raw_hour + 1).astype(np.int32)            # -1 -> 0 (padding)
            # Mark invalid positions as 0 across all three so the model
            # can treat id 0 as padding consistently.
            invalid = (raw_dow < 0)
            dow[invalid] = 0
            weekend[invalid] = 0
            hour[invalid] = 0
            dow_seqs[i] = dow
            weekend_seqs[i] = weekend
            hour_seqs[i] = hour
        return dow_seqs, weekend_seqs, hour_seqs

    def _load_block_side_info(self, user_info_file, item_info_file):
        # Mimic TimestampBatchCollator._load_block_side_info exactly: call
        # super() (which already loads base cols + item_info + timestamp),
        # then do one additional read for the context columns, exactly like
        # how TimestampBatchCollator does one additional read for timestamp.
        side_info = super()._load_block_side_info(user_info_file, item_info_file)
        if "user_day_of_week_seqs" in side_info:
            return side_info

        # Only read context columns that actually exist in this file (e.g.
        # Taobao has no full_day_of_week_seq), so a missing field does not
        # force a fallback for the fields that ARE present.
        schema_cols = set(get_parquet_schema_names(user_info_file))
        context_cols = ["user_index"] + [c for c in
                        (col for col, _ in self.EXTRA_CONTEXT_COLUMNS)
                        if c in schema_cols]
        # No context columns at all -> derive lazily from timestamps.
        if len(context_cols) == 1:
            timestamp_seqs = side_info.get("user_timestamp_seqs")
            dow, weekend, hour = self._derive_context_seqs(timestamp_seqs)
            side_info["user_day_of_week_seqs"] = dow
            side_info["user_is_weekend_seqs"] = weekend
            side_info["user_hour_seqs"] = hour
            for _, key in self.EXTRA_CONTEXT_COLUMNS[3:]:
                side_info[key] = None
            return side_info

        try:
            user_df = pd.read_parquet(user_info_file, columns=context_cols)
        except Exception:
            timestamp_seqs = side_info.get("user_timestamp_seqs")
            dow, weekend, hour = self._derive_context_seqs(timestamp_seqs)
            side_info["user_day_of_week_seqs"] = dow
            side_info["user_is_weekend_seqs"] = weekend
            side_info["user_hour_seqs"] = hour
            for _, key in self.EXTRA_CONTEXT_COLUMNS[3:]:
                side_info[key] = None
            return side_info

        user_df = user_df.set_index("user_index").sort_index()
        for col, key in self.EXTRA_CONTEXT_COLUMNS:
            if col in user_df.columns:
                side_info[key] = user_df[col].to_numpy()
            else:
                side_info[key] = None
        return side_info

    def __call__(self, payload):
        # Mirror TimestampBatchCollator.__call__ exactly: super().__call__
        # produces (batch_dict, item_dict, mask, multi_masks); then ONE call
        # to _load_block_side_info; then pad the extra sequences. The only
        # difference from time is that we also pad the three context seqs.
        outputs = super().__call__(payload)
        # The parent TimestampBatchCollator has already appended timestamp
        # tensors, so its result contains six values. Only the four base
        # values are needed to build the additional context sequences.
        batch_dict, _, mask, _ = outputs[:4]

        side = self._load_block_side_info(payload["user_info_file"], payload["item_info_file"])

        # TimestampBatchCollator.__call__ already produced these two tensors.
        # Do not append them again, otherwise model inputs shift and context
        # fields are read from the wrong positions.
        batch_timestamps, target_timestamps = outputs[4], outputs[5]

        # ---- context (same pattern, reuse user_row_ids / seq_lens) ----
        day_of_week_seqs = side.get("user_day_of_week_seqs")
        is_weekend_seqs = side.get("user_is_weekend_seqs")
        hour_seqs = side.get("user_hour_seqs")
        tab_seqs = side.get("user_tab_seqs")
        pid_seqs = side.get("user_pid_seqs")
        session_id_seqs = side.get("user_session_id_seqs")

        user_index_tensor = batch_dict["user_index"].long().cpu()
        lookup_pos = user_index_tensor - side["user_index_min"]
        user_row_ids = side["user_row_lookup"][lookup_pos].numpy().astype(np.int64, copy=False)
        seq_lens = batch_dict["seq_len"].int().cpu().numpy()

        def _pad_or_zeros(seqs):
            if seqs is None:
                return torch.zeros_like(mask, dtype=torch.long)
            return torch.from_numpy(self._fast_pad(seqs[user_row_ids], seq_lens).astype(np.int64, copy=False))

        batch_day_of_week = _pad_or_zeros(day_of_week_seqs)
        batch_is_weekend = _pad_or_zeros(is_weekend_seqs)
        batch_hour = _pad_or_zeros(hour_seqs)
        batch_tab = _pad_or_zeros(tab_seqs)
        batch_pid = _pad_or_zeros(pid_seqs)
        batch_session_id = _pad_or_zeros(session_id_seqs)
        return outputs + (
            batch_day_of_week, batch_is_weekend, batch_hour,
            batch_tab, batch_pid, batch_session_id,
        )
