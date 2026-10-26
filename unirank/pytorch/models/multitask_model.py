# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os, sys
import logging
from unirank.pytorch.models import BaseModel
from unirank.pytorch.torch_utils import get_device, get_loss
from tqdm import tqdm
from collections import defaultdict, OrderedDict


class MultiTaskModel(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="MultiTaskModel",
                 task="binary_classification",
                 num_tasks=None,
                 loss_weight='EQ',
                 gpu=-1,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(MultiTaskModel, self).__init__(feature_map=feature_map,
                                             model_id=model_id,
                                             task="binary_classification",
                                             gpu=gpu,
                                             loss_weight=loss_weight,
                                             monitor=monitor,
                                             save_best_only=save_best_only,
                                             monitor_mode=monitor_mode,
                                             early_stop_patience=early_stop_patience,
                                             eval_steps=eval_steps,
                                             reduce_lr_on_plateau=reduce_lr_on_plateau,
                                             **kwargs)
        self.device = get_device(gpu)
        self.num_tasks = (
            len(self.get_active_labels()) if num_tasks is None else num_tasks
        )
        self.loss_weight = loss_weight
        self._validate_task_contract()
        if isinstance(task, list):
            assert len(task) == self.num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(self.num_tasks)]
            )

    def compile(self, optimizer, loss, lr):
        self._validate_task_contract(loss)
        self._compile_optimizers(optimizer, lr)
        if isinstance(loss, list):
            self.loss_fn = [get_loss(l) for l in loss]
        else:
            self.loss_fn = [get_loss(loss) for _ in range(self.num_tasks)]
        self._maybe_enable_torch_compile()

    def _validate_task_contract(self, loss=None):
        labels = list(self.feature_map.labels)
        active_labels = self.get_active_labels()
        if self.num_tasks < 1:
            raise ValueError(f"num_tasks must be positive, got {self.num_tasks}.")
        if len(active_labels) != self.num_tasks:
            raise ValueError(
                f"num_tasks={self.num_tasks} does not match active labels "
                f"{active_labels}; available dataset labels are {labels}."
            )
        if loss is not None and isinstance(loss, list) and len(loss) != self.num_tasks:
            raise ValueError(
                f"Expected exactly {self.num_tasks} loss function(s) for labels "
                f"{active_labels}, "
                f"but got {len(loss)}: {loss}."
            )
        tower = getattr(self, "tower", None)
        if tower is not None and len(tower) != self.num_tasks:
            raise ValueError(
                f"Prediction tower count {len(tower)} does not match "
                f"num_tasks={self.num_tasks}."
            )

    def get_active_labels(self):
        target_label = getattr(self.feature_map, "target_label", None)
        return [target_label] if target_label else list(self.feature_map.labels)

    def get_labels(self, inputs):
        """ Override get_labels() to use multiple labels """
        labels = self.get_active_labels()
        batch_dict = inputs[0]
        y = [batch_dict[label].to(self.device).float().view(-1, 1)
             for label in labels]
        return y

    def _validate_prediction_contract(self, return_dict):
        expected = {f"{label}_pred" for label in self.get_active_labels()}
        actual = {key for key in return_dict if key.endswith("_pred")}
        if actual != expected:
            raise ValueError(
                "Prediction outputs do not match the active task contract: "
                f"expected {sorted(expected)}, got {sorted(actual)}."
            )

    def add_loss(self, return_dict, y_true):
        labels = self.get_active_labels()
        self._validate_task_contract()
        self._validate_prediction_contract(return_dict)
        if len(y_true) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} target tensor(s), got {len(y_true)}."
            )
        loss = [self.loss_fn[i](return_dict["{}_pred".format(labels[i])], y_true[i], reduction='mean')
                for i in range(self.num_tasks)]
        if self.loss_weight == 'EQ':
            # Default: All losses are weighted equally
            loss = torch.sum(torch.stack(loss))
        return loss

    def compute_loss(self, return_dict, y_true):
        return self.add_loss(return_dict, y_true)

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            raw_generator = data_generator
            y_pred_all = defaultdict(list)
            y_true_all = defaultdict(list)
            labels = self.get_active_labels()
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(
                    data_generator,
                    disable=False,
                    file=sys.stdout,
                    desc=f"Rank {self.rank} | Eval",
                    position=self.rank,
                    leave=(self.rank == 0),
                    dynamic_ncols=True
                )

            amp_ctx = self._get_amp_context()
            with amp_ctx:
                for batch_data in data_generator:
                    self._profile_inference_gflops_per_sample(batch_data)
                    return_dict = self.forward(batch_data)
                    self._validate_prediction_contract(return_dict)
                    batch_y_true = self.get_labels(batch_data)
                    for i in range(len(labels)):
                        y_pred_all[labels[i]].extend(
                            return_dict["{}_pred".format(labels[i])].float().data.cpu().numpy().reshape(-1))
                        y_true_all[labels[i]].extend(batch_y_true[i].data.cpu().numpy().reshape(-1))
                    if self.feature_map.group_id is not None:
                        group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))

            all_val_logs = {}
            mean_val_logs = defaultdict(list)
            group_id = np.array(group_id) if len(group_id) > 0 else None

            # ---- Distributed consistency check: All ranks must agree on "whether to perform distributed evaluation aggregation" ----
            _distributed_eval = False
            _is_sampler_distributed_eval = False
            _is_blocked_distributed_eval = False
            total_samples = None

            if self._is_distributed():
                _is_sampler_distributed_eval = bool(
                    hasattr(raw_generator, "sampler")
                    and isinstance(raw_generator.sampler, DistributedSampler)
                )

                _is_blocked_distributed_eval = bool(
                    getattr(raw_generator, "blocked", False)
                    and hasattr(raw_generator, "dataset")
                    and getattr(raw_generator.dataset, "distributed", False)
                )

                local_flag = int(_is_sampler_distributed_eval or _is_blocked_distributed_eval)
                flag_t = torch.tensor([local_flag], dtype=torch.int32, device=self.device)
                gathered_flags = [torch.zeros_like(flag_t) for _ in range(self.world_size)]
                dist.all_gather(gathered_flags, flag_t)
                flags = [int(x.item()) for x in gathered_flags]

                if len(set(flags)) != 1:
                    raise RuntimeError(
                        f"[Rank {self.rank}] Inconsistent distributed-eval flags across ranks: {flags}. "
                        "This can cause a collective deadlock. Please ensure that the validation DataLoader configuration of each rank is consistent."
                    )

                _distributed_eval = bool(flags[0])

                if self.force_distributed_eval and not _distributed_eval:
                    raise RuntimeError(
                        "force_distributed_eval=True, but the currently evaluated DataLoader does not have distributed sharding enabled."
                    )

                if _distributed_eval:
                    if _is_blocked_distributed_eval:
                        local_samples_t = torch.tensor(
                            [len(y_true_all[labels[0]])],
                            dtype=torch.int64,
                            device=self.device
                        )
                        dist.all_reduce(local_samples_t, op=dist.ReduceOp.SUM)
                        total_samples = int(local_samples_t.item())
                    else:
                        total_samples = len(raw_generator.dataset)

            for i in range(len(labels)):
                y_pred = np.array(y_pred_all[labels[i]], np.float64)
                y_true = np.array(y_true_all[labels[i]], np.float64)

                if _distributed_eval:
                    y_pred, y_true, task_group_id = self._gather_eval_results(
                        y_pred, y_true, group_id, total_samples
                    )
                    if not self._is_main_process():
                        continue
                else:
                    task_group_id = group_id

                if metrics is not None:
                    val_logs = self.evaluate_metrics(y_true, y_pred, metrics, task_group_id)
                else:
                    val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, task_group_id)
                if self._is_main_process():
                    logging.info('[Task: {}][Metrics] '.format(labels[i]) + ' - '.join(
                        '{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                for k, v in val_logs.items():
                    all_val_logs['{}_{}'.format(labels[i], k)] = v
                    mean_val_logs[k].append(v)

            if _distributed_eval and not self._is_main_process():
                return OrderedDict()

            for k, v in mean_val_logs.items():
                mean_val_logs[k] = np.mean(v)
            all_val_logs.update(mean_val_logs)
            return all_val_logs

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_all = defaultdict(list)
            labels = self.get_active_labels()
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                self._validate_prediction_contract(return_dict)
                for i in range(len(labels)):
                    y_pred_all[labels[i]].extend(
                        return_dict["{}_pred".format(labels[i])].float().data.cpu().numpy().reshape(-1))
        return y_pred_all
