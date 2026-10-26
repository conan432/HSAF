# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2025. FuxiCTR Authors. All rights reserved.

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

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import gc
import logging
import argparse
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._dynamo

torch._dynamo.config.optimize_ddp = False

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from unirank.utils import (
    load_config, resolve_task_contract, set_logger, print_to_json, print_to_list
)
from unirank.features import FeatureMap
from unirank.pytorch.dataloaders import RankDataLoader
from unirank.pytorch.torch_utils import (
    distributed_barrier,
    init_distributed_env,
    is_main_process,
    parse_gpu_ids,
    seed_everything,
    setup_visible_devices,
)
from unirank.preprocess import FeatureProcessor, build_dataset
import model_zoo


if __name__ == '__main__':
    """
    Single card:
      python run_expid.py

    DDP multi-card (single machine):
      torchrun --standalone --nproc_per_node=4 run_expid.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='RankMixer_KuaiRand_Video_Action', help='The experiment id to run.')
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU ids, e.g. "0" or "0,1,2,3"; use "-1" for cpu')
    parser.add_argument(
        '--target_label',
        default=argparse.SUPPRESS,
        help=(
            'Override the configured single-task label. Use "null", "none", '
            '"all", or "multi_task" to train all dataset labels.'
        ),
    )
    parser.add_argument(
        '--enable_bf16',
        type=lambda value: str(value).strip().lower() in {'1', 'true', 'yes', 'on'},
        default=True,
        help='Enable bfloat16 mixed precision training (default: True).'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Load the existing model checkpoint and run test evaluation only.'
    )
    args = vars(parser.parse_args())

    try:
        gpu_ids = parse_gpu_ids(args['gpu'])
        setup_visible_devices(gpu_ids)

        distributed, rank, local_rank, world_size, local_world_size = init_distributed_env()

        # Basic legality check + distributed initialization
        if distributed:
            if len(gpu_ids) == 0:
                raise ValueError("CPU cannot be used in DDP mode (--gpu -1)")
            if local_world_size != len(gpu_ids):
                raise ValueError(
                    f"LOCAL_WORLD_SIZE({local_world_size}) is inconsistent with --gpu number({len(gpu_ids)})."
                    f"Please ensure that the number of torchrun --nproc_per_node and --gpu are consistent."
                )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available and DDP+NCCL cannot be used.")

            torch.cuda.set_device(local_rank)

            # Only initialize when not initialized to avoid repeated initialization errors.
            if not dist.is_available():
                raise RuntimeError("torch.distributed is not available.")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
            # NCCL is responsible for training tensor communication; large verification result objects are aggregated through CPU/Gloo,
            # Avoid gather_object producing large serialized byte tensor on GPU.
            eval_process_group = dist.new_group(
                backend="gloo",
                timeout=timedelta(minutes=60)
            )
            distributed_barrier(local_rank)
        else:
            eval_process_group = None
            # In non-torchrun mode, multiple GPUs are not allowed (to avoid being mistaken for automatic DDP)
            if len(gpu_ids) > 1:
                raise ValueError(
                    "Multiple GPUs detected, but currently not in torchrun mode. \n"
                    "Please use: torchrun --standalone --nproc_per_node=<Number of GPUs> run_expid.py ... --gpu 0,1,..."
                )
            if len(gpu_ids) == 1 and not torch.cuda.is_available():
                raise RuntimeError("GPU specified but CUDA is not available.")

        experiment_id = args['expid']
        params = load_config(args['config'], experiment_id)
        # Optional observation-experiment diagnostics.  Pop these runner-only
        # keys before they are passed to FeatureProcessor or a model class.
        clock_telemetry_dir = params.pop('clock_telemetry_dir', None)
        clock_telemetry_every = int(params.pop('clock_telemetry_every', 50))
        if 'target_label' in args:
            target_override = args['target_label']
            if target_override.lower() in {'null', 'none', 'all', 'multi_task'}:
                target_override = None
            params['target_label'] = target_override
            resolve_task_contract(params)

        configured_labels = [
            col['name'] for col in params.get('label_col', [])
            if col.get('active', True)
        ]
        target_label = (
            params['target_label'] if 'target_label' in params
            else params.get('default_target_label')
        )
        if target_label:
            loss = params.get('loss')
            if target_label not in configured_labels:
                raise ValueError(
                    f"target_label={target_label!r} must be one of "
                    f"the dataset labels {configured_labels}."
                )
            if params.get('num_tasks') != 1:
                raise ValueError(
                    "A configured target_label requires num_tasks=1, "
                    f"got {params.get('num_tasks')}."
                )
            if not isinstance(loss, list) or len(loss) != 1:
                raise ValueError(
                    "A configured target_label requires a one-element loss list, "
                    f"got {loss}."
                )
            if params.get('task') != 'binary_classification':
                raise ValueError(
                    "Single-task CTR requires task=binary_classification, "
                    f"got {params.get('task')}."
                )
        else:
            loss = params.get('loss')
            if params.get('num_tasks') != len(configured_labels):
                raise ValueError(
                    "Multi-task training requires num_tasks to match all dataset "
                    f"labels: num_tasks={params.get('num_tasks')}, "
                    f"labels={configured_labels}."
                )
            if isinstance(loss, list) and len(loss) != len(configured_labels):
                raise ValueError(
                    "Multi-task training requires one loss per dataset label, "
                    f"got {loss} for {configured_labels}."
                )

        # UltraHSTU's flex_attention block-mask compiler is not compatible
        # with the current TorchInductor path, so keep this model in eager mode.
        if params.get('model') in ('UltraHSTU', 'HSTU', 'HSTUPaper'):
            params['enable_torch_compile'] = False

        # Equipment parameters:
        # - DDP: local_rank maps to CUDA_VISIBLE_DEVICES internal sequence number
        # - Single card GPU: Fixed use of visible devices 0
        if len(gpu_ids) == 0:
            params['gpu'] = -1
        else:
            params['gpu'] = local_rank if distributed else 0

        params['distributed'] = distributed
        params['rank'] = rank
        params['local_rank'] = local_rank
        params['world_size'] = world_size

        # bf16 switch: command line parameters override fields of the same name in config (if present)
        params['enable_bf16'] = args['enable_bf16']

        strict_deterministic = bool(params.get("strict_deterministic", False))
        if strict_deterministic:
            # Strict runs trade throughput for a reproducible execution path.
            # Normal runs are unaffected because Base sets strict_deterministic=False.
            params["enable_bf16"] = False
            params["enable_torch_compile"] = False
            params["num_workers"] = 0
            if torch.cuda.is_available():
                torch.set_float32_matmul_precision("highest")

        if is_main_process(rank):
            set_logger(params)
            logging.info("Params: " + print_to_json(params))
        else:
            logging.getLogger().handlers = []
            logging.basicConfig(level=logging.ERROR)

        # Each rank uses a different seed offset
        seed_everything(
            seed=params['seed'] + rank,
            deterministic=strict_deterministic,
            warn_only=bool(params.get("deterministic_warn_only", True)),
        )

        data_dir = params.get(
            'dataset_dir', os.path.join(params['data_root'], params['dataset_id'])
        )
        feature_map_json = os.path.join(data_dir, "feature_map.json")

        if distributed:
            if is_main_process(rank):
                feature_encoder = FeatureProcessor(**params)
                params["train_data"], params["valid_data"], params["test_data"] = \
                    build_dataset(feature_encoder, **params)

            obj_list = [[
                params.get("train_data", None),
                params.get("valid_data", None),
                params.get("test_data", None)
            ]] if is_main_process(rank) else [[None, None, None]]

            dist.broadcast_object_list(obj_list, src=0)
            params["train_data"], params["valid_data"], params["test_data"] = obj_list[0]
            distributed_barrier(local_rank)
        else:
            feature_encoder = FeatureProcessor(**params)
            params["train_data"], params["valid_data"], params["test_data"] = \
                build_dataset(feature_encoder, **params)

        feature_map = FeatureMap(params['dataset_id'], data_dir)
        feature_map.load(feature_map_json, params)
        if target_label and feature_map.target_label is None:
            raise ValueError(
                "Single-task training requires feature_map.target_label to be set."
            )
        if is_main_process(rank):
            logging.info("Feature specs: " + print_to_json(feature_map.features))

        model_class = getattr(model_zoo, params['model'])
        model = model_class(feature_map, **params)
        model.model_to_device()
        if clock_telemetry_dir and is_main_process(rank):
            from observation_experiment.clock_telemetry import ClockTelemetry
            model.clock_telemetry = ClockTelemetry(
                experiment_id, clock_telemetry_dir, clock_telemetry_every
            ).attach(model)
            logging.info(
                "History-clock telemetry: %s (every %d steps)",
                model.clock_telemetry.path, clock_telemetry_every
            )
        if distributed:
            model.set_eval_process_group(eval_process_group)

        if distributed:
            ddp_model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=params.get("find_unused_parameters", True)
            )
            # Rely on the rank_model.py you modified earlier
            model.set_ddp_model(ddp_model)

        if is_main_process(rank):
            model.count_parameters()

        if args.get('test_only', False):
            if not os.path.exists(model.checkpoint):
                raise FileNotFoundError(
                    f"Test-only checkpoint does not exist: {model.checkpoint}")
            if is_main_process(rank):
                logging.info('Test-only loading checkpoint: %s', model.checkpoint)
            model.load_weights(model.checkpoint)
            test_params = dict(params)
            test_params.update({
                'distributed': False,
                'rank': 0,
                'local_rank': 0,
                'world_size': 1,
            })
            logging.info('******** Test evaluation ********')
            test_gen = RankDataLoader(
                feature_map, stage='test', **test_params).make_iterator()
            model.evaluate(test_gen)
            logging.info('Keeping model checkpoint: %s', model.checkpoint)
            raise SystemExit(0)

        # --------------------
        # Build data iterators
        # --------------------
        if distributed:
            train_params = dict(params)
            train_params.update({
                "distributed": True,
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size
            })
            # All ranks build train and valid. During verification, each rank performs parallel inference and then all_gather converges.
            train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **train_params).make_iterator()
        else:
            train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()


        if distributed and dist.is_initialized():
            distributed_barrier(local_rank)
        model.fit(train_gen, validation_data=valid_gen, **params)

        del train_gen, valid_gen
        gc.collect()

        if params.get("test_data", None):
            if is_main_process(rank):
                logging.info('******** Test evaluation ********')

            test_params = dict(params)
            if distributed:
                test_params.update({
                    "distributed": True,
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size
                })
            else:
                test_params.update({
                    "distributed": False,
                    "rank": 0,
                    "local_rank": 0,
                    "world_size": 1
                })

            test_gen = RankDataLoader(feature_map, stage='test', **test_params).make_iterator()

            if distributed and dist.is_available() and dist.is_initialized():
                distributed_barrier(local_rank)
            model.evaluate(test_gen)

            del test_gen
            gc.collect()

            if distributed and dist.is_available() and dist.is_initialized():
                distributed_barrier(local_rank)
            if is_main_process(rank) and params.get('keep_checkpoint', True):
                logging.info("Keeping model checkpoint: %s", model.checkpoint)
            elif is_main_process(rank):
                model.delete_checkpoint()
            if distributed and dist.is_available() and dist.is_initialized():
                distributed_barrier(local_rank)

    finally:
        if distributed and dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
