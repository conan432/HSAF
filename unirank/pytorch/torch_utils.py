# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


import os
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import random
import inspect


def disable_torch_compile(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "disable"):
        return dynamo.disable(fn)
    return fn


def make_flop_counter(display=False):
    """Build a FlopCounterMode that also counts SDPA variants.

    Torch's built-in registry already covers CUDA flash/efficient/cudnn SDPA
    kernels but not the high-level op or the CPU flash kernel.  Register those
    missing formulas so the counter does not silently drop attention FLOPs.
    Returns ``None`` when the installed torch version has no FlopCounterMode.
    """
    try:
        from torch.utils.flop_counter import (
            FlopCounterMode,
            sdpa_backward_flop_count,
            sdpa_flop_count,
        )
    except Exception:
        return None

    def _sdpa_forward(
        query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs
    ):
        return sdpa_flop_count(query_shape, key_shape, value_shape)

    def _sdpa_backward(
        grad_out_shape,
        query_shape,
        key_shape,
        value_shape,
        *args,
        out_shape=None,
        **kwargs,
    ):
        return sdpa_backward_flop_count(
            grad_out_shape, query_shape, key_shape, value_shape
        )

    custom_mapping = {}
    for op_name in (
        "scaled_dot_product_attention",
        "_scaled_dot_product_flash_attention_for_cpu",
    ):
        op = getattr(torch.ops.aten, op_name, None)
        if op is not None:
            custom_mapping[op] = _sdpa_forward
    backward_op = getattr(
        torch.ops.aten,
        "_scaled_dot_product_flash_attention_for_cpu_backward",
        None,
    )
    if backward_op is not None:
        custom_mapping[backward_op] = _sdpa_backward
    return FlopCounterMode(display=display, custom_mapping=custom_mapping)


def seed_everything(seed=1029, deterministic=False, warn_only=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=warn_only)


def build_batch_dict_from_tensor(batch_tensor, batch_columns):
    """Slice a batch tensor into a feature dictionary."""
    return {column: batch_tensor[:, index] for column, index in batch_columns}


def parse_gpu_ids(gpu_arg):
    """Parse a comma-separated GPU list; -1 selects CPU."""
    value = str(gpu_arg).strip()
    if value == "-1":
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--gpu cannot be empty; use --gpu -1 for CPU")
    if any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid --gpu value: {gpu_arg}; expected values such as 0,1,2,3")
    gpu_ids = [int(part) for part in parts]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"Duplicate GPU IDs in --gpu: {gpu_arg}")
    return gpu_ids


def setup_visible_devices(gpu_ids):
    """Set CUDA_VISIBLE_DEVICES from physical GPU IDs."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))


def init_distributed_env():
    """Read the distributed process coordinates injected by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return world_size > 1, rank, local_rank, world_size, local_world_size


def is_main_process(rank):
    return rank == 0


def distributed_barrier(local_rank=None):
    """Synchronize the active process group, using the NCCL device when required."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() == "nccl" and local_rank is not None:
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()

def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device

def get_optimizer(optimizer, params, lr, weight_decay=0):
    params = list(params)
    if len(params) == 0:
        return None
    if isinstance(optimizer, str):
        optimizer_name = optimizer.strip()
        optimizer_key = optimizer_name.lower()
        heavyball_alias = {
            "muon": "Muon",
            "laprop": "LaProp",
            "scion": "Scion",
            "soap": "SOAP",
        }
        use_heavyball = optimizer_key.startswith("heavyball.")
        if use_heavyball:
            heavyball_name = optimizer_name.split(".", 1)[1]
        else:
            heavyball_name = heavyball_alias.get(optimizer_key)
            use_heavyball = heavyball_name is not None

        if use_heavyball:
            if sys.version_info < (3, 10):
                raise RuntimeError(
                    "HeavyBall 3.2.0 requires Python 3.10 or newer."
                )
            try:
                import heavyball
            except ImportError as error:
                raise ImportError(
                    "HeavyBall optimizer '{}' requires the 'heavyball' package. "
                    "Install dependencies with `pip install -r requirements.txt`."
                    .format(optimizer_name)
                ) from error

            heavyball_classes = {}
            for name in dir(heavyball):
                candidate = getattr(heavyball, name)
                if (
                    inspect.isclass(candidate)
                    and issubclass(candidate, torch.optim.Optimizer)
                ):
                    heavyball_classes[name.lower()] = name
            resolved_name = heavyball_classes.get(heavyball_name.lower())
            if resolved_name is None:
                raise NotImplementedError(
                    "HeavyBall optimizer={} is not supported.".format(
                        heavyball_name
                    )
                )
            optimizer_cls = getattr(heavyball, resolved_name)
        else:
            optimizer_alias = {
                "adam": "Adam",
                "adamw": "AdamW",
                "adagrad": "Adagrad",
                "sgd": "SGD",
                "sparseadam": "SparseAdam",
            }
            optimizer_name = optimizer_alias.get(optimizer_key, optimizer_name)
            try:
                optimizer_cls = getattr(torch.optim, optimizer_name)
            except AttributeError as error:
                raise NotImplementedError(
                    "optimizer={} is not supported.".format(optimizer_name)
                ) from error
    else:
        optimizer_cls = optimizer

    if not callable(optimizer_cls):
        raise NotImplementedError(
            "optimizer={} is not supported.".format(optimizer_cls)
        )

    kwargs = {"lr": lr}
    if "weight_decay" in inspect.signature(optimizer_cls).parameters:
        kwargs["weight_decay"] = weight_decay
    try:
        return optimizer_cls(params, **kwargs)
    except Exception as error:
        raise RuntimeError(
            "Failed to initialize optimizer={} with lr={} and weight_decay={}."
            .format(
                getattr(optimizer_cls, "__name__", str(optimizer_cls)),
                lr,
                weight_decay,
            )
        ) from error

def get_loss(loss):
    if isinstance(loss, str):
        if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
    try:
        loss_fn = getattr(torch.functional.F, loss)
    except:
        try:
            loss_fn = eval("losses." + loss)
        except:
            raise NotImplementedError("loss={} is not supported.".format(loss))
    return loss_fn

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        activation_name = activation.strip().lower()
        if activation_name in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation_name == "none":
            return None
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "softplus":
            return nn.Softplus()
        elif activation_name == "silu":
            return nn.SiLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "mish":
            return nn.Mish()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "softmax":
            return nn.Softmax(dim=-1)
        elif activation_name == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        elif activation_name == "dice":
            from unirank.pytorch.layers.activations import Dice
            return Dice(hidden_units)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation

def get_initializer(initializer):
    if isinstance(initializer, str):
        try:
            initializer = eval(initializer)
        except:
            raise ValueError("initializer={} is not supported."\
                             .format(initializer))
    return initializer
