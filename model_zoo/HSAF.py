"""Public HSAF model interfaces.

The paper uses two architecture instances:

* ``HSAFStaged``: HSAF built on the staged RankMixer backbone.
* ``HSAFInterleaved``: HSAF built on the interleaved EST backbone.

The implementation in :mod:`model_zoo.EviTime_gatecalib_use_time` preserves the
internal names used in the reported experiments. These thin public aliases avoid
changing state keys or numerical behavior while exposing the paper terminology.
"""
from .EST import EST
from .RankMixer import RankMixer
from .EviTime import EST_EviTime
from .EviTime_gatecalib_use_time import (
    EST_EviTime_gatecalib_use_time,
    RankMixer_EviTime_gatecalib_use_time,
)
from .StateAsFeature import EST_StateFeature, RankMixer_StateFeature


class HSAFStaged(RankMixer_EviTime_gatecalib_use_time):
    """HSAF instance with staged sequence modeling and feature interaction."""


class HSAFInterleaved(EST_EviTime_gatecalib_use_time):
    """HSAF instance with layer-wise interleaved history interaction."""


# Exact entry point used by the reported interleaved component-ablation runs.
# It is retained separately because those historical runs predate the current
# per-layer-gate mainline class; see README.md for the reproduction note.
HSAFInterleavedReported = EST_EviTime

# Paper-facing short aliases used in tables.
HSAFStg = HSAFStaged
HSAFInt = HSAFInterleaved


class HSAFStagedStateInput(RankMixer_StateFeature):
    """Numeric state-as-input control for the staged architecture."""


class HSAFInterleavedStateInput(EST_StateFeature):
    """Numeric state-as-input control for the interleaved architecture."""


__all__ = [
    "HSAFStaged",
    "HSAFInterleaved",
    "HSAFStg",
    "HSAFInt",
    "HSAFStagedStateInput",
    "HSAFInterleavedStateInput",
    "HSAFInterleavedReported",
]
