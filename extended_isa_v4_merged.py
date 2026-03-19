"""
Backwards-compatibility shim. The canonical implementation is in loom_v1_merged.py.
"""
# ruff: noqa: F401
from loom_v1_merged import *  # noqa: F403
from loom_v1_merged import (
    LoomComputerMerged as ExtendedNeuralComputerV4Merged,
)
