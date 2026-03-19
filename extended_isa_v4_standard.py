"""
Backwards-compatibility shim. The canonical implementation is in loom_v1_standard.py.
"""
# ruff: noqa: F401
from loom_v1_standard import *  # noqa: F403
from loom_v1_standard import (
    LoomComputerStandard as ExtendedStandardTransformerV4,
    LoomComputerStandard as ExtendedNeuralComputerV4Standard,
    LoomStandardConfig as StandardConfig,
)
