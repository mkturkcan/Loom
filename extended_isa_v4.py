"""
Backwards-compatibility shim. The canonical implementation is in loom_v1.py.
"""
# ruff: noqa: F401
from loom_v1 import *  # noqa: F403
from loom_v1 import (
    LoomConfig as ExtendedConfigV4,
    LoomComputer as ExtendedNeuralComputerV4,
    init_state as init_state_v4,
    read_memory as read_memory_v4,
    get_pc as get_pc_v4,
)
