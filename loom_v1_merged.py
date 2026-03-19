"""
Extended ISA V4 Merged — 7-Layer Optimized Architecture
========================================================

**DEPRECATED**: Incompatible with dim=0 softmax. L3's attention heads have
zero Q-projection at column 0; with dim=0 this produces uniform attention
noise that the merged L3+L4 FFN amplifies into bit errors. The standard
8-layer architecture (extended_isa_v4.py) works correctly with dim=0.

Collapses 8 V4 layers into 7 by merging L3+L4:
  L1:       Fetch instruction (attention, 1 head)
  L2:       Read memory + decode (attention, 3 heads)
  L3+L4:   Indirect read/correct + subtract (attention 2 heads, expanded FFN)
  L5:       Write memory (attention, 2 heads)
  L6:       Branch flag + PC+1 (FFN only)
  L7:       Branch select (FFN only)
  L8:       Error correction (FFN only)

The L3+L4 merge works because L4 has identity attention (Q=K=V=0), so its
FFN can be appended to L3's FFN. Both operate on L3's attention output.

Further merges (L7+L8, L6+L7+L8) fail due to sequential dependencies:
L7 modifies PC rows that L8 reads, and L6 writes a flag that L7 reads.
"""

import torch
from typing import List
from subleq import TransformerLayer
from loom_v1 import (
    LoomComputer, LoomConfig,
    init_state, read_memory, get_pc,
    OP_HALT, OP_MOV, OP_ADD, OP_JMP, OP_JZ, OP_JNZ, OP_INC, OP_DEC,
    OP_SHL, OP_SHR, OP_CMP, OP_LOAD, OP_AND, OP_OR, OP_XOR, OP_SUB,
    OP_FIND, OP_SWAP, OP_CMOV, OP_MULACC,
)


class LoomComputerMerged(LoomComputer):
    """7-layer V4: merges L3+L4 (indirect read + subtract)."""

    def __init__(self, cfg: LoomConfig):
        super().__init__(cfg)
        L1, L2, L3, L4, L5, L6, L7, L8 = self.layers
        self.layers = [L1, L2, self._merge_ffn(L3, L4), L5, L6, L7, L8]

    @staticmethod
    def _merge_ffn(layer_a, layer_b):
        """Append layer_b's FFN rows to layer_a. Keep layer_a's attention."""
        cfg = layer_a.cfg
        wa, wb = layer_a.W1.shape[0], layer_b.W1.shape[0]
        heads = layer_a.num_heads
        qr = layer_a.Q1.shape[0] if heads > 1 else layer_a.Q.shape[0]

        m = TransformerLayer(cfg, qr, wa + wb, num_heads=heads)

        # Copy attention from layer_a
        for attr in ['Q', 'K', 'V', 'Q1', 'K1', 'V1', 'Q2', 'K2', 'V2', 'Q3', 'K3', 'V3']:
            if hasattr(layer_a, attr) and getattr(layer_a, attr) is not None:
                setattr(m, attr, getattr(layer_a, attr).clone())

        # Concatenate FFN rows
        m.W1[:wa] = layer_a.W1
        m.W1[wa:] = layer_b.W1
        m.b1[:wa] = layer_a.b1
        m.b1[wa:] = layer_b.b1
        m.W2[:, :wa] = layer_a.W2
        m.W2[:, wa:] = layer_b.W2
        m.b2 = layer_a.b2 + layer_b.b2

        return m


# Backwards-compatibility aliases
ExtendedNeuralComputerV4Merged = LoomComputerMerged


if __name__ == "__main__":
    import sys

    def run_test(name, comp, cfg, memory, commands, expected, max_steps=20):
        X = init_state(cfg, memory, commands)
        with torch.no_grad():
            for step in range(max_steps):
                if get_pc(X, cfg) == 0: break
                X = comp.step(X)
        result = read_memory(X, cfg)
        ok = result == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={result[:len(expected)]}")
        return ok

    cfg = LoomConfig(s=32, m=8, n=64, N=8)
    s = cfg.s
    merged = LoomComputerMerged(cfg)

    print(f"Merged: {len(merged.layers)} layers (down from 8)")
    for i, layer in enumerate(merged.layers):
        print(f"  L{i+1}: w1={layer.W1.shape[0]:4d}  heads={layer.num_heads}")

    print("\nTests:")
    ok = True
    ok &= run_test("INC 5->6", merged, cfg, [5,0,0,0,0,0,0,0], [(OP_INC,s,0),(OP_HALT,0,0)], [6,0,0,0,0,0,0,0])
    ok &= run_test("DEC 5->4", merged, cfg, [5,0,0,0,0,0,0,0], [(OP_DEC,s,0),(OP_HALT,0,0)], [4,0,0,0,0,0,0,0])
    ok &= run_test("MOV", merged, cfg, [0,7,0,0,0,0,0,0], [(OP_MOV,s,s+1),(OP_HALT,0,0)], [7,7,0,0,0,0,0,0])
    ok &= run_test("ADD", merged, cfg, [3,5,0,0,0,0,0,0], [(OP_ADD,s,s+1),(OP_HALT,0,0)], [8,5,0,0,0,0,0,0])
    ok &= run_test("SUB", merged, cfg, [8,3,0,0,0,0,0,0], [(OP_SUB,s,s+1),(OP_HALT,0,0)], [5,3,0,0,0,0,0,0])
    ok &= run_test("AND", merged, cfg, [12,10,0,0,0,0,0,0], [(OP_AND,s,s+1),(OP_HALT,0,0)], [8,10,0,0,0,0,0,0])
    ok &= run_test("OR", merged, cfg, [12,10,0,0,0,0,0,0], [(OP_OR,s,s+1),(OP_HALT,0,0)], [14,10,0,0,0,0,0,0])
    ok &= run_test("XOR", merged, cfg, [12,10,0,0,0,0,0,0], [(OP_XOR,s,s+1),(OP_HALT,0,0)], [6,10,0,0,0,0,0,0])
    ok &= run_test("SHL", merged, cfg, [0,5,0,0,0,0,0,0], [(OP_SHL,s+1,0),(OP_HALT,0,0)], [0,10,0,0,0,0,0,0])
    ok &= run_test("SHR", merged, cfg, [0,10,0,0,0,0,0,0], [(OP_SHR,s+1,0),(OP_HALT,0,0)], [0,5,0,0,0,0,0,0])
    ok &= run_test("SWAP", merged, cfg, [3,7,0,0,0,0,0,0], [(OP_SWAP,s,s+1),(OP_HALT,0,0)], [7,3,0,0,0,0,0,0])
    ok &= run_test("JMP", merged, cfg, [10,0,0,0,0,0,0,0], [(OP_JMP,0,s+8+2),(OP_INC,s,0),(OP_HALT,0,0)], [10,0,0,0,0,0,0,0])
    ok &= run_test("JZ taken", merged, cfg, [0,99,0,0,0,0,0,0], [(OP_JZ,s,s+8+2),(OP_INC,s+1,0),(OP_HALT,0,0)], [0,99,0,0,0,0,0,0])
    ok &= run_test("JZ fall", merged, cfg, [5,0,0,0,0,0,0,0], [(OP_JZ,s,s+8+3),(OP_INC,s+1,0),(OP_HALT,0,0)], [5,1,0,0,0,0,0,0])
    ok &= run_test("JNZ taken", merged, cfg, [5,0,0,0,0,0,0,0], [(OP_JNZ,s,s+8+2),(OP_INC,s+1,0),(OP_HALT,0,0)], [5,0,0,0,0,0,0,0])
    ok &= run_test("CMP neg", merged, cfg, [-1,0,0,0,0,0,0,0], [(OP_CMP,s,s+8+2),(OP_INC,s+1,0),(OP_HALT,0,0)], [-1,0,0,0,0,0,0,0])
    ok &= run_test("CMP pos", merged, cfg, [5,0,0,0,0,0,0,0], [(OP_CMP,s,s+8+3),(OP_INC,s+1,0),(OP_HALT,0,0)], [5,1,0,0,0,0,0,0])
    ok &= run_test("CMOV taken", merged, cfg, [-1,42,0,0,0,0,0,0], [(OP_CMOV,s,s+1),(OP_HALT,0,0)], [42,42,0,0,0,0,0,0])
    ok &= run_test("INC loop 3x", merged, cfg, [0,3,0,0,0,0,0,0],
                    [(OP_INC,s,0),(OP_DEC,s+1,0),(OP_JNZ,s+1,s+8),(OP_HALT,0,0)], [3,0,0,0,0,0,0,0], 50)
    ok &= run_test("MOV+ADD", merged, cfg, [0,3,5,0,0,0,0,0],
                    [(OP_MOV,s,s+1),(OP_ADD,s,s+2),(OP_HALT,0,0)], [8,3,5,0,0,0,0,0])

    print(f"\n{'ALL PASSED' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)
