"""
Extended ISA V4 - Comprehensive Test Suite
==========================================

Tests all opcodes with detailed debugging output.
"""

import torch
import numpy as np
from extended_isa_v4 import (
    ExtendedNeuralComputerV4, ExtendedConfigV4,
    init_state_v4, read_memory_v4, get_pc_v4,
    opcode_pattern, create_bipolar_value,
    OP_HALT, OP_MOV, OP_ADD, OP_JMP, OP_JZ, OP_JNZ, OP_INC, OP_DEC,
    OP_AND, OP_OR, OP_XOR, OP_SUB
)
from subleq import signed_from_bipolar, from_bipolar


def run_test(name, cfg, memory, commands, expected, max_steps=20, trace=False):
    """Run a single test."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    ext = ExtendedNeuralComputerV4(cfg)
    X = init_state_v4(cfg, memory, commands)
    
    if trace:
        print(f"Initial: PC={get_pc_v4(X, cfg)}, mem={read_memory_v4(X, cfg)[:4]}")
    
    with torch.no_grad():
        for step in range(max_steps):
            pc = get_pc_v4(X, cfg)
            if pc == 0:
                print(f"  HALT at step {step}")
                break
            if trace:
                print(f"  Step {step}: PC={pc}")
            X = ext.step(X)
            if trace:
                print(f"    -> mem={read_memory_v4(X, cfg)[:4]}")
    
    result = read_memory_v4(X, cfg)
    passed = result == expected
    
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    
    if not passed:
        for i, (r, e) in enumerate(zip(result, expected)):
            if r != e:
                print(f"  mem[{i}]: got {r}, expected {e}")
    
    return passed


def debug_layer2(name, cfg, memory, commands):
    """Debug Layer 2 output for a test case."""
    print(f"\n{'='*60}")
    print(f"DEBUG Layer 2: {name}")
    print(f"{'='*60}")
    
    ext = ExtendedNeuralComputerV4(cfg)
    X = init_state_v4(cfg, memory, commands)
    
    # Layer 1
    with torch.no_grad():
        X1 = ext.layers[0].forward(X)
    
    addr_a = cfg.idx_scratch_cmd
    print(f"Opcode: {from_bipolar(X1[addr_a:addr_a+cfg.logn, 0])}")
    print(f"Opcode bits: {X1[addr_a:addr_a+cfg.logn, 0].tolist()}")
    
    # Attention output
    layer2 = ext.layers[1]
    with torch.no_grad():
        scores1 = X1.T @ layer2.K1.T @ layer2.Q1 @ X1
        scores2 = X1.T @ layer2.K2.T @ layer2.Q2 @ X1
        scores3 = X1.T @ layer2.K3.T @ layer2.Q3 @ X1
        attn1 = torch.nn.functional.softmax(layer2.lam * scores1, dim=0)
        attn2 = torch.nn.functional.softmax(layer2.lam * scores2, dim=0)
        attn3 = torch.nn.functional.softmax(layer2.lam * scores3, dim=0)
        attn_out = X1 + layer2.V1 @ X1 @ attn1 + layer2.V2 @ X1 @ attn2 + layer2.V3 @ X1 @ attn3
    
    buf_a = cfg.idx_buffer
    buf_b = cfg.idx_buffer + cfg.N
    buf_c = cfg.idx_buffer + 2*cfg.N
    
    print(f"\nAfter attention:")
    print(f"  buf_a: {signed_from_bipolar(attn_out[buf_a:buf_a+cfg.N, 0])}")
    print(f"  buf_b: {signed_from_bipolar(attn_out[buf_b:buf_b+cfg.N, 0])}")
    print(f"  buf_c: {signed_from_bipolar(attn_out[buf_c:buf_c+cfg.N, 0])}")
    
    # Full Layer 2
    with torch.no_grad():
        X2 = ext.layers[1].forward(X1)
    
    scr_sub = cfg.idx_scratchpad
    scr_min = cfg.idx_scratchpad + cfg.N
    
    print(f"\nAfter FFN:")
    print(f"  scr_sub: {X2[scr_sub:scr_sub+cfg.N, 0].tolist()}")
    print(f"  scr_sub decoded: {signed_from_bipolar(X2[scr_sub:scr_sub+cfg.N, 0])}")
    print(f"  scr_min: {X2[scr_min:scr_min+cfg.N, 0].tolist()}")
    print(f"  scr_min decoded: {signed_from_bipolar(X2[scr_min:scr_min+cfg.N, 0])}")


def run_all_tests():
    cfg = ExtendedConfigV4(s=32, m=8, n=64, N=8)
    
    passed = 0
    total = 0
    
    # ========== SUBLEQ Tests ==========
    
    total += 1
    if run_test("SUBLEQ countdown", cfg,
            [0, 1, 5, 0, 0, 0, 0, 0],
            [(cfg.s + 1, cfg.s + 2, 0), (cfg.s + 0, cfg.s + 0, cfg.s + cfg.m)],
            [0, 1, 0, 0, 0, 0, 0, 0], trace=True):
        passed += 1
    
    # ========== INC/DEC Tests ==========
    
    total += 1
    if run_test("INC x3", cfg,
            [5, 0, 0, 0, 0, 0, 0, 0],
            [(OP_INC, cfg.s + 0, 0), (OP_INC, cfg.s + 0, 0), (OP_INC, cfg.s + 0, 0), (OP_HALT, 0, 0)],
            [8, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("DEC x3", cfg,
            [10, 0, 0, 0, 0, 0, 0, 0],
            [(OP_DEC, cfg.s + 0, 0), (OP_DEC, cfg.s + 0, 0), (OP_DEC, cfg.s + 0, 0), (OP_HALT, 0, 0)],
            [7, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # ========== Jump Tests ==========
    
    total += 1
    if run_test("JMP (skip INC)", cfg,
            [0, 0, 0, 0, 0, 0, 0, 0],
            [(OP_JMP, 0, cfg.s + cfg.m + 2), (OP_INC, cfg.s + 0, 0), (OP_HALT, 0, 0)],
            [0, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("JZ zero (branch)", cfg,
            [0, 10, 0, 0, 0, 0, 0, 0],
            [(OP_JZ, cfg.s + 0, cfg.s + cfg.m + 2), (OP_INC, cfg.s + 1, 0), (OP_HALT, 0, 0)],
            [0, 10, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("JZ non-zero (no branch)", cfg,
            [5, 10, 0, 0, 0, 0, 0, 0],
            [(OP_JZ, cfg.s + 0, cfg.s + cfg.m + 2), (OP_INC, cfg.s + 1, 0), (OP_HALT, 0, 0)],
            [5, 11, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("JNZ non-zero (branch)", cfg,
            [5, 10, 0, 0, 0, 0, 0, 0],
            [(OP_JNZ, cfg.s + 0, cfg.s + cfg.m + 2), (OP_INC, cfg.s + 1, 0), (OP_HALT, 0, 0)],
            [5, 10, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("JNZ zero (no branch)", cfg,
            [0, 10, 0, 0, 0, 0, 0, 0],
            [(OP_JNZ, cfg.s + 0, cfg.s + cfg.m + 2), (OP_INC, cfg.s + 1, 0), (OP_HALT, 0, 0)],
            [0, 11, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # ========== Loop Test ==========
    
    total += 1
    if run_test("Loop countdown", cfg,
            [3, 0, 0, 0, 0, 0, 0, 0],
            [(OP_DEC, cfg.s + 0, 0), (OP_INC, cfg.s + 1, 0), (OP_JNZ, cfg.s + 0, cfg.s + cfg.m), (OP_HALT, 0, 0)],
            [0, 3, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # ========== Edge Cases ==========
    
    total += 1
    if run_test("Negative values", cfg,
            [1, 0, 0, 0, 0, 0, 0, 0],
            [(OP_DEC, cfg.s + 0, 0), (OP_DEC, cfg.s + 0, 0), (OP_DEC, cfg.s + 0, 0), (OP_HALT, 0, 0)],
            [-2, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("Overflow 127+1", cfg,
            [127, 0, 0, 0, 0, 0, 0, 0],
            [(OP_INC, cfg.s + 0, 0), (OP_HALT, 0, 0)],
            [-128, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("Underflow -128-1", cfg,
            [-128, 0, 0, 0, 0, 0, 0, 0],
            [(OP_DEC, cfg.s + 0, 0), (OP_HALT, 0, 0)],
            [127, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("HALT immediate", cfg,
            [7, -3, 0, 0, 0, 0, 0, 0],
            [(OP_HALT, 0, 0), (OP_INC, cfg.s + 0, 0)],
            [7, -3, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # ========== MOV Test ==========
    
    print("\n" + "="*60)
    print("Debugging MOV operation")
    debug_layer2("MOV", cfg, [0, 42, 0, 0, 0, 0, 0, 0], [(OP_MOV, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)])
    
    total += 1
    if run_test("MOV", cfg,
            [0, 42, 0, 0, 0, 0, 0, 0],
            [(OP_MOV, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [42, 42, 0, 0, 0, 0, 0, 0], trace=True):
        passed += 1
    
    # ========== ADD Test ==========
    
    print("\n" + "="*60)
    print("Debugging ADD operation")
    debug_layer2("ADD", cfg, [10, 5, 0, 0, 0, 0, 0, 0], [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)])
    
    total += 1
    if run_test("ADD", cfg,
            [10, 5, 0, 0, 0, 0, 0, 0],
            [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [15, 5, 0, 0, 0, 0, 0, 0], trace=True):
        passed += 1
    
    # ========== SUB Test ==========
    
    print("\n" + "="*60)
    print("Debugging SUB operation")
    debug_layer2("SUB", cfg, [10, 3, 0, 0, 0, 0, 0, 0], [(OP_SUB, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)])
    
    total += 1
    if run_test("SUB", cfg,
            [10, 3, 0, 0, 0, 0, 0, 0],
            [(OP_SUB, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [7, 3, 0, 0, 0, 0, 0, 0], trace=True):
        passed += 1
    
    # ========== Simple XOR Test ==========
    
    total += 1
    if run_test("XOR (5^3=6)", cfg,
            [5, 3, 0, 0, 0, 0, 0, 0],
            [(OP_XOR, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [6, 3, 0, 0, 0, 0, 0, 0], trace=True):
        passed += 1
    
    # ========== Additional ADD Tests ==========
    
    total += 1
    if run_test("ADD (3+7=10)", cfg,
            [3, 7, 0, 0, 0, 0, 0, 0],
            [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [10, 7, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("ADD (1+1=2)", cfg,
            [1, 1, 0, 0, 0, 0, 0, 0],
            [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [2, 1, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("ADD (100+27=127)", cfg,
            [100, 27, 0, 0, 0, 0, 0, 0],
            [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [127, 27, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("ADD negative (-5+3=-2)", cfg,
            [-5, 3, 0, 0, 0, 0, 0, 0],
            [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [-2, 3, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("ADD overflow (100+100=-56)", cfg,
            [100, 100, 0, 0, 0, 0, 0, 0],
            [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [-56, 100, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("ADD double negative (-50+-80=126)", cfg,
            [-50, -80, 0, 0, 0, 0, 0, 0],
            [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [126, -80, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # ========== Additional SUB Tests ==========
    
    total += 1
    if run_test("SUB (20-7=13)", cfg,
            [20, 7, 0, 0, 0, 0, 0, 0],
            [(OP_SUB, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [13, 7, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("SUB negative (5-10=-5)", cfg,
            [5, 10, 0, 0, 0, 0, 0, 0],
            [(OP_SUB, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [-5, 10, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("SUB overflow (80-(-80)=-96)", cfg,
            [80, -80, 0, 0, 0, 0, 0, 0],
            [(OP_SUB, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [-96, -80, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # ========== Combined Operations ==========
    
    total += 1
    if run_test("MOV then INC", cfg,
            [0, 10, 0, 0, 0, 0, 0, 0],
            [(OP_MOV, cfg.s + 0, cfg.s + 1), (OP_INC, cfg.s + 0, 0), (OP_HALT, 0, 0)],
            [11, 10, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("MOV self (no-op)", cfg,
            [33, 0, 0, 0, 0, 0, 0, 0],
            [(OP_MOV, cfg.s + 0, cfg.s + 0), (OP_HALT, 0, 0)],
            [33, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # ========== Bitwise Operations ==========
    
    # XOR tests
    total += 1
    if run_test("XOR (-16 ^ -86 = 90)", cfg,
            [-16, -86, 0, 0, 0, 0, 0, 0],
            [(OP_XOR, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [90, -86, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("XOR (0 ^ 127 = 127)", cfg,
            [0, 127, 0, 0, 0, 0, 0, 0],
            [(OP_XOR, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [127, 127, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("XOR (127 ^ 127 = 0)", cfg,
            [127, 127, 0, 0, 0, 0, 0, 0],
            [(OP_XOR, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [0, 127, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("XOR (85 ^ -1 = -86)", cfg,
            [85, -1, 0, 0, 0, 0, 0, 0],
            [(OP_XOR, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [-86, -1, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # AND tests
    total += 1
    if run_test("AND (-16 & -86 = -96)", cfg,
            [-16, -86, 0, 0, 0, 0, 0, 0],
            [(OP_AND, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [-96, -86, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("AND (127 & 0 = 0)", cfg,
            [127, 0, 0, 0, 0, 0, 0, 0],
            [(OP_AND, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [0, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("AND (15 & 7 = 7)", cfg,
            [15, 7, 0, 0, 0, 0, 0, 0],
            [(OP_AND, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [7, 7, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("AND (85 & -1 = 85)", cfg,
            [85, -1, 0, 0, 0, 0, 0, 0],
            [(OP_AND, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [85, -1, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # OR tests
    total += 1
    if run_test("OR (-16 | 15 = -1)", cfg,
            [-16, 15, 0, 0, 0, 0, 0, 0],
            [(OP_OR, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [-1, 15, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("OR (0 | 0 = 0)", cfg,
            [0, 0, 0, 0, 0, 0, 0, 0],
            [(OP_OR, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [0, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    total += 1
    if run_test("OR (8 | 4 = 12)", cfg,
            [8, 4, 0, 0, 0, 0, 0, 0],
            [(OP_OR, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [12, 4, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("OR (85 | -1 = -1)", cfg,
            [85, -1, 0, 0, 0, 0, 0, 0],
            [(OP_OR, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)],
            [-1, -1, 0, 0, 0, 0, 0, 0]):
        passed += 1

    # ========== Branching Edge Cases ==========

    total += 1
    if run_test("JZ negative (no branch)", cfg,
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [(OP_JZ, cfg.s + 0, cfg.s + cfg.m + 2), (OP_INC, cfg.s + 1, 0), (OP_HALT, 0, 0)],
            [-1, 1, 0, 0, 0, 0, 0, 0]):
        passed += 1

    total += 1
    if run_test("JNZ negative (branch)", cfg,
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [(OP_JNZ, cfg.s + 0, cfg.s + cfg.m + 2), (OP_INC, cfg.s + 1, 0), (OP_HALT, 0, 0)],
            [-1, 0, 0, 0, 0, 0, 0, 0]):
        passed += 1
    
    # ========== Summary ==========
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
