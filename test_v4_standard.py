"""
Test suite for V4 Standard — compares against V4 original for all opcodes.
Runs both softmax (dim=0) and argmax modes.
"""

import torch
import sys
from extended_isa_v4_standard import (
    ExtendedNeuralComputerV4Standard, StandardConfig,
    init_state_standard, read_memory_standard, get_pc_standard,
    OP_HALT, OP_MOV, OP_ADD, OP_JMP, OP_JZ, OP_JNZ, OP_INC, OP_DEC,
    OP_SHL, OP_SHR, OP_CMP, OP_LOAD, OP_AND, OP_OR, OP_XOR, OP_SUB,
    OP_FIND, OP_SWAP, OP_CMOV, OP_MULACC,
)
from extended_isa_v4 import (
    ExtendedNeuralComputerV4, ExtendedConfigV4,
    init_state_v4, read_memory_v4, get_pc_v4,
)


def run_v4(cfg_v4, memory, commands, max_steps=20):
    """Run with original V4."""
    ext = ExtendedNeuralComputerV4(cfg_v4)
    X = init_state_v4(cfg_v4, memory, commands)
    with torch.no_grad():
        for _ in range(max_steps):
            pc = get_pc_v4(X, cfg_v4)
            if pc == 0:
                break
            X = ext.step(X)
    return read_memory_v4(X, cfg_v4)


def run_std(cfg_std, ext_std, memory, commands, max_steps=20):
    """Run with standard (softmax dim=0)."""
    X = init_state_standard(cfg_std, memory, commands)
    with torch.no_grad():
        for _ in range(max_steps):
            pc = get_pc_standard(X, cfg_std)
            if pc == 0:
                break
            X = ext_std.step(X)
    return read_memory_standard(X, cfg_std)


def run_argmax(cfg_std, memory, commands, max_steps=20):
    """Run with argmax attention."""
    ext = ExtendedNeuralComputerV4Standard(cfg_std, use_argmax=True)
    X = init_state_standard(cfg_std, memory, commands)
    with torch.no_grad():
        for _ in range(max_steps):
            pc = get_pc_standard(X, cfg_std)
            if pc == 0:
                break
            X = ext.step(X)
    return read_memory_standard(X, cfg_std)


def test_opcode(name, cfg_v4, cfg_std, ext_std, memory, commands, expected,
                max_steps=20, test_argmax=True):
    """Test one opcode: V4, standard, and optionally argmax."""
    s = cfg_v4.s
    m_count = len(expected)

    v4_result = run_v4(cfg_v4, memory, commands, max_steps)[:m_count]
    std_result = run_std(cfg_std, ext_std, memory, commands, max_steps)[:m_count]

    v4_ok = v4_result == expected
    std_ok = std_result == expected

    status = "PASS" if (v4_ok and std_ok) else "FAIL"
    print(f"  [{status}] {name}: expected={expected} v4={v4_result} std={std_result}", end="")

    if test_argmax:
        arg_result = run_argmax(cfg_std, memory, commands, max_steps)[:m_count]
        arg_ok = arg_result == expected
        print(f" argmax={arg_result}", end="")
        if not arg_ok:
            status = "FAIL(argmax)"
    print()

    return v4_ok and std_ok


def main():
    cfg_v4 = ExtendedConfigV4(s=32, m=8, n=64, N=8)
    cfg_std = StandardConfig(s=32, m=8, n=64, N=8)
    ext_std = ExtendedNeuralComputerV4Standard(cfg_std)
    s = cfg_v4.s

    tests_passed = 0
    tests_total = 0

    def T(name, memory, commands, expected, max_steps=20, test_argmax=True):
        nonlocal tests_passed, tests_total
        tests_total += 1
        if test_opcode(name, cfg_v4, cfg_std, ext_std, memory, commands,
                       expected, max_steps, test_argmax):
            tests_passed += 1

    print("=" * 60)
    print("V4 Standard Test Suite")
    print("=" * 60)

    # --- Basic SUBLEQ ---
    print("\n-- SUBLEQ --")
    T("SUBLEQ 5-3=2",
      [5, 3, 0, 0, 0, 0, 0, 0],
      [(s+1, s+0, 0)],  # mem[0] -= mem[1]; halt if <= 0
      [2, 3, 0, 0, 0, 0, 0, 0])

    T("SUBLEQ 5-3-3=-1 halt",
      [5, 3, 0, 0, 0, 0, 0, 0],
      [(s+1, s+0, 0),         # mem[0] -= mem[1]; if <= 0 goto 0 (halt)
       (s+1, s+0, 0)],        # second subtract
      [-1, 3, 0, 0, 0, 0, 0, 0],
      max_steps=5)

    # --- INC / DEC ---
    print("\n-- INC / DEC --")
    T("INC 5->6",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_INC, s+0, 0), (OP_HALT, 0, 0)],
      [6, 0, 0, 0, 0, 0, 0, 0])

    T("DEC 5->4",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_DEC, s+0, 0), (OP_HALT, 0, 0)],
      [4, 0, 0, 0, 0, 0, 0, 0])

    T("INC 0->1",
      [0, 0, 0, 0, 0, 0, 0, 0],
      [(OP_INC, s+0, 0), (OP_HALT, 0, 0)],
      [1, 0, 0, 0, 0, 0, 0, 0])

    T("DEC 0->-1",
      [0, 0, 0, 0, 0, 0, 0, 0],
      [(OP_DEC, s+0, 0), (OP_HALT, 0, 0)],
      [-1, 0, 0, 0, 0, 0, 0, 0])

    T("INC -1->0",
      [-1, 0, 0, 0, 0, 0, 0, 0],
      [(OP_INC, s+0, 0), (OP_HALT, 0, 0)],
      [0, 0, 0, 0, 0, 0, 0, 0])

    # --- MOV ---
    print("\n-- MOV --")
    T("MOV 0<-7",
      [0, 7, 0, 0, 0, 0, 0, 0],
      [(OP_MOV, s+0, s+1), (OP_HALT, 0, 0)],
      [7, 7, 0, 0, 0, 0, 0, 0])

    T("MOV 0<-(-3)",
      [0, -3, 0, 0, 0, 0, 0, 0],
      [(OP_MOV, s+0, s+1), (OP_HALT, 0, 0)],
      [-3, -3, 0, 0, 0, 0, 0, 0])

    # --- ADD / SUB ---
    print("\n-- ADD / SUB --")
    T("ADD 3+5=8",
      [3, 5, 0, 0, 0, 0, 0, 0],
      [(OP_ADD, s+0, s+1), (OP_HALT, 0, 0)],
      [8, 5, 0, 0, 0, 0, 0, 0])

    T("ADD 3+(-2)=1",
      [3, -2, 0, 0, 0, 0, 0, 0],
      [(OP_ADD, s+0, s+1), (OP_HALT, 0, 0)],
      [1, -2, 0, 0, 0, 0, 0, 0])

    T("SUB 8-3=5",
      [8, 3, 0, 0, 0, 0, 0, 0],
      [(OP_SUB, s+0, s+1), (OP_HALT, 0, 0)],
      [5, 3, 0, 0, 0, 0, 0, 0])

    # --- JMP / JZ / JNZ ---
    print("\n-- JMP / JZ / JNZ --")
    T("JMP skip",
      [10, 0, 0, 0, 0, 0, 0, 0],
      [(OP_JMP, 0, s+8+2),
       (OP_INC, s+0, 0),   # skipped
       (OP_HALT, 0, 0)],
      [10, 0, 0, 0, 0, 0, 0, 0])

    T("JZ taken (val=0)",
      [0, 99, 0, 0, 0, 0, 0, 0],
      [(OP_JZ, s+0, s+8+2),
       (OP_INC, s+1, 0),   # skipped
       (OP_HALT, 0, 0)],
      [0, 99, 0, 0, 0, 0, 0, 0])

    T("JZ not taken (val=5)",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_JZ, s+0, s+8+3),
       (OP_INC, s+1, 0),   # NOT skipped
       (OP_HALT, 0, 0)],
      [5, 1, 0, 0, 0, 0, 0, 0])

    T("JNZ taken (val=5)",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_JNZ, s+0, s+8+2),
       (OP_INC, s+1, 0),   # skipped
       (OP_HALT, 0, 0)],
      [5, 0, 0, 0, 0, 0, 0, 0])

    T("JNZ not taken (val=0)",
      [0, 0, 0, 0, 0, 0, 0, 0],
      [(OP_JNZ, s+0, s+8+3),
       (OP_INC, s+1, 0),   # NOT skipped
       (OP_HALT, 0, 0)],
      [0, 1, 0, 0, 0, 0, 0, 0])

    # --- Bitwise ---
    print("\n-- AND / OR / XOR --")
    T("AND 0b1100 & 0b1010 = 0b1000",
      [12, 10, 0, 0, 0, 0, 0, 0],
      [(OP_AND, s+0, s+1), (OP_HALT, 0, 0)],
      [8, 10, 0, 0, 0, 0, 0, 0])

    T("OR 0b1100 | 0b1010 = 0b1110",
      [12, 10, 0, 0, 0, 0, 0, 0],
      [(OP_OR, s+0, s+1), (OP_HALT, 0, 0)],
      [14, 10, 0, 0, 0, 0, 0, 0])

    T("XOR 0b1100 ^ 0b1010 = 0b0110",
      [12, 10, 0, 0, 0, 0, 0, 0],
      [(OP_XOR, s+0, s+1), (OP_HALT, 0, 0)],
      [6, 10, 0, 0, 0, 0, 0, 0])

    # --- SHL / SHR ---
    print("\n-- SHL / SHR --")
    T("SHL 5<<1=10",
      [0, 5, 0, 0, 0, 0, 0, 0],
      [(OP_SHL, s+1, 0), (OP_HALT, 0, 0)],
      [0, 10, 0, 0, 0, 0, 0, 0])

    T("SHR 10>>1=5",
      [0, 10, 0, 0, 0, 0, 0, 0],
      [(OP_SHR, s+1, 0), (OP_HALT, 0, 0)],
      [0, 5, 0, 0, 0, 0, 0, 0])

    # --- CMP ---
    print("\n-- CMP --")
    T("CMP neg branches",
      [-1, 0, 0, 0, 0, 0, 0, 0],
      [(OP_CMP, s+0, s+8+2),
       (OP_INC, s+1, 0),   # skipped
       (OP_HALT, 0, 0)],
      [-1, 0, 0, 0, 0, 0, 0, 0])

    T("CMP pos falls through",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_CMP, s+0, s+8+3),
       (OP_INC, s+1, 0),   # NOT skipped
       (OP_HALT, 0, 0)],
      [5, 1, 0, 0, 0, 0, 0, 0])

    # --- SWAP ---
    print("\n-- SWAP --")
    T("SWAP 3,7",
      [3, 7, 0, 0, 0, 0, 0, 0],
      [(OP_SWAP, s+0, s+1), (OP_HALT, 0, 0)],
      [7, 3, 0, 0, 0, 0, 0, 0])

    # --- CMOV ---
    print("\n-- CMOV --")
    T("CMOV taken (neg)",
      [-1, 42, 0, 0, 0, 0, 0, 0],
      [(OP_CMOV, s+0, s+1), (OP_HALT, 0, 0)],
      [42, 42, 0, 0, 0, 0, 0, 0])

    T("CMOV not taken (pos)",
      [5, 42, 0, 0, 0, 0, 0, 0],
      [(OP_CMOV, s+0, s+1), (OP_HALT, 0, 0)],
      [5, 42, 0, 0, 0, 0, 0, 0])

    # --- Multi-step programs ---
    print("\n-- Multi-step --")
    T("INC loop 3x",
      [0, 3, 0, 0, 0, 0, 0, 0],
      [(OP_INC, s+0, 0),
       (OP_DEC, s+1, 0),
       (OP_JNZ, s+1, s+8+0),
       (OP_HALT, 0, 0)],
      [3, 0, 0, 0, 0, 0, 0, 0],
      max_steps=50)

    T("MOV+ADD: a=b+c",
      [0, 3, 5, 0, 0, 0, 0, 0],
      [(OP_MOV, s+0, s+1),   # a = b
       (OP_ADD, s+0, s+2),   # a += c
       (OP_HALT, 0, 0)],
      [8, 3, 5, 0, 0, 0, 0, 0])

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Results: {tests_passed}/{tests_total} passed")
    if tests_passed == tests_total:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {tests_total - tests_passed}")
    print(f"{'='*60}")

    return tests_passed == tests_total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
