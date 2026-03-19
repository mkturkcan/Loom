"""
Export demo programs for the Neural SUBLEQ web demos.

For each demo program:
  1. Compiles C source to Extended ISA via c_compiler
  2. Creates initial state tensor via init_state_v4
  3. Exports program_info.json with source, variables, source map
  4. Exports initial_state.json with the state tensor

All demos share the same ONNX model (copied from snake_game/).
"""

import json
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from c_compiler import compile_c
from extended_isa_v4 import (
    ExtendedNeuralComputerV4, ExtendedConfigV4,
    init_state_v4, read_memory_v4, get_pc_v4,
    OP_HALT, OP_MOV, OP_ADD, OP_JMP, OP_JZ, OP_JNZ,
    OP_INC, OP_DEC, OP_SHL, OP_SHR, OP_CMP, OP_LOAD,
    OP_AND, OP_OR, OP_XOR, OP_SUB, OP_FIND, OP_SWAP,
    OP_CMOV, OP_MULACC,
)

OPCODE_NAMES = {
    OP_HALT: "HALT", OP_MOV: "MOV", OP_ADD: "ADD", OP_JMP: "JMP",
    OP_JZ: "JZ", OP_JNZ: "JNZ", OP_INC: "INC", OP_DEC: "DEC",
    OP_SHL: "SHL", OP_SHR: "SHR", OP_CMP: "CMP", OP_LOAD: "LOAD",
    OP_AND: "AND", OP_OR: "OR", OP_XOR: "XOR", OP_SUB: "SUB",
    OP_FIND: "FIND", OP_SWAP: "SWAP", OP_CMOV: "CMOV", OP_MULACC: "MULACC",
}
from snake_game.export_onnx import ExtendedISAModule

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Demo Programs
# ============================================================

PROGRAMS = {
    # --- Debugger demos ---
    "fibonacci": {
        "title": "Fibonacci",
        "description": "Compute the 7th Fibonacci number",
        "source": """\
int main() {
    int a = 0;
    int b = 1;
    int n = 7;
    int i = 0;
    while (i < n) {
        int t = a + b;
        a = b;
        b = t;
        i += 1;
    }
    return a;
}
""",
        "expected": {"RETVAL": 13},
    },

    "gcd": {
        "title": "GCD (Euclid)",
        "description": "Greatest common divisor of 48 and 18",
        "source": """\
int main() {
    int a = 48;
    int b = 18;
    while (b != 0) {
        int t = a;
        while (t >= b) {
            t = t - b;
        }
        a = b;
        b = t;
    }
    return a;
}
""",
        "expected": {"RETVAL": 6},
    },

    "primes": {
        "title": "Prime Counter",
        "description": "Count primes up to 10",
        "source": """\
int is_prime(int p) {
    int d = 2;
    while (d < p) {
        int rem = p;
        while (rem >= d) {
            rem = rem - d;
        }
        if (rem == 0) {
            return 0;
        }
        d += 1;
    }
    return 1;
}

int main() {
    int count = 0;
    int p = 2;
    while (p <= 10) {
        if (is_prime(p)) {
            count += 1;
        }
        p += 1;
    }
    return count;
}
""",
        "expected": {"RETVAL": 4},
    },

    "collatz": {
        "title": "Collatz Steps",
        "description": "Count Collatz steps for n=6 (6->3->10->5->16->8->4->2->1 = 8 steps)",
        "source": """\
int main() {
    int n = 6;
    int steps = 0;
    while (n != 1) {
        int odd = n & 1;
        if (odd != 0) {
            n = n + n + n + 1;
        } else {
            int half = 0;
            int test = 0;
            while (test + test < n) {
                half += 1;
                test = half;
            }
            n = half;
        }
        steps += 1;
    }
    return steps;
}
""",
        "expected": {"RETVAL": 8},
    },

    # --- Sorting demos ---
    "bubblesort": {
        "title": "Bubble Sort",
        "description": "Sort 6 numbers using bubble sort",
        "source": """\
int main() {
    int a[6];
    a[0] = 34;
    a[1] = 12;
    a[2] = 45;
    a[3] = 7;
    a[4] = 23;
    a[5] = 18;
    int n = 6;
    int i = 0;
    while (i < n) {
        int j = 0;
        while (j < n - 1 - i) {
            int aj = a[j];
            int aj1 = a[j + 1];
            if (aj > aj1) {
                a[j] = aj1;
                a[j + 1] = aj;
            }
            j += 1;
        }
        i += 1;
    }
    return a[0];
}
""",
        "expected": {"RETVAL": 7},
    },

    # Sudoku 9x9 is exported separately by export_sudoku_model.py
    # (requires larger computer: n=2048, m=224, d_model=164, 1166 instructions)

    # Game of Life is exported separately by export_life_model.py
    # (requires larger computer: n=2048, d_model=164, 570 instructions)
    # Wolfenstein is exported separately by export_wolfenstein.py
    # (uses n=1024 but has its own export script for multi-tick testing)
}


def export_program(name, prog, output_dir):
    """Compile a C program and export state + metadata."""
    print(f"  Compiling {name}...")

    source = prog["source"]
    cfg, memory, commands, meta = compile_c(source, s=32, m=64, n=1024, N=8)

    # Count opcodes
    opcode_counts = {}
    for op, b, c in commands:
        op_name = OPCODE_NAMES.get(op, f"OP_{op}")
        opcode_counts[op_name] = opcode_counts.get(op_name, 0) + 1
    # Print opcode histogram
    non_halt = {k: v for k, v in sorted(opcode_counts.items()) if k != "HALT"}
    hist = ", ".join(f"{k}:{v}" for k, v in non_halt.items())
    print(f"    Opcodes: {hist}")

    # Build instruction-to-source-line lookup
    # source_map: [(instr_index, source_line), ...]
    # Build a dict: instr_index -> source_line for the ISA instruction address space
    cmd_start = cfg.s + cfg.m  # = 80
    instr_to_line = {}
    for instr_idx, src_line in meta['source_map']:
        instr_to_line[cmd_start + instr_idx] = src_line

    # Create initial state
    X = init_state_v4(cfg, memory, commands)

    # Save initial state
    state_path = os.path.join(output_dir, f"{name}_state.json")
    with open(state_path, 'w') as f:
        json.dump({"state": X.tolist()}, f)
    size_kb = os.path.getsize(state_path) / 1024
    print(f"    State: {size_kb:.0f} KB")

    # Save program info
    info = {
        "name": name,
        "title": prog["title"],
        "description": prog["description"],
        "source": source,
        "variables": meta["variables"],
        "num_instructions": meta["num_instructions"],
        "opcode_counts": opcode_counts,
        "source_map": instr_to_line,  # PC address -> source line
        "retval_addr": meta["retval_addr"],
        "output_addr": meta["output_addr"],
        "expected": prog.get("expected", {}),
        "config": {
            "s": cfg.s,
            "m": cfg.m,
            "n": cfg.n,
            "N": cfg.N,
            "d_model": cfg.d_model,
            "logn": cfg.logn,
            "idx_memory": cfg.idx_memory,
            "idx_pc": cfg.idx_pc,
            "idx_pos_enc": cfg.idx_pos_enc,
            "idx_tag": cfg.idx_tag,
        },
    }

    info_path = os.path.join(output_dir, f"{name}_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"    Info: {name}_info.json ({meta['num_instructions']} instructions)")


def main():
    print("=" * 60)
    print("Neural SUBLEQ — Demo Program Export")
    print("=" * 60)

    # Create output directory
    programs_dir = os.path.join(DEMO_DIR, "programs")
    os.makedirs(programs_dir, exist_ok=True)

    # Export ONNX model for demos (s=32, n=1024)
    demo_cfg = ExtendedConfigV4(s=32, m=64, n=1024, N=8)
    print(f"  Building model: d_model={demo_cfg.d_model}, n={demo_cfg.n}")
    model = ExtendedISAModule(demo_cfg)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Sanity test
    test_mem = [5] + [0] * (demo_cfg.m - 1)
    test_cmds = [(OP_INC, demo_cfg.s + 0, 0), (OP_HALT, 0, 0)]
    test_X = init_state_v4(demo_cfg, test_mem, test_cmds)
    with torch.no_grad():
        test_Y = model(test_X)
    test_result = read_memory_v4(test_Y, demo_cfg)
    assert test_result[0] == 6, f"INC test failed: got {test_result[0]}"
    print("  INC 5 -> 6: PASS")

    onnx_path = os.path.join(programs_dir, "snake.onnx")
    os.environ['PYTORCH_ONNX_USE_LEGACY_EXPORT'] = '1'
    dummy_X = init_state_v4(demo_cfg, [0] * demo_cfg.m,
                            [(OP_HALT, 0, 0)])
    torch.onnx.export(
        model, dummy_X, onnx_path,
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['state'], output_names=['new_state'],
        verbose=False,
    )
    # Consolidate external data into single file
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        import onnx
        print("  Consolidating external data...")
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
        os.remove(data_path)
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Exported snake.onnx ({size_mb:.1f} MB)")

    # Copy Python source files for REPL (Pyodide)
    parent_dir = os.path.dirname(DEMO_DIR)
    for f in ["subleq.py", "extended_isa_v4.py", "c_compiler.py"]:
        src = os.path.join(parent_dir, f)
        dst = os.path.join(programs_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {f} (for REPL)")

    print()

    # Export each program
    for name, prog in PROGRAMS.items():
        try:
            export_program(name, prog, programs_dir)
        except Exception as e:
            print(f"  ERROR exporting {name}: {e}")
            import traceback
            traceback.print_exc()

    # Write manifest
    manifest = {name: {
        "title": prog["title"],
        "description": prog["description"],
        "state_file": f"{name}_state.json",
        "info_file": f"{name}_info.json",
    } for name, prog in PROGRAMS.items()}

    manifest_path = os.path.join(programs_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExported {len(PROGRAMS)} programs to {programs_dir}/")
    print("Manifest: manifest.json")


if __name__ == "__main__":
    main()
