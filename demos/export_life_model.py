"""
Export a larger ONNX model for Game of Life (n=2048, d_model=146).

The standard model (n=1024, d_model=137) cannot hold the 1,284 instructions
that Game of Life compiles to. This script builds a larger neural computer
with the same 11-layer, 13-opcode ISA but a wider state tensor (146x2048),
exports it to ONNX, and re-compiles the Game of Life program for this config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extended_isa_v4 import (
    ExtendedNeuralComputerV4, ExtendedConfigV4,
    init_state_v4, read_memory_v4, get_pc_v4,
    OP_HALT, OP_INC
)
from c_compiler import compile_c
from snake_game.export_onnx import ExtendedISAModule


LIFE_CONFIG = dict(s=32, m=64, n=2048, N=8)

GAMEOFLIFE_SOURCE = """\
int main() {
    int g[16];
    int i = 0;
    while (i < 16) { g[i] = 0; i += 1; }
    g[5] = 1; g[6] = 1; g[7] = 1;

    int ng[16];
    int idx = 0;
    int r = 0;
    while (r < 4) {
        int c = 0;
        while (c < 4) {
            int neighbors = 0;
            int above = idx - 4;
            int below = idx + 4;

            if (r > 0 && c > 0) { if (g[above - 1] != 0) { neighbors += 1; } }
            if (r > 0)          { if (g[above] != 0)     { neighbors += 1; } }
            if (r > 0 && c < 3) { if (g[above + 1] != 0) { neighbors += 1; } }
            if (c > 0)          { if (g[idx - 1] != 0)   { neighbors += 1; } }
            if (c < 3)          { if (g[idx + 1] != 0)   { neighbors += 1; } }
            if (r < 3 && c > 0) { if (g[below - 1] != 0) { neighbors += 1; } }
            if (r < 3)          { if (g[below] != 0)     { neighbors += 1; } }
            if (r < 3 && c < 3) { if (g[below + 1] != 0) { neighbors += 1; } }

            int alive = g[idx];
            if (alive != 0) {
                if (neighbors == 2 || neighbors == 3) { ng[idx] = 1; } else { ng[idx] = 0; }
            } else {
                if (neighbors == 3) { ng[idx] = 1; } else { ng[idx] = 0; }
            }

            c += 1;
            idx += 1;
        }
        r += 1;
    }
    return ng[2];
}
"""


def main():
    print("=" * 60)
    print("Game of Life - Large Model Export")
    print("=" * 60)

    cfg = ExtendedConfigV4(**LIFE_CONFIG)
    print(f"Config: n={cfg.n}, m={cfg.m}, d_model={cfg.d_model}, logn={cfg.logn}")
    print(f"Instruction slots: {cfg.n - cfg.s - cfg.m}")

    # Compile Game of Life
    print("\nCompiling Game of Life...")
    cfg_c, memory, commands, meta = compile_c(GAMEOFLIFE_SOURCE, **LIFE_CONFIG)
    print(f"  {meta['num_instructions']} instructions")
    print(f"  {len(meta['variables'])} variables")

    # Count opcodes
    OPCODE_NAMES = {0:"HALT",1:"MOV",2:"ADD",3:"JMP",4:"JZ",5:"JNZ",6:"INC",7:"DEC",
        8:"SHL",9:"SHR",10:"CMP",11:"LOAD",12:"AND",13:"OR",14:"XOR",15:"SUB",
        16:"FIND",17:"SWAP",18:"CMOV",19:"MULACC"}
    opcode_counts = {}
    for op, b, c in commands:
        name = OPCODE_NAMES.get(op, f"OP_{op}")
        opcode_counts[name] = opcode_counts.get(name, 0) + 1
    non_halt = {k: v for k, v in sorted(opcode_counts.items()) if k != "HALT"}
    print(f"  Opcodes: {', '.join(f'{k}:{v}' for k, v in non_halt.items())}")

    # Build model
    print("\nBuilding neural computer (11 layers)...")
    model = ExtendedISAModule(cfg)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    print(f"  Parameters: {total_params + total_buffers:,}")

    # Quick sanity test
    print("\nSanity test (INC)...")
    test_cfg = ExtendedConfigV4(**LIFE_CONFIG)
    test_mem = [5] + [0] * (cfg.m - 1)
    test_cmds = [(OP_INC, cfg.s + 0, 0), (OP_HALT, 0, 0)]
    test_X = init_state_v4(test_cfg, test_mem, test_cmds)
    with torch.no_grad():
        test_Y = model(test_X)
    test_result = read_memory_v4(test_Y, test_cfg)
    assert test_result[0] == 6, f"INC test failed: got {test_result[0]}"
    print("  INC 5 -> 6: PASS")

    # Export ONNX
    output_dir = os.path.join(os.path.dirname(__file__), "programs")
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "life.onnx")

    print(f"\nExporting ONNX to {onnx_path}...")
    os.environ['PYTORCH_ONNX_USE_LEGACY_EXPORT'] = '1'

    X = init_state_v4(cfg_c, memory, commands)

    torch.onnx.export(
        model, X, onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['new_state'],
        verbose=False
    )

    # Consolidate external data into single file (ONNX Runtime Web
    # cannot load .data sidecar files via MountedFiles)
    import onnx
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        print("  Consolidating external data into single ONNX file...")
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
        os.remove(data_path)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Saved: {size_mb:.1f} MB")

    # Export state and info
    cmd_start = cfg_c.s + cfg_c.m
    instr_to_line = {}
    for instr_idx, src_line in meta['source_map']:
        instr_to_line[cmd_start + instr_idx] = src_line

    state_path = os.path.join(output_dir, "gameoflife_state.json")
    with open(state_path, 'w') as f:
        json.dump({"state": X.tolist()}, f)
    state_kb = os.path.getsize(state_path) / 1024
    print(f"  State: {state_kb:.0f} KB")

    info = {
        "name": "gameoflife",
        "title": "Game of Life",
        "description": "One generation of Conway's Game of Life on a 4x4 grid",
        "source": GAMEOFLIFE_SOURCE,
        "variables": meta["variables"],
        "num_instructions": meta["num_instructions"],
        "opcode_counts": opcode_counts,
        "source_map": instr_to_line,
        "retval_addr": meta["retval_addr"],
        "output_addr": meta["output_addr"],
        "expected": {},
        "config": {
            "s": cfg_c.s,
            "m": cfg_c.m,
            "n": cfg_c.n,
            "N": cfg_c.N,
            "d_model": cfg_c.d_model,
            "logn": cfg_c.logn,
            "idx_memory": cfg_c.idx_memory,
            "idx_pc": cfg_c.idx_pc,
            "idx_pos_enc": cfg_c.idx_pos_enc,
            "idx_tag": cfg_c.idx_tag,
        },
    }

    info_path = os.path.join(output_dir, "gameoflife_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Info: gameoflife_info.json")

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        result = session.run(None, {'state': X.numpy()})
        X_onnx = torch.from_numpy(result[0])
        X_torch = model(X)
        diff = torch.abs(X_onnx - X_torch).max().item()
        print(f"\nONNX verification: max diff = {diff:.6f}")
    except Exception as e:
        print(f"\nONNX verification skipped: {e}")

    print(f"\nDone! Game of Life model: {cfg.d_model}x{cfg.n} state tensor, {total_params + total_buffers:,} params")


if __name__ == "__main__":
    main()
