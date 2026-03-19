"""
Export a larger ONNX model for 9x9 Sudoku (m=224, n=2048).

The standard model (m=64, n=1024) cannot hold the 81-element grid array
and constants needed by the Sudoku solver. This script builds a neural
computer with m=224 memory slots and n=2048 columns, exports it to ONNX,
and compiles the Sudoku solver for this config.
"""

import torch
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


SUDOKU_CONFIG = dict(s=32, m=224, n=2048, N=8)

SUDOKU9_SOURCE = """\
int main() {
    int grid[81];
    int pos;
    int val;
    int ok;
    int found;
    int row;
    int col;
    int rs;
    int bs;
    int idx;
    int g;
    int done;
    int i;
    int br;
    int bc;
    int br2;
    int bc2;
    int temp;

    grid[0] = -5; grid[1] = -3; grid[2] = 0; grid[3] = 0; grid[4] = -7; grid[5] = 0; grid[6] = 0; grid[7] = 0; grid[8] = 0;
    grid[9] = -6; grid[10] = 0; grid[11] = 0; grid[12] = -1; grid[13] = -9; grid[14] = -5; grid[15] = 0; grid[16] = 0; grid[17] = 0;
    grid[18] = 0; grid[19] = -9; grid[20] = -8; grid[21] = 0; grid[22] = 0; grid[23] = 0; grid[24] = 0; grid[25] = -6; grid[26] = 0;
    grid[27] = -8; grid[28] = 0; grid[29] = 0; grid[30] = 0; grid[31] = -6; grid[32] = 0; grid[33] = 0; grid[34] = 0; grid[35] = -3;
    grid[36] = -4; grid[37] = 0; grid[38] = 0; grid[39] = -8; grid[40] = 0; grid[41] = -3; grid[42] = 0; grid[43] = 0; grid[44] = -1;
    grid[45] = -7; grid[46] = 0; grid[47] = 0; grid[48] = 0; grid[49] = -2; grid[50] = 0; grid[51] = 0; grid[52] = 0; grid[53] = -6;
    grid[54] = 0; grid[55] = -6; grid[56] = 0; grid[57] = 0; grid[58] = 0; grid[59] = 0; grid[60] = -2; grid[61] = -8; grid[62] = 0;
    grid[63] = 0; grid[64] = 0; grid[65] = 0; grid[66] = -4; grid[67] = -1; grid[68] = -9; grid[69] = 0; grid[70] = 0; grid[71] = -5;
    grid[72] = 0; grid[73] = 0; grid[74] = 0; grid[75] = 0; grid[76] = -8; grid[77] = 0; grid[78] = 0; grid[79] = -7; grid[80] = -9;

    pos = 0;
    found = 0;
    while (pos < 81 && found == 0) {
        if (grid[pos] == 0) {
            found = 1;
        } else {
            pos = pos + 1;
        }
    }

    done = 0;
    while (done == 0) {
        if (pos > 80) {
            done = 1;
        } else if (pos < 0) {
            done = 2;
        } else {
            val = grid[pos] + 1;
            found = 0;

            while (val < 10 && found == 0) {
                ok = 1;

                row = 0;
                temp = pos;
                while (temp >= 9) {
                    temp = temp - 9;
                    row = row + 1;
                }
                col = temp;

                rs = (row << 3) + row;
                idx = rs;
                i = 0;
                while (i < 9) {
                    if (idx != pos) {
                        g = grid[idx];
                        if (g < 0) { g = -g; }
                        if (g == val) { ok = 0; }
                    }
                    idx = idx + 1;
                    i = i + 1;
                }

                if (ok != 0) {
                    idx = col;
                    i = 0;
                    while (i < 9) {
                        if (idx != pos) {
                            g = grid[idx];
                            if (g < 0) { g = -g; }
                            if (g == val) { ok = 0; }
                        }
                        idx = idx + 9;
                        i = i + 1;
                    }
                }

                if (ok != 0) {
                    br = row;
                    temp = br;
                    while (temp >= 3) { temp = temp - 3; }
                    br = br - temp;
                    bc = col;
                    temp = bc;
                    while (temp >= 3) { temp = temp - 3; }
                    bc = bc - temp;
                    bs = (br << 3) + br + bc;

                    idx = bs;
                    br2 = 0;
                    while (br2 < 3) {
                        bc2 = 0;
                        while (bc2 < 3) {
                            if (idx != pos) {
                                g = grid[idx];
                                if (g < 0) { g = -g; }
                                if (g == val) { ok = 0; }
                            }
                            idx = idx + 1;
                            bc2 = bc2 + 1;
                        }
                        idx = idx + 6;
                        br2 = br2 + 1;
                    }
                }

                if (ok != 0) {
                    found = 1;
                } else {
                    val = val + 1;
                }
            }

            if (found != 0) {
                grid[pos] = val;
                pos = pos + 1;
                found = 0;
                while (pos < 81 && found == 0) {
                    if (grid[pos] < 0) {
                        pos = pos + 1;
                    } else {
                        found = 1;
                    }
                }
            } else {
                grid[pos] = 0;
                pos = pos - 1;
                found = 0;
                while (pos >= 0 && found == 0) {
                    if (grid[pos] < 0) {
                        pos = pos - 1;
                    } else {
                        found = 1;
                    }
                }
            }
        }
    }

    return done;
}
"""


def main():
    print("=" * 60)
    print("9x9 Sudoku Solver - Large Model Export")
    print("=" * 60)

    cfg = ExtendedConfigV4(**SUDOKU_CONFIG)
    print(f"Config: s={cfg.s}, m={cfg.m}, n={cfg.n}, N={cfg.N}")
    print(f"  d_model={cfg.d_model}, logn={cfg.logn}")
    print(f"  Instruction slots: {cfg.n - cfg.s - cfg.m}")

    # Compile
    print("\nCompiling 9x9 Sudoku solver...")
    cfg_c, memory, commands, meta = compile_c(SUDOKU9_SOURCE, **SUDOKU_CONFIG)
    print(f"  {meta['num_instructions']} instructions")
    print(f"  {len(meta['variables'])} variables: {list(meta['variables'].keys())}")

    # Count opcodes
    from extended_isa_v4 import (OP_MOV, OP_ADD, OP_JMP, OP_JZ, OP_JNZ, OP_DEC,
        OP_SHL, OP_SHR, OP_CMP, OP_LOAD, OP_AND, OP_OR, OP_XOR, OP_SUB,
        OP_FIND, OP_SWAP, OP_CMOV, OP_MULACC)
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
    print("\nBuilding neural computer...")
    model = ExtendedISAModule(cfg)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    print(f"  Parameters: {total_params + total_buffers:,}")

    # Sanity test
    print("\nSanity test (INC)...")
    test_mem = [5] + [0] * (cfg.m - 1)
    test_cmds = [(OP_INC, cfg.s + 0, 0), (OP_HALT, 0, 0)]
    test_X = init_state_v4(cfg, test_mem, test_cmds)
    with torch.no_grad():
        test_Y = model(test_X)
    test_result = read_memory_v4(test_Y, cfg)
    assert test_result[0] == 6, f"INC test failed: got {test_result[0]}"
    print("  INC 5 -> 6: PASS")

    # Export ONNX
    output_dir = os.path.join(os.path.dirname(__file__), "programs")
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "sudoku.onnx")

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

    # Consolidate external data
    import onnx
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        print("  Consolidating external data...")
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

    state_path = os.path.join(output_dir, "sudoku9_state.json")
    with open(state_path, 'w') as f:
        json.dump({"state": X.tolist()}, f)
    state_kb = os.path.getsize(state_path) / 1024
    print(f"  State: {state_kb:.0f} KB")

    info = {
        "name": "sudoku9",
        "title": "9x9 Sudoku Solver",
        "description": "Solve a 9x9 Sudoku by iterative backtracking",
        "source": SUDOKU9_SOURCE,
        "variables": meta["variables"],
        "num_instructions": meta["num_instructions"],
        "opcode_counts": opcode_counts,
        "source_map": instr_to_line,
        "retval_addr": meta["retval_addr"],
        "output_addr": meta["output_addr"],
        "expected": {"RETVAL": 1},
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

    info_path = os.path.join(output_dir, "sudoku9_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Info: sudoku9_info.json")

    print(f"\nDone! Sudoku model: {cfg.d_model}x{cfg.n} ({total_params + total_buffers:,} params)")
    print(f"  {meta['num_instructions']} instructions, m={cfg.m} memory slots")


if __name__ == "__main__":
    main()
