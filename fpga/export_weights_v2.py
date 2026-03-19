"""
Export V4 weights for FPGA v2 (n=1024, 8 layers, STORE opcode).

Generates:
  data_v2/config.bin       - model config
  data_v2/weights.bin      - structured sparse weights (for CPU test)
  data_v2/weights_flat.bin - flat float buffer (for FPGA kernel)
  data_v2/test_input.bin   - test state (INC 5 -> 6)
  data_v2/test_output.bin  - reference output from Python

Usage:
    cd extended_isa
    python fpga/export_weights_v2.py
"""

import struct
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extended_isa_v4 import (
    ExtendedNeuralComputerV4, ExtendedConfigV4,
    init_state_v4, read_memory_v4, get_pc_v4,
    OP_HALT, OP_INC, OP_STORE
)


def write_i32(f, v): f.write(struct.pack('<i', v))
def write_f32(f, v): f.write(struct.pack('<f', v))
def write_i16(f, v): f.write(struct.pack('<h', v))


def argmax_step(comp, X):
    """Run one step using argmax attention (matches C++ exactly)."""
    TIE_THRESH = 1.0
    for layer in comp.layers:
        # Attention via argmax
        VX_acc = torch.zeros_like(X)
        if layer.num_heads == 1:
            heads = [(layer.Q, layer.K, layer.V)]
        elif layer.num_heads == 2:
            heads = [(layer.Q1, layer.K1, layer.V1), (layer.Q2, layer.K2, layer.V2)]
        else:
            heads = [(layer.Q1, layer.K1, layer.V1), (layer.Q2, layer.K2, layer.V2),
                     (layer.Q3, layer.K3, layer.V3)]
        for Q, K, V in heads:
            QX = Q @ X  # q_rows x n
            KX = K @ X
            if QX.abs().max() < 1e-6 and KX.abs().max() < 1e-6:
                continue
            VX = V @ X  # d x n
            n = X.shape[1]
            scores = KX.T @ QX  # n x n
            for i in range(n):
                if KX[:, i].abs().max() < 1e-6:
                    continue
                row = scores[i]
                top2 = torch.topk(row, 2)
                j1, j2 = top2.indices[0].item(), top2.indices[1].item()
                tied = (top2.values[0] - top2.values[1]).item() < TIE_THRESH
                if tied:
                    VX_acc[:, j1] += 0.5 * VX[:, i]
                    VX_acc[:, j2] += 0.5 * VX[:, i]
                else:
                    VX_acc[:, j1] += VX[:, i]
        attn = X + VX_acc
        # FFN
        ff1 = torch.relu(layer.W1 @ attn + layer.b1)
        X = attn + layer.W2 @ ff1 + layer.b2
    return X


def main():
    cfg = ExtendedConfigV4(s=32, m=160, n=1024, N=8)
    comp = ExtendedNeuralComputerV4(cfg)
    print(f"Config: d={cfg.d_model}, n={cfg.n}, logn={cfg.logn}, s={cfg.s}, m={cfg.m}")
    print(f"  idx_pc={cfg.idx_pc}, idx_memory={cfg.idx_memory}")

    out_dir = os.path.join(os.path.dirname(__file__), "data_v2")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Config ----
    with open(os.path.join(out_dir, "config.bin"), 'wb') as f:
        write_i32(f, cfg.d_model)
        write_i32(f, cfg.n)
        write_i32(f, cfg.s)
        write_i32(f, cfg.m)
        write_i32(f, cfg.N)
        write_i32(f, cfg.logn)
        write_f32(f, cfg.lam)
        write_i32(f, cfg.idx_pc)
        write_i32(f, cfg.idx_memory)
    print(f"  config.bin: {os.path.getsize(os.path.join(out_dir, 'config.bin'))} bytes")

    # ---- Structured weights (for CPU test) ----
    total_nnz = 0
    with open(os.path.join(out_dir, "weights.bin"), 'wb') as f:
        write_i32(f, len(comp.layers))
        for li, layer in enumerate(comp.layers):
            heads = layer.num_heads
            q_rows = layer.Q.shape[0] if heads == 1 else layer.Q1.shape[0]
            w1_rows = layer.W1.shape[0]
            write_i32(f, heads)
            write_i32(f, q_rows)
            write_i32(f, w1_rows)

            for h in range(heads):
                suffix = '' if heads == 1 else str(h + 1)
                for mat_name in ['Q', 'K', 'V']:
                    M = getattr(layer, f'{mat_name}{suffix}')
                    nz = (M != 0).nonzero(as_tuple=False)
                    write_i32(f, len(nz))
                    for idx in nz:
                        r, c = idx[0].item(), idx[1].item()
                        write_i16(f, r)
                        write_i16(f, c)
                        write_f32(f, M[r, c].item())
                    total_nnz += len(nz)

            # W1, b1, W2, b2 (must match load_model order in transformer.h)
            for M, b in [(layer.W1, layer.b1), (layer.W2, layer.b2)]:
                nz = (M != 0).nonzero(as_tuple=False)
                write_i32(f, len(nz))
                for idx in nz:
                    r, c = idx[0].item(), idx[1].item()
                    write_i16(f, r)
                    write_i16(f, c)
                    write_f32(f, M[r, c].item())
                total_nnz += len(nz)

                rows, cols = b.shape
                write_i32(f, rows)
                write_i32(f, cols)
                f.write(b.numpy().astype(np.float32).tobytes())

    size_kb = os.path.getsize(os.path.join(out_dir, "weights.bin")) / 1024
    print(f"  weights.bin: {size_kb:.1f} KB, {total_nnz} nonzeros")

    # ---- Flat weights (for FPGA kernel) ----
    buf = []
    buf.append(float(cfg.s))
    buf.append(float(len(comp.layers)))
    for layer in comp.layers:
        heads = layer.num_heads
        q_rows = layer.Q.shape[0] if heads == 1 else layer.Q1.shape[0]
        w1_rows = layer.W1.shape[0]
        buf.extend([float(heads), float(q_rows), float(w1_rows)])

        for h in range(heads):
            suffix = '' if heads == 1 else str(h + 1)
            for mat_name in ['Q', 'K', 'V']:
                M = getattr(layer, f'{mat_name}{suffix}')
                nz = (M != 0).nonzero(as_tuple=False)
                buf.append(float(len(nz)))
                for idx in nz:
                    r, c = idx[0].item(), idx[1].item()
                    buf.extend([float(r), float(c), M[r, c].item()])

        # W1
        nz = (layer.W1 != 0).nonzero(as_tuple=False)
        buf.append(float(len(nz)))
        for idx in nz:
            r, c = idx[0].item(), idx[1].item()
            buf.extend([float(r), float(c), layer.W1[r, c].item()])

        # b1 compact (3-zone)
        b1 = layer.b1
        buf.append(float(b1.shape[0]))
        for r in range(b1.shape[0]):
            buf.append(b1[r, 0].item())
            buf.append(b1[r, min(1, b1.shape[1]-1)].item())
            buf.append(b1[r, cfg.s].item() if cfg.s < b1.shape[1] else 0.0)

        # W2
        nz = (layer.W2 != 0).nonzero(as_tuple=False)
        buf.append(float(len(nz)))
        for idx in nz:
            r, c = idx[0].item(), idx[1].item()
            buf.extend([float(r), float(c), layer.W2[r, c].item()])

        # b2 compact (3-zone)
        b2 = layer.b2
        buf.append(float(b2.shape[0]))
        for r in range(b2.shape[0]):
            buf.append(b2[r, 0].item())
            buf.append(b2[r, min(1, b2.shape[1]-1)].item())
            buf.append(b2[r, cfg.s].item() if cfg.s < b2.shape[1] else 0.0)

    with open(os.path.join(out_dir, "weights_flat.bin"), 'wb') as f:
        f.write(np.array(buf, dtype=np.float32).tobytes())
    size_kb = len(buf) * 4 / 1024
    print(f"  weights_flat.bin: {size_kb:.1f} KB ({len(buf)} floats)")

    # ---- Test vectors: INC 5 -> 6 ----
    test_mem = [5] + [0] * (cfg.m - 1)
    test_cmds = [(OP_INC, cfg.s + 0, 0), (OP_HALT, 0, 0)]
    X_in = init_state_v4(cfg, test_mem, test_cmds)

    with open(os.path.join(out_dir, "test_input.bin"), 'wb') as f:
        write_i32(f, cfg.d_model)
        write_i32(f, cfg.n)
        f.write(X_in.numpy().astype(np.float32).tobytes())

    # Generate reference output using argmax (matches C++ exactly)
    with torch.no_grad():
        X_out = argmax_step(comp, X_in)

    with open(os.path.join(out_dir, "test_output.bin"), 'wb') as f:
        write_i32(f, cfg.d_model)
        write_i32(f, cfg.n)
        f.write(X_out.numpy().astype(np.float32).tobytes())

    result = read_memory_v4(X_out, cfg)
    pc = get_pc_v4(X_out, cfg)
    print(f"  test: INC 5 -> {result[0]}, PC={pc}")
    assert result[0] == 6, f"Test failed: {result[0]}"

    # ---- Also test STORE ----
    test_mem2 = [42, 3, 0, 0] + [0] * (cfg.m - 4)
    test_cmds2 = [(OP_STORE, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)]
    X_store = init_state_v4(cfg, test_mem2, test_cmds2)
    with torch.no_grad():
        X_store_out = argmax_step(comp, X_store)
    r2 = read_memory_v4(X_store_out, cfg)
    print(f"  test: STORE 42->slot3: mem[3]={r2[3]}")
    assert r2[3] == 42

    print(f"\nDone! Files in {out_dir}/")
    print(f"  State: {cfg.d_model}x{cfg.n} = {cfg.d_model*cfg.n:,} floats")
    print(f"  On-chip estimate: ~4 MB (fits in U200's 44 MB)")


if __name__ == "__main__":
    main()
