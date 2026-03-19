"""
Export V4 sparse weights with 3-zone biases for web sparse engine.

Fixes the bias representation: instead of (col0, broadcast), uses
(col0, scratchpad, memory) to correctly handle layers where b2 has
different values at scratchpad vs memory columns (L3, L8).

Usage:
    cd extended_isa
    python demos/export_sparse_weights_3zone.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extended_isa_v4 import ExtendedNeuralComputerV4, ExtendedConfigV4


def sparse_to_coo(tensor):
    nz = (tensor != 0).nonzero(as_tuple=False)
    rows = nz[:, 0].tolist()
    cols = nz[:, 1].tolist()
    vals = [round(tensor[r, c].item(), 6) for r, c in nz]
    return [rows, cols, vals]


def bias_to_3zone(bias_tensor, s):
    """Extract 3-zone bias: [col0[], scratchpad[], memory[]]."""
    rows = bias_tensor.shape[0]
    n = bias_tensor.shape[1]
    col0 = []
    scr = []
    mem = []
    for r in range(rows):
        col0.append(round(bias_tensor[r, 0].item(), 6))
        scr.append(round(bias_tensor[r, min(1, n - 1)].item(), 6))
        mem.append(round(bias_tensor[r, min(s, n - 1)].item(), 6))
    return [col0, scr, mem]


def export(s, m, n, N, output_path):
    print(f"Building V4 computer (s={s}, m={m}, n={n}, N={N})...")
    cfg = ExtendedConfigV4(s=s, m=m, n=n, N=N)
    comp = ExtendedNeuralComputerV4(cfg)
    print(f"  d_model={cfg.d_model}, logn={cfg.logn}")

    total_nnz = 0
    layers = []

    for li, layer in enumerate(comp.layers):
        heads = layer.num_heads
        q_rows = layer.Q.shape[0] if heads == 1 else layer.Q1.shape[0]
        w1_rows = layer.W1.shape[0]

        heads_data = []
        for h in range(heads):
            suffix = '' if heads == 1 else str(h + 1)
            Q = getattr(layer, f'Q{suffix}')
            K = getattr(layer, f'K{suffix}')
            V = getattr(layer, f'V{suffix}')
            heads_data.append([sparse_to_coo(Q), sparse_to_coo(K), sparse_to_coo(V)])
            total_nnz += (Q != 0).sum().item() + (K != 0).sum().item() + (V != 0).sum().item()

        W1_coo = sparse_to_coo(layer.W1)
        W2_coo = sparse_to_coo(layer.W2)
        total_nnz += (layer.W1 != 0).sum().item() + (layer.W2 != 0).sum().item()

        layers.append([
            heads, q_rows, w1_rows, heads_data,
            W1_coo, bias_to_3zone(layer.b1, s),
            W2_coo, bias_to_3zone(layer.b2, s),
        ])

    data = {"d": cfg.d_model, "s": s, "layers": layers}

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported: {output_path} ({size_kb:.1f} KB)")
    print(f"  Total nonzero weights: {total_nnz}")
    print(f"  Bias format: 3-zone (col0, scratchpad, memory)")


if __name__ == "__main__":
    # Export for doom/sudoku9 config
    export(s=32, m=224, n=2048, N=8,
           output_path="demos/programs/doom_sparse_weights_3z.json")

    # Also export for standard config
    export(s=32, m=64, n=1024, N=8,
           output_path="demos/programs/sparse_weights_3z.json")
