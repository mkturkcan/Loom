"""
Export V4 transformer for the DOOM E1M1 demo.

The DOOM game logic needs a larger config than the default (s=32, m=64, n=1024):
  - 16 enemies × 10 arrays = 160 memory slots for enemy state
  - 8 player state vars + 10 scratch vars + 2 output vars = 20 more
  - Total: ~180 variables → m=224 memory slots
  - n=2048 columns for instruction space
  - d_model = 161 (vs 155 for the default config)

Exports both ONNX and sparse weights JSON.

Usage:
    cd extended_isa
    python demos/export_doom_model.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from extended_isa_v4 import ExtendedNeuralComputerV4, ExtendedConfigV4


# ── Sparse weights export ────────────────────────────────────

def sparse_to_coo(tensor):
    """Convert a 2D tensor to COO format: [rows[], cols[], vals[]]."""
    nz = (tensor != 0).nonzero(as_tuple=False)
    rows = nz[:, 0].tolist()
    cols = nz[:, 1].tolist()
    vals = [round(tensor[r, c].item(), 4) for r, c in nz]
    return [rows, cols, vals]


def bias_to_compact(bias_tensor):
    """Extract compact bias: col0 value + broadcast value per row."""
    rows = bias_tensor.shape[0]
    col0 = []
    bcast = []
    for r in range(rows):
        row = bias_tensor[r]
        c0 = round(row[0].item(), 4)
        rest = row[1:]
        nz = rest[rest != 0]
        b = round(nz[0].item(), 4) if len(nz) > 0 else 0.0
        col0.append(c0)
        bcast.append(b)
    return [col0, bcast]


def export_sparse(comp, cfg, out_path):
    """Export sparse weights JSON."""
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
            heads_data.append([
                sparse_to_coo(Q),
                sparse_to_coo(K),
                sparse_to_coo(V),
            ])
            total_nnz += (Q != 0).sum().item()
            total_nnz += (K != 0).sum().item()
            total_nnz += (V != 0).sum().item()

        W1_coo = sparse_to_coo(layer.W1)
        W2_coo = sparse_to_coo(layer.W2)
        total_nnz += (layer.W1 != 0).sum().item()
        total_nnz += (layer.W2 != 0).sum().item()

        layers.append([
            heads, q_rows, w1_rows,
            heads_data,
            W1_coo,
            bias_to_compact(layer.b1),
            W2_coo,
            bias_to_compact(layer.b2),
        ])

        print(f"  L{li+1}: {heads}H, q_rows={q_rows}, w1_rows={w1_rows}")

    data = {"d": cfg.d_model, "layers": layers}

    with open(out_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n  Total nonzero weights: {total_nnz}")
    print(f"  Sparse weights: {out_path} ({size_kb:.0f} KB)")
    return data


# ── ONNX export ──────────────────────────────────────────────

class ExtendedISAModule(nn.Module):
    """PyTorch Module wrapper for Extended ISA with fixed weights."""

    def __init__(self, cfg, computer):
        super().__init__()
        self.cfg = cfg
        self.layer_data = nn.ModuleList()

        for layer in computer.layers:
            lm = nn.Module()
            if hasattr(layer, 'Q') and layer.Q is not None:
                lm.register_buffer('Q', layer.Q)
                lm.register_buffer('K', layer.K)
                lm.register_buffer('V', layer.V)
            for i in range(1, 4):
                for mat in ['Q', 'K', 'V']:
                    name = f'{mat}{i}'
                    if hasattr(layer, name) and getattr(layer, name) is not None:
                        lm.register_buffer(name, getattr(layer, name))
            lm.register_buffer('W1', layer.W1)
            lm.register_buffer('b1', layer.b1)
            lm.register_buffer('W2', layer.W2)
            lm.register_buffer('b2', layer.b2)
            lm.num_heads = layer.num_heads
            lm.lam = layer.lam
            self.layer_data.append(lm)

    def _attn(self, X, Q, K, V, lam):
        QX = Q @ X
        KX = K @ X
        scores = QX.T @ KX * lam
        A = torch.softmax(scores, dim=0)
        VX = V @ X
        return VX @ A

    def _layer_forward(self, X, layer):
        Y = torch.zeros_like(X)
        if layer.num_heads == 1 and hasattr(layer, 'Q'):
            Y = self._attn(X, layer.Q, layer.K, layer.V, layer.lam)
        elif layer.num_heads >= 2:
            for i in range(1, layer.num_heads + 1):
                Qi = getattr(layer, f'Q{i}', None)
                Ki = getattr(layer, f'K{i}', None)
                Vi = getattr(layer, f'V{i}', None)
                if Qi is not None:
                    Y = Y + self._attn(X, Qi, Ki, Vi, layer.lam)
        X = X + Y
        H = torch.relu(layer.W1 @ X + layer.b1)
        X = X + layer.W2 @ H + layer.b2
        return X

    def forward(self, X):
        for layer in self.layer_data:
            X = self._layer_forward(X, layer)
        return X


def export_onnx(comp, cfg, out_path):
    """Export ONNX model."""
    model = ExtendedISAModule(cfg, comp)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    X = torch.zeros(cfg.d_model, cfg.n)

    os.environ['PYTORCH_ONNX_USE_LEGACY_EXPORT'] = '1'
    torch.onnx.export(
        model, X, out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['new_state'],
        verbose=False
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ONNX model: {out_path} ({size_mb:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────

def main():
    # DOOM config: 224 memory slots, 2048 columns
    S, M, N_COLS, N_BITS = 32, 224, 2048, 8

    print("=" * 60)
    print("DOOM E1M1 — V4 Transformer Export")
    print("=" * 60)
    print(f"\nConfig: s={S}, m={M}, n={N_COLS}, N={N_BITS}")

    cfg = ExtendedConfigV4(s=S, m=M, n=N_COLS, N=N_BITS)
    print(f"  d_model:    {cfg.d_model}")
    print(f"  logn:       {cfg.logn}")
    print(f"  State size: {cfg.d_model * cfg.n * 4 / 1024:.0f} KB")
    print(f"  Max instructions: {cfg.n - cfg.s - cfg.m}")

    print("\nBuilding V4 computer...")
    comp = ExtendedNeuralComputerV4(cfg)
    print(f"  Layers: {len(comp.layers)}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "programs")
    os.makedirs(out_dir, exist_ok=True)

    # Export sparse weights
    print("\nExporting sparse weights...")
    sparse_path = os.path.join(out_dir, "doom_sparse_weights.json")
    export_sparse(comp, cfg, sparse_path)

    # Export ONNX
    print("\nExporting ONNX model...")
    onnx_path = os.path.join(out_dir, "doom.onnx")
    export_onnx(comp, cfg, onnx_path)

    # Export config JSON
    config = {
        's': cfg.s, 'm': cfg.m, 'n': cfg.n, 'N': cfg.N,
        'logn': cfg.logn, 'd_model': cfg.d_model,
        'idx_memory': cfg.idx_memory,
        'idx_scratchpad': cfg.idx_scratchpad,
        'idx_pc': cfg.idx_pc,
        'idx_pos_enc': cfg.idx_pos_enc,
        'idx_buffer': cfg.idx_buffer,
        'idx_tag': cfg.idx_tag,
        'isa_type': 'extended',
    }
    config_path = os.path.join(out_dir, "doom_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")

    print("\n" + "=" * 60)
    print("Done! Files exported to demos/programs/")
    print("  doom_sparse_weights.json  — for sparse transformer engine")
    print("  doom.onnx                 — for ONNX Runtime engine")
    print("  doom_config.json          — config metadata")
    print("=" * 60)


if __name__ == '__main__':
    main()
