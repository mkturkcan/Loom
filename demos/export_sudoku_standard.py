"""
Export standard 8-layer ONNX for the compact 146x512 config.

Unlike the deprecated merged export, this uses the full 8-layer V4 architecture
which is compatible with dim=0 softmax.

Output: demos/programs/sudoku_standard.onnx (~7 MB)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extended_isa_v4 import (
    ExtendedNeuralComputerV4, ExtendedConfigV4,
    init_state_v4, read_memory_v4, get_pc_v4,
    OP_HALT, OP_INC
)


class StandardISAModule(nn.Module):
    """Standard 8-layer V4 as a traceable nn.Module for ONNX export."""

    def __init__(self, cfg):
        super().__init__()
        comp = ExtendedNeuralComputerV4(cfg)
        self._lam = comp.layers[0].lam
        self._layer_defs = []
        all_KtQ, all_V, all_W1, all_b1, all_W2, all_b2 = [], [], [], [], [], []
        for layer in comp.layers:
            heads = layer.num_heads
            start = len(all_KtQ)
            for h in range(heads):
                suffix = '' if heads == 1 else str(h + 1)
                Q = getattr(layer, f'Q{suffix}')
                K = getattr(layer, f'K{suffix}')
                V = getattr(layer, f'V{suffix}')
                if Q.abs().sum() == 0 and K.abs().sum() == 0 and V.abs().sum() == 0:
                    continue
                all_KtQ.append(K.t() @ Q)
                all_V.append(V)
            self._layer_defs.append((start, len(all_KtQ) - start))
            all_W1.append(layer.W1)
            all_b1.append(layer.b1)
            all_W2.append(layer.W2)
            all_b2.append(layer.b2)
        for i, t in enumerate(all_KtQ): self.register_buffer(f'KtQ_{i}', t)
        for i, t in enumerate(all_V): self.register_buffer(f'V_{i}', t)
        for i, t in enumerate(all_W1): self.register_buffer(f'W1_{i}', t)
        for i, t in enumerate(all_b1): self.register_buffer(f'b1_{i}', t)
        for i, t in enumerate(all_W2): self.register_buffer(f'W2_{i}', t)
        for i, t in enumerate(all_b2): self.register_buffer(f'b2_{i}', t)
        self._n_layers = len(comp.layers)

    def forward(self, X):
        lam = self._lam
        for li in range(self._n_layers):
            start, n_active = self._layer_defs[li]
            attn = X
            for hi in range(n_active):
                idx = start + hi
                scores = X.t() @ getattr(self, f'KtQ_{idx}') @ X
                weights = F.softmax(lam * scores, dim=0)
                attn = attn + getattr(self, f'V_{idx}') @ (X @ weights)
            ff1 = F.relu(getattr(self, f'W1_{li}') @ attn + getattr(self, f'b1_{li}'))
            X = attn + getattr(self, f'W2_{li}') @ ff1 + getattr(self, f'b2_{li}')
        return X


def main():
    cfg = ExtendedConfigV4(s=32, m=160, n=512, N=8)
    print(f"Config: d={cfg.d_model}, n={cfg.n}, logn={cfg.logn}, m={cfg.m}")
    print(f"  Instruction slots: {cfg.n - cfg.s - cfg.m}")

    model = StandardISAModule(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    total_active = sum(n for _, n in model._layer_defs)
    print(f"Layers: {model._n_layers}, active heads: {total_active}")

    # Sanity check
    print("\nSanity: INC 5->6...")
    X = init_state_v4(cfg, [5] + [0] * 127, [(OP_INC, cfg.s, 0), (OP_HALT, 0, 0)])
    with torch.no_grad():
        Y = model(X)
    assert read_memory_v4(Y, cfg)[0] == 6, f"Got {read_memory_v4(Y, cfg)[0]}"
    print("  PASS")

    # Export ONNX
    output_dir = os.path.join(os.path.dirname(__file__), "programs")
    onnx_path = os.path.join(output_dir, "sudoku_standard.onnx")
    print(f"\nExporting ONNX to {onnx_path}...")

    os.environ['PYTORCH_ONNX_USE_LEGACY_EXPORT'] = '1'
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
    try:
        import onnx
        data_path = onnx_path + ".data"
        if os.path.exists(data_path):
            onnx_model = onnx.load(onnx_path, load_external_data=True)
            onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
            if os.path.exists(data_path):
                os.remove(data_path)
    except ImportError:
        pass

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Saved: {size_mb:.1f} MB")

    # Save config JSON
    config_json = {
        's': cfg.s, 'm': cfg.m, 'n': cfg.n, 'N': cfg.N,
        'logn': cfg.logn, 'd_model': cfg.d_model,
        'idx_memory': cfg.idx_memory, 'idx_pc': cfg.idx_pc,
    }
    config_path = os.path.join(output_dir, "sudoku_standard_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    print(f"  Config: {config_path}")

    # Verify ONNX matches PyTorch
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        result = session.run(None, {'state': X.numpy()})
        X_onnx = torch.from_numpy(result[0])
        with torch.no_grad():
            X_torch = model(X)
        diff = torch.abs(X_onnx - X_torch).max().item()
        print(f"  ONNX verification: max diff = {diff:.6f}")
    except ImportError:
        print("  (onnxruntime not available, skipping verification)")

    print(f"\nDone! {cfg.d_model}x{cfg.n} standard model ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
