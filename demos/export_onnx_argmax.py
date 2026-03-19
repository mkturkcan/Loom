"""
Export ONNX model with argmax attention (no softmax).

Replaces softmax with TopK + OneHot for numerically exact attention.
This eliminates the floating-point drift that causes multi-step programs
to produce wrong results with the softmax ONNX model.

Output: demos/programs/argmax_146x512.onnx
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
    OP_HALT, OP_INC, OP_ADD, OP_JZ, OP_JNZ, OP_JMP
)
from subleq import signed_from_bipolar


class ArgmaxISAModule(nn.Module):
    """8-layer V4 with argmax attention for ONNX export."""

    def __init__(self, cfg):
        super().__init__()
        comp = ExtendedNeuralComputerV4(cfg)
        self._lam = comp.layers[0].lam
        self._n = cfg.n
        self._layer_defs = []  # (start_idx, num_active_heads)
        all_KtQ, all_K, all_V = [], [], []
        all_W1, all_b1, all_W2, all_b2 = [], [], [], []

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
                all_K.append(K)  # q_rows × d (small!)
                all_V.append(V)
            self._layer_defs.append((start, len(all_KtQ) - start))
            all_W1.append(layer.W1)
            all_b1.append(layer.b1)
            all_W2.append(layer.W2)
            all_b2.append(layer.b2)

        # Store K (=Q) separately for factored score computation: (K@X)^T @ (K@X)
        # K is q_rows × d (very small: 9×146), vs KtQ which is d×d (146×146)
        for i, t in enumerate(all_KtQ):
            self.register_buffer(f'KtQ_{i}', t)  # kept for reference
        for i, t in enumerate(all_K):
            self.register_buffer(f'K_{i}', t)
        for i, t in enumerate(all_V):
            self.register_buffer(f'V_{i}', t)
        for i, t in enumerate(all_W1):
            self.register_buffer(f'W1_{i}', t)
        for i, t in enumerate(all_b1):
            self.register_buffer(f'b1_{i}', t)
        for i, t in enumerate(all_W2):
            self.register_buffer(f'W2_{i}', t)
        for i, t in enumerate(all_b2):
            self.register_buffer(f'b2_{i}', t)
        self._n_layers = len(comp.layers)

    def _argmax_attn(self, K, V, X):
        """Argmax attention using only GPU-native ops.

        Factored score computation: (K@X)^T @ (K@X) instead of X^T @ K^T@K @ X.
        K is q_rows×d (9×146), so KX is 9×512 — much smaller than the
        146×146 KtQ matrix approach. This is 16x less compute per head.

        Then: ReduceMax + threshold mask + normalize (all GPU-native).
        """
        KX = K @ X  # [q_rows, n] — small! (9×512)
        scores = KX.t() @ KX  # [n, n] — via 512×9 @ 9×512

        # Max per row
        max_vals = scores.max(dim=1, keepdim=True).values  # [n, 1]

        # Mask: entries within 0.5 of row max
        mask = (scores >= max_vals - 0.5).float()  # [n, n]

        # Normalize: each row sums to 1
        row_sums = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [n, 1]
        weights = mask / row_sums  # [n, n]

        # Zero out rows where max score ≈ 0 (no-op attention for those source columns)
        active = (max_vals.abs() > 0.5).float()  # [n, 1]
        weights = weights * active

        return V @ (X @ weights)

    def forward(self, X):
        for li in range(self._n_layers):
            start, n_active = self._layer_defs[li]
            attn = X
            for hi in range(n_active):
                idx = start + hi
                K = getattr(self, f'K_{idx}')
                V = getattr(self, f'V_{idx}')
                attn = attn + self._argmax_attn(K, V, X)
            ff1 = F.relu(getattr(self, f'W1_{li}') @ attn + getattr(self, f'b1_{li}'))
            X = attn + getattr(self, f'W2_{li}') @ ff1 + getattr(self, f'b2_{li}')
        return X


def export_argmax(s, m, n, N, name):
    """Export argmax ONNX for a given config."""
    cfg = ExtendedConfigV4(s=s, m=m, n=n, N=N)
    print(f"\n{'='*60}")
    print(f"Config '{name}': d={cfg.d_model}, n={cfg.n}, logn={cfg.logn}, m={cfg.m}")

    model = ArgmaxISAModule(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ---- Sanity checks ----
    print("\nSanity checks:")

    # INC 5 -> 6
    X = init_state_v4(cfg, [5] + [0] * 159, [(OP_INC, cfg.s, 0), (OP_HALT, 0, 0)])
    with torch.no_grad():
        Y = model(X)
    mem = read_memory_v4(Y, cfg)
    pc = get_pc_v4(Y, cfg)
    print(f"  INC 5->6: mem[0]={mem[0]}, PC={pc} {'PASS' if mem[0] == 6 else 'FAIL'}")
    assert mem[0] == 6

    # Multi-step: INC x3
    X = init_state_v4(cfg, [0] + [0] * 159,
                       [(OP_INC, cfg.s, 0), (OP_INC, cfg.s, 0), (OP_INC, cfg.s, 0), (OP_HALT, 0, 0)])
    with torch.no_grad():
        for step in range(5):
            pc = get_pc_v4(X, cfg)
            if pc == 0:
                break
            X = model(X)
    mem = read_memory_v4(X, cfg)
    print(f"  INC x3: mem[0]={mem[0]} (expect 3) {'PASS' if mem[0] == 3 else 'FAIL'}")

    # ADD: 3 + 7 = 10
    X = init_state_v4(cfg, [3, 7] + [0] * 158,
                       [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)])
    with torch.no_grad():
        for step in range(3):
            pc = get_pc_v4(X, cfg)
            if pc == 0:
                break
            X = model(X)
    mem = read_memory_v4(X, cfg)
    print(f"  ADD 3+7: mem[0]={mem[0]} (expect 10) {'PASS' if mem[0] == 10 else 'FAIL'}")

    # Loop: INC x100
    from c_compiler import compile_c
    loop_src = "int main() { int x; int i; i = 0; while (i < 10) { x = x + 1; i = i + 1; } return x; }"
    _, mem_init, cmds, meta = compile_c(loop_src, s=32, m=160, n=512, N=8)
    X = init_state_v4(cfg, mem_init, cmds)
    with torch.no_grad():
        for step in range(2000):
            pc = get_pc_v4(X, cfg)
            if pc == 0:
                break
            X = model(X)
    x_val = signed_from_bipolar(X[cfg.idx_memory:cfg.idx_memory + cfg.N, cfg.s + meta['variables']['x']])
    print(f"  Loop INC x10: x={x_val} (expect 10), steps={step} {'PASS' if x_val == 10 else 'FAIL'}")

    # ---- Export ONNX ----
    output_dir = os.path.join(os.path.dirname(__file__), "programs")
    onnx_path = os.path.join(output_dir, f"argmax_{name}.onnx")
    print(f"\nExporting ONNX to {onnx_path}...")

    X_dummy = init_state_v4(cfg, [5] + [0] * 159, [(OP_INC, cfg.s, 0), (OP_HALT, 0, 0)])
    os.environ['PYTORCH_ONNX_USE_LEGACY_EXPORT'] = '1'
    torch.onnx.export(
        model, X_dummy, onnx_path,
        export_params=True,
        opset_version=17,  # Need 17 for OneHot
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

    # Verify ONNX matches PyTorch
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        result = session.run(None, {'state': X_dummy.numpy()})
        X_onnx = torch.from_numpy(result[0])
        with torch.no_grad():
            X_torch = model(X_dummy)
        diff = torch.abs(X_onnx - X_torch).max().item()
        print(f"  ONNX verification: max diff = {diff:.6f}")
    except ImportError:
        print("  (onnxruntime not available, skipping verification)")

    # Save config JSON
    config_json = {
        's': cfg.s, 'm': cfg.m, 'n': cfg.n, 'N': cfg.N,
        'logn': cfg.logn, 'd_model': cfg.d_model,
        'idx_memory': cfg.idx_memory, 'idx_pc': cfg.idx_pc,
    }
    config_path = os.path.join(output_dir, f"argmax_{name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_json, f, indent=2)

    print(f"\nDone! {cfg.d_model}x{cfg.n} argmax model ({size_mb:.1f} MB)")


def main():
    # Compact: sudoku, floodfill, benchmark
    export_argmax(s=32, m=160, n=512, N=8, name="146x512")
    # Standard: sorting, debugger, repl, snake, doom_fast
    export_argmax(s=32, m=64, n=1024, N=8, name="155x1024")
    # Large: life, doom_e1m1, sudoku_fast
    export_argmax(s=32, m=224, n=2048, N=8, name="164x2048")


if __name__ == "__main__":
    main()
