"""
Extended ISA V4 Standard — Standard Transformer Formulation
============================================================

Same computation as V4 but with:
1. Broadcast-only FFN biases (per-column biases absorbed into weights via encoding rows)
2. Optional argmax fast-attention mode (top-2 argmax + tie detection)

Softmax uses dim=0 (column-normalised, standard transformer convention).
With Q=K symmetric scores, dim=0 and dim=1 produce identical results for all
computation-relevant columns. The only difference is at "dead" columns (e.g.
col n-1 whose Q-projection is zero), where dim=0 produces uniform attention
instead of near-zero contributions — but these perturbations stay in buffer
rows that are never read by subsequent gated FFN layers.

State layout appends one row to V4:
  [..., tag (N rows), indicator (1 row), col0_indicator (1 row)]
  col0_indicator = 1.0 at column 0, 0.0 elsewhere
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from subleq import (
    SUBLEQConfig, to_bipolar, from_bipolar, signed_to_bipolar, signed_from_bipolar
)

from loom_v1 import (
    LoomComputer, LoomConfig,
    init_state, read_memory, get_pc,
    opcode_pattern, create_bipolar_value,
    OP_HALT, OP_MOV, OP_ADD, OP_JMP, OP_JZ, OP_JNZ, OP_INC, OP_DEC,
    OP_SHL, OP_SHR, OP_CMP, OP_LOAD, OP_AND, OP_OR, OP_XOR, OP_SUB,
    OP_FIND, OP_SWAP, OP_CMOV, OP_MULACC, OP_STORE,
)


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass
class LoomStandardConfig(LoomConfig):
    """Config with extra col0_indicator row appended after indicator."""

    def __post_init__(self):
        super().__post_init__()
        # Append col0_indicator as the very last row
        self.idx_col0 = self.d_model          # right after indicator
        self.idx_indicator = self.d_model - 1  # original indicator position
        self.d_model += 1


# ──────────────────────────────────────────────────────────────
# Standard Transformer Layer
# ──────────────────────────────────────────────────────────────

class StandardTransformerLayer:
    """
    Transformer layer with:
      - 1-D broadcast biases (no per-column bias matrices)
      - Optional argmax fast-attention mode

    Forward:
        scores = X^T K^T Q X                          (n × n)
        attn_weights = softmax(λ * scores, dim=0)      (n × n)
        attn_out = X + Σ_h V_h @ X @ attn_weights_h
        ff1 = ReLU(W1 @ attn_out + b1)                (b1 is 1-D, broadcast)
        output = attn_out + W2 @ ff1 + b2              (b2 is 1-D, broadcast)
    """

    def __init__(self, num_heads: int, lam: float):
        self.num_heads = num_heads
        self.lam = lam
        self.Qs: List[torch.Tensor] = []
        self.Ks: List[torch.Tensor] = []
        self.Vs: List[torch.Tensor] = []
        self.W1: torch.Tensor = None
        self.b1: torch.Tensor = None   # 1-D
        self.W2: torch.Tensor = None
        self.b2: torch.Tensor = None   # 1-D

    # ── forward (argmax attention) ─────────────────────────────

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        attn = X
        for Q, K, V in zip(self.Qs, self.Ks, self.Vs):
            attn = attn + _argmax_attention(Q, K, V, attn)
        ff1 = F.relu(self.W1 @ attn + self.b1.unsqueeze(1))
        return attn + self.W2 @ ff1 + self.b2.unsqueeze(1)

    forward_argmax = forward  # alias for compatibility


def _argmax_attention(Q, K, V, X):
    """
    Replace softmax with top-2 argmax + tie detection.

    Implementation uses row-wise top-2 (dim=1 convention internally), which
    is equivalent to column-wise top-2 (dim=0) because all our Q=K heads
    produce symmetric score matrices.

    For output column j: weight from source i is A[i,j].
    With λ=10, each row i has at most 2 tied maxima.

    When 2 entries tie: 50/50 average.
    When 1 entry dominates: one-hot.
    When all scores are 0 (Q/K zeros): returns 0 (matching V4 behaviour).
    """
    n = X.shape[1]
    QX = Q @ X                         # q_rows × n
    KX = K @ X                         # q_rows × n
    VX = V @ X                         # d × n

    # Check if Q/K are effectively zero (no-op attention head)
    if QX.abs().max() < 1e-6 and KX.abs().max() < 1e-6:
        return torch.zeros_like(VX)

    scores = KX.T @ QX                 # n × n, scores[i,j]

    # For dim=1 (row-wise softmax): each row i has its own max.
    # Output col j = sum_i A[i,j] * VX[:,i]
    # A[i,j] ≈ 1/k if j is among the k top entries of row i, else ≈ 0.
    #
    # With λ=10: each row has 1 or 2 tied maxima.
    # For each row i, find top-2 columns. Then for output column j,
    # accumulate contributions from rows whose top-k includes j.

    # Top-2 per row (dim=1)
    top2_vals, top2_idx = scores.topk(2, dim=1)  # n × 2

    # True ties have gap ~0 (identical scores). Hamming-1 neighbours
    # have gap = 2.0 (with possible ±0.5 noise from residual effects).
    # Threshold of 1.0 catches true ties but rejects false ties.
    tied = (top2_vals[:, 0] - top2_vals[:, 1]) < 1.0  # n bools

    # Build output: for each output column j, sum contributions
    result = torch.zeros_like(VX)

    for i in range(n):
        j1 = top2_idx[i, 0].item()
        j2 = top2_idx[i, 1].item()
        if tied[i]:
            # Row i splits 50/50 between j1 and j2
            result[:, j1] += 0.5 * VX[:, i]
            result[:, j2] += 0.5 * VX[:, i]
        else:
            # Row i goes entirely to j1
            result[:, j1] += VX[:, i]

    return result


# ──────────────────────────────────────────────────────────────
# Conversion from V4 layers
# ──────────────────────────────────────────────────────────────

def _standardize_layer(old_layer, d_old: int, d_new: int,
                       col0_idx: int, indicator_idx: int,
                       n: int, s: int) -> StandardTransformerLayer:
    """
    Convert one V4 TransformerLayer to StandardTransformerLayer.

    Steps:
      1. Expand Q/K/V by appending a zero column/row for col0_indicator.
      2. Convert per-column b1 matrix → broadcast b1 vector by folding the
         column-0 delta into W1 via the col0_indicator column.
      3. Convert per-column b2 matrix → broadcast b2 vector by adding extra
         FFN rows that read the scratchpad indicator and col0_indicator.
    """
    new = StandardTransformerLayer(old_layer.num_heads, old_layer.lam)

    # ── 1. Expand Q, K, V ────────────────────────────────────
    def pad_col(M):
        return F.pad(M, (0, 1))            # append zero column

    def pad_square(M):
        return F.pad(F.pad(M, (0, 1)), (0, 0, 0, 1))  # append col then row

    heads = old_layer.num_heads
    if heads == 1:
        new.Qs = [pad_col(old_layer.Q)]
        new.Ks = [pad_col(old_layer.K)]
        new.Vs = [pad_square(old_layer.V)]
    elif heads == 2:
        new.Qs = [pad_col(old_layer.Q1), pad_col(old_layer.Q2)]
        new.Ks = [pad_col(old_layer.K1), pad_col(old_layer.K2)]
        new.Vs = [pad_square(old_layer.V1), pad_square(old_layer.V2)]
    else:
        new.Qs = [pad_col(old_layer.Q1), pad_col(old_layer.Q2),
                  pad_col(old_layer.Q3)]
        new.Ks = [pad_col(old_layer.K1), pad_col(old_layer.K2),
                  pad_col(old_layer.K3)]
        new.Vs = [pad_square(old_layer.V1), pad_square(old_layer.V2),
                  pad_square(old_layer.V3)]

    # ── 2. Convert b1 → broadcast + W1 col0 weight ───────────
    w1_rows = old_layer.W1.shape[0]
    ref_col = s                              # first memory column
    b1_broadcast = old_layer.b1[:, ref_col].clone()
    b1_col0_delta = old_layer.b1[:, 0] - b1_broadcast

    W1_padded = pad_col(old_layer.W1)        # w × d_new
    W1_padded[:, col0_idx] += b1_col0_delta  # fold col-0 bias into weight

    # ── 3. Convert b2 → broadcast + extra FFN rows ───────────
    b2_mem_ref = old_layer.b2[:, s].clone()            # memory-column baseline
    b2_scr_ref = old_layer.b2[:, min(1, s - 1)].clone() if s > 1 \
                 else old_layer.b2[:, 0].clone()
    b2_scr_delta = b2_scr_ref - b2_mem_ref             # scratchpad offset
    b2_col0_extra = old_layer.b2[:, 0] - b2_scr_ref   # col-0 extra offset

    has_scr = b2_scr_delta.abs().max().item() > 1e-10
    has_col0 = b2_col0_extra.abs().max().item() > 1e-10
    extra = int(has_scr) + int(has_col0)
    total_rows = w1_rows + extra

    # Build final W1
    W1_final = torch.zeros(total_rows, d_new)
    W1_final[:w1_rows] = W1_padded

    # Build final W2 (expand d_old → d_new by padding one row)
    W2_padded = F.pad(old_layer.W2, (0, 0, 0, 1))  # (d_new) × w1_rows
    W2_final = torch.zeros(d_new, total_rows)
    W2_final[:, :w1_rows] = W2_padded

    # Biases
    b1_final = torch.zeros(total_rows)
    b1_final[:w1_rows] = b1_broadcast

    b2_final = F.pad(b2_mem_ref, (0, 1))   # d_new (broadcast)

    # Extra FFN rows for b2 per-column parts
    idx = w1_rows
    if has_scr:
        # Row that reads scratchpad indicator → 1 at scratchpad, 0 at memory
        W1_final[idx, indicator_idx] = 1.0
        b1_final[idx] = 0.0
        W2_final[:d_old, idx] = b2_scr_delta
        idx += 1
    if has_col0:
        # Row that reads col0_indicator → 1 at col 0, 0 elsewhere
        W1_final[idx, col0_idx] = 1.0
        b1_final[idx] = 0.0
        W2_final[:d_old, idx] = b2_col0_extra
        idx += 1

    new.W1 = W1_final
    new.b1 = b1_final
    new.W2 = W2_final
    new.b2 = b2_final

    return new


# ──────────────────────────────────────────────────────────────
# Main Computer Class
# ──────────────────────────────────────────────────────────────

class LoomComputerStandard:
    """
    Standard-transformer version of the V4 neural SUBLEQ computer.

    Builds a V4 computer internally, then converts every layer to standard form:
      - dim=0 softmax (normalise over keys)
      - broadcast-only biases
    """

    def __init__(self, cfg: LoomStandardConfig, use_argmax: bool = False):
        self.cfg = cfg
        self.use_argmax = use_argmax

        # Build V4 computer with the original config
        v4_cfg = LoomConfig(s=cfg.s, m=cfg.m, n=cfg.n, N=cfg.N, lam=cfg.lam)
        v4 = LoomComputer(v4_cfg)

        # Convert every layer
        d_old = v4_cfg.d_model
        d_new = cfg.d_model
        self.layers: List[StandardTransformerLayer] = []
        for layer in v4.layers:
            std = _standardize_layer(
                layer, d_old, d_new,
                col0_idx=cfg.idx_col0,
                indicator_idx=cfg.idx_indicator,
                n=cfg.n, s=cfg.s,
            )
            self.layers.append(std)

    def step(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.use_argmax:
                X = layer.forward_argmax(X)
            else:
                X = layer.forward(X)
        return X

    def run(self, X: torch.Tensor, max_steps: int = 1000) -> Tuple[torch.Tensor, int]:
        for step in range(max_steps):
            pc = get_pc_standard(X, self.cfg)
            if pc == 0 or pc >= self.cfg.n:
                return X, step
            X = self.step(X)
        return X, max_steps


# ──────────────────────────────────────────────────────────────
# State Init / Read helpers
# ──────────────────────────────────────────────────────────────

def init_state_standard(cfg: LoomStandardConfig, memory, commands):
    """Initialise state tensor with col0_indicator appended."""
    v4_cfg = LoomConfig(s=cfg.s, m=cfg.m, n=cfg.n, N=cfg.N, lam=cfg.lam)
    X_v4 = init_state(v4_cfg, memory, commands)

    X = torch.zeros(cfg.d_model, cfg.n)
    X[:v4_cfg.d_model] = X_v4          # all V4 rows including indicator
    X[cfg.idx_col0, 0] = 1.0           # col0_indicator
    return X


def read_memory_standard(X, cfg: LoomStandardConfig):
    """Read memory values (same indices as V4)."""
    memory = []
    for i in range(cfg.m):
        col = cfg.s + i
        bipolar = X[cfg.idx_memory:cfg.idx_memory + cfg.N, col]
        memory.append(signed_from_bipolar(bipolar))
    return memory


def get_pc_standard(X, cfg: LoomStandardConfig):
    """Get PC value (same indices as V4)."""
    return from_bipolar(X[cfg.idx_pc:cfg.idx_pc + cfg.logn, 0])


# ──────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = LoomStandardConfig(s=32, m=8, n=64, N=8)
    print(f"LoomStandardConfig: d_model={cfg.d_model} "
          f"(V4 was {cfg.d_model - 1}), idx_col0={cfg.idx_col0}")

    ext = LoomComputerStandard(cfg)
    print(f"Layers: {len(ext.layers)}")
    for i, layer in enumerate(ext.layers):
        print(f"  L{i+1}: heads={layer.num_heads}, "
              f"W1={list(layer.W1.shape)}, "
              f"b1={list(layer.b1.shape)}, "
              f"W2={list(layer.W2.shape)}, "
              f"b2={list(layer.b2.shape)}")

    # Smoke test: INC instruction
    memory = [5, 0, 0, 0, 0, 0, 0, 0]
    commands = [
        (OP_INC, cfg.s + 0, 0),
        (OP_HALT, 0, 0),
    ]
    X = init_state_standard(cfg, memory, commands)
    with torch.no_grad():
        X = ext.step(X)
    result = read_memory_standard(X, cfg)
    print(f"\nINC test: mem[0]=5 -> {result[0]} (expected 6)")
    assert result[0] == 6, f"FAIL: got {result[0]}"
    print("PASS")

# Backwards-compatibility aliases
ExtendedStandardTransformerV4 = LoomComputerStandard
LoomStandardTransformer = LoomComputerStandard  # alias
StandardConfig = LoomStandardConfig
