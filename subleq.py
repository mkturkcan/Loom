"""
Neural SUBLEQ Computer - PyTorch Implementation

A clean PyTorch implementation of the Looped Transformer SUBLEQ computer,
directly translated from the original subleq.py.

SUBLEQ (SUbtract and Branch if Less than or EQual to zero) instruction:
    mem[b] = mem[b] - mem[a]
    if mem[b] <= 0: PC = c
    else: PC = PC + 1

The computer state is a 2D tensor X of shape (d_model, n_cols) where:
- Columns 0 to s-1: Scratchpad (working memory)
- Columns s to s+m-1: Program memory
- Columns s+m to n-1: Instruction memory (commands)

Values are encoded in BIPOLAR format: -1 for 0, +1 for 1
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SUBLEQConfig:
    """Configuration for the SUBLEQ computer."""
    s: int = 8          # Scratchpad size
    m: int = 8          # Memory size  
    n: int = 32         # Total columns (must be power of 2)
    N: int = 8          # Precision (bits per memory value)
    lam: float = 10.0   # Softmax temperature
    
    def __post_init__(self):
        self.logn = int(np.log2(self.n))
        
        # Row counts
        self.nrows_cmds = 3 * self.logn
        self.nrows_memory = self.N
        self.nrows_scratchpad = 3 * self.logn + 2 * self.N
        self.nrows_pc = self.logn
        self.nrows_pos_enc = self.logn
        self.nrows_buffer = max(3 * self.logn, 2 * self.N)
        self.nrows_indicator = 1
        
        # Total rows
        self.d_model = (self.nrows_cmds + self.nrows_memory + self.nrows_scratchpad +
                       self.nrows_pc + self.nrows_pos_enc + self.nrows_buffer + 
                       self.nrows_indicator)
        
        # Row indices
        self.idx_memory = self.nrows_cmds
        self.idx_scratchpad = self.idx_memory + self.nrows_memory
        self.idx_pc = self.idx_scratchpad + self.nrows_scratchpad
        self.idx_pos_enc = self.idx_pc + self.nrows_pc
        self.idx_buffer = self.idx_pos_enc + self.nrows_pos_enc
        
        # Derived
        self.idx_scratch_cmd = self.idx_scratchpad + 2 * self.N


def to_bipolar(x: int, bits: int) -> torch.Tensor:
    """Convert unsigned integer to bipolar encoding (MSB-first)."""
    # Original format: [b_n, ..., b_1] where b_n is MSB (index 0), b_1 is LSB (index bits-1)
    result = []
    for i in range(bits - 1, -1, -1):
        bit = (x >> i) & 1
        result.append(2.0 * bit - 1.0)
    return torch.tensor(result, dtype=torch.float32)


def signed_to_bipolar(x: int, bits: int) -> torch.Tensor:
    """Convert signed integer to bipolar encoding (MSB-first)."""
    if x < 0:
        x = x + (1 << bits)
    return to_bipolar(x, bits)


def from_bipolar(b: torch.Tensor) -> int:
    """Convert bipolar encoding (MSB-first) to unsigned integer."""
    result = 0
    bits = len(b)
    for i in range(bits):
        if b[i].item() > 0:
            result |= (1 << (bits - 1 - i))
    return result


def signed_from_bipolar(b: torch.Tensor) -> int:
    """Convert bipolar encoding (MSB-first) to signed integer."""
    val = from_bipolar(b)
    bits = len(b)
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def _attn_head(Q, K, V, X, lam, use_softmax=False):
    r"""Attention mechanism for one head.

    Computes V @ X @ A where A is an n x n attention matrix derived from
    scores S = (K X)^T (Q X).  Two execution modes are supported:

    **Argmax mode** (default).  For each source row i of S, select the
    top-2 target columns by score.  If the gap between the first and
    second largest scores is below a threshold (< 1.0), assign weight
    0.5 to each; otherwise assign weight 1.0 to the winner.  Rows whose
    projected key K X_{:,i} is zero contribute nothing.  This produces
    a piecewise-constant attention matrix that is numerically exact
    regardless of the number of sequential steps.

    **Softmax STE mode** (use_softmax=True).  The forward pass uses the
    argmax weights above; the backward pass differentiates through a
    masked softmax(lambda * S, dim=0) via the straight-through estimator
    (STE).  This provides valid gradients for fine-tuning or learning
    while preserving the exact forward semantics.

    Score structure and tie analysis
    --------------------------------
    All attention heads in this architecture use symmetric Q = K.  Each
    head's Q/K matrix projects a pair of row regions additively: an
    *address field* (stored in column 0 only) and a *position encoding*
    (unique per column).  For column 0 targeting column t:

        score(0, t) = (addr + pos_enc(0))^T (addr + pos_enc(t))

    When addr = pos_enc(t) (the intended match), this equals
    ||addr||^2 + addr . pos_enc(0) + pos_enc(t) . addr + pos_enc(0) . pos_enc(t).
    Column 0 has pos_enc = 0 (by construction), so the cross-terms
    involving pos_enc(0) vanish, giving score(0, t) = ||addr||^2 = logn.
    The self-score score(0, 0) = ||addr + 0||^2 = ||addr||^2 = logn,
    producing the designed 50/50 tie between column 0 and the target.

    For any bystander column b != t with pos_enc(b) != addr:

        score(0, b) = addr . pos_enc(b) < logn

    since pos_enc(b) differs from addr in at least one bit (Hamming
    distance >= 1), giving score <= logn - 2.  The gap of >= 2.0
    between the winning pair {0, t} and all bystanders exceeds the tie
    threshold of 1.0, so the argmax unambiguously selects the correct
    pair.

    Among bystander columns, self-scores may be tied (all non-addressed
    memory columns with identical content produce ||0 + pos_enc||^2 =
    logn).  This tie is inconsequential: every bystander's score against
    column 0 is strictly below the winners, so no bystander enters the
    top-2 set of any participating row.  The V-mapped contributions to
    bystander columns land in buffer/scratchpad rows that are never read
    by subsequent gated FFN layers.

    Parameters
    ----------
    Q, K : Tensor, shape (q_rows, d)
        Query and key projection matrices.  Typically Q = K (symmetric).
    V : Tensor, shape (d, d)
        Value projection matrix.
    X : Tensor, shape (d, n)
        State tensor (bipolar +/-1 entries).
    lam : float
        Softmax temperature (used only in STE mode).
    use_softmax : bool
        If True, use the STE mode for differentiable backward pass.
    """
    n = X.shape[1]
    QX = Q @ X   # q_rows x n
    KX = K @ X   # q_rows x n
    VX = V @ X   # d x n

    # Skip no-op heads (all zeros)
    if QX.abs().max() < 1e-6 and KX.abs().max() < 1e-6:
        return torch.zeros_like(VX)

    scores = KX.T @ QX  # n x n

    if use_softmax:
        # Straight-Through Estimator: forward uses argmax (exact),
        # backward uses softmax gradient (differentiable).
        # This gives correct execution AND valid gradients for training.
        kx_norm = KX.pow(2).sum(dim=0)  # [n]
        active_mask = (kx_norm > 1e-6).float()  # [n]
        mask_bias = (1.0 - active_mask).unsqueeze(1) * (-1e9)  # [n, 1]
        softmax_weights = F.softmax(lam * scores + mask_bias, dim=0)  # [n, n]

        # Build argmax weights (detached, no gradient)
        top2_vals, top2_idx = scores.topk(2, dim=1)
        tied = (top2_vals[:, 0] - top2_vals[:, 1]) < 1.0
        argmax_weights = torch.zeros_like(scores)
        for i in range(n):
            if KX[:, i].abs().max() < 1e-6:
                continue
            j1, j2 = top2_idx[i, 0].item(), top2_idx[i, 1].item()
            if tied[i]:
                argmax_weights[i, j1] = 0.5
                argmax_weights[i, j2] = 0.5
            else:
                argmax_weights[i, j1] = 1.0

        # STE: use argmax values but softmax gradient
        weights = softmax_weights + (argmax_weights - softmax_weights).detach()
        return V @ (X @ weights)
    else:
        # Argmax: top-2 per row with tie detection
        top2_vals, top2_idx = scores.topk(2, dim=1)  # n x 2
        tied = (top2_vals[:, 0] - top2_vals[:, 1]) < 1.0

        result = torch.zeros_like(VX)
        for i in range(n):
            # Skip zero-KX rows
            if KX[:, i].abs().max() < 1e-6:
                continue
            j1 = top2_idx[i, 0].item()
            j2 = top2_idx[i, 1].item()
            if tied[i]:
                result[:, j1] += 0.5 * VX[:, i]
                result[:, j2] += 0.5 * VX[:, i]
            else:
                result[:, j1] += VX[:, i]
        return result


class TransformerLayer:
    r"""One transformer layer: multi-head attention followed by a feedforward
    network, both with residual connections.

    Forward pass:
        A = X + \sum_h attn_head(Q_h, K_h, V_h, X)
        Y = A + W_2 \, \mathrm{ReLU}(W_1 A + b_1) + b_2

    Attention uses argmax by default (numerically exact for arbitrary-length
    execution).  Set use_softmax=True for a differentiable backward pass via
    the straight-through estimator; see _attn_head for the full analysis.
    """
    
    def __init__(self, cfg: SUBLEQConfig, num_rows_Q: int, num_rows_W: int, num_heads: int = 1):
        self.cfg = cfg
        self.num_heads = num_heads
        self.lam = cfg.lam
        
        d = cfg.d_model
        n = cfg.n
        
        if num_heads == 1:
            self.Q = torch.zeros(num_rows_Q, d)
            self.K = torch.zeros(num_rows_Q, d)
            self.V = torch.zeros(d, d)
        elif num_heads == 2:
            self.Q1 = torch.zeros(num_rows_Q, d)
            self.K1 = torch.zeros(num_rows_Q, d)
            self.V1 = torch.zeros(d, d)
            self.Q2 = torch.zeros(num_rows_Q, d)
            self.K2 = torch.zeros(num_rows_Q, d)
            self.V2 = torch.zeros(d, d)
        else:  # 3 heads
            self.Q1 = torch.zeros(num_rows_Q, d)
            self.K1 = torch.zeros(num_rows_Q, d)
            self.V1 = torch.zeros(d, d)
            self.Q2 = torch.zeros(num_rows_Q, d)
            self.K2 = torch.zeros(num_rows_Q, d)
            self.V2 = torch.zeros(d, d)
            self.Q3 = torch.zeros(num_rows_Q, d)
            self.K3 = torch.zeros(num_rows_Q, d)
            self.V3 = torch.zeros(d, d)
        
        self.W1 = torch.zeros(num_rows_W, d)
        self.b1 = torch.zeros(num_rows_W, n)
        self.W2 = torch.zeros(d, num_rows_W)
        self.b2 = torch.zeros(d, n)
    
    def forward(self, X: torch.Tensor, use_softmax: bool = False) -> torch.Tensor:
        """Forward pass. use_softmax=True for differentiable (training) mode."""
        if self.num_heads == 1:
            attn = X + _attn_head(self.Q, self.K, self.V, X, self.lam, use_softmax)
        elif self.num_heads == 2:
            attn = X + _attn_head(self.Q1, self.K1, self.V1, X, self.lam, use_softmax) + \
                       _attn_head(self.Q2, self.K2, self.V2, X, self.lam, use_softmax)
        else:  # 3 heads
            attn = X + _attn_head(self.Q1, self.K1, self.V1, X, self.lam, use_softmax) + \
                       _attn_head(self.Q2, self.K2, self.V2, X, self.lam, use_softmax) + \
                       _attn_head(self.Q3, self.K3, self.V3, X, self.lam, use_softmax)
        
        # FFN
        ff1 = F.relu(self.W1 @ attn + self.b1)
        output = attn + self.W2 @ ff1 + self.b2
        
        return output


class NeuralSUBLEQ:
    """Neural SUBLEQ computer using transformer layers."""
    
    def __init__(self, cfg: SUBLEQConfig):
        self.cfg = cfg
        self.layers = self._build_layers()
    
    def _build_layers(self) -> List[TransformerLayer]:
        """Build all transformer layers with initialized weights."""
        cfg = self.cfg
        logn = cfg.logn
        N = cfg.N
        
        layers = []
        
        # Layer 1: Read Instruction
        layers.append(self._init_read_inst())
        
        # Layer 2: Read Memory (2 heads)
        layers.append(self._init_read_mem())
        
        # Layers 3-5: Subtract Memory
        layers.extend(self._init_subtract_mem())
        
        # Layer 6: Write Memory
        layers.append(self._init_write_mem())
        
        # Layers 7-9: Conditional Branching
        layers.extend(self._init_cond_branch())
        
        # Layer 10: Error Correction
        layers.append(self._init_error_correction())
        
        return layers
    
    def _init_read_inst(self) -> TransformerLayer:
        """Initialize Read Instruction layer."""
        cfg = self.cfg
        size_inst = cfg.nrows_cmds
        size_pos_enc = cfg.logn
        
        layer = TransformerLayer(cfg, cfg.logn, 4 * size_inst, num_heads=1)
        
        # Q and K: Extract PC and position encoding
        layer.Q[:, cfg.idx_pc:cfg.idx_pos_enc] = torch.eye(size_pos_enc)
        layer.Q[:, cfg.idx_pos_enc:cfg.idx_buffer] = torch.eye(size_pos_enc)
        layer.K = layer.Q.clone()
        
        # V: Read commands to buffer
        layer.V[cfg.idx_buffer:cfg.idx_buffer+size_inst, :cfg.nrows_cmds] = torch.eye(size_inst)
        layer.V[cfg.idx_buffer:cfg.idx_buffer+size_inst, cfg.idx_scratch_cmd:cfg.idx_scratch_cmd+size_inst] = torch.eye(size_inst)
        
        # FFN weights
        large_const = 100.0
        layer.W1[:size_inst, cfg.idx_scratch_cmd:cfg.idx_scratch_cmd+size_inst] = -2 * torch.eye(size_inst)
        layer.W1[:size_inst, cfg.idx_buffer:cfg.idx_buffer+size_inst] = 2 * torch.eye(size_inst)
        layer.W1[:size_inst, -1] = large_const
        
        layer.W1[size_inst:2*size_inst, cfg.idx_scratch_cmd:cfg.idx_scratch_cmd+size_inst] = 2 * torch.eye(size_inst)
        layer.W1[size_inst:2*size_inst, cfg.idx_buffer:cfg.idx_buffer+size_inst] = -2 * torch.eye(size_inst)
        layer.W1[size_inst:2*size_inst, -1] = large_const
        
        layer.W1[2*size_inst:3*size_inst, cfg.idx_buffer:cfg.idx_buffer+size_inst] = torch.eye(size_inst)
        layer.W1[3*size_inst:4*size_inst, cfg.idx_buffer:cfg.idx_buffer+size_inst] = -torch.eye(size_inst)
        
        layer.b1[:2*size_inst, :] = -large_const
        
        layer.W2[cfg.idx_scratch_cmd:cfg.idx_scratch_cmd+size_inst, :size_inst] = torch.eye(size_inst)
        layer.W2[cfg.idx_scratch_cmd:cfg.idx_scratch_cmd+size_inst, size_inst:2*size_inst] = -torch.eye(size_inst)
        layer.W2[cfg.idx_buffer:cfg.idx_buffer+size_inst, 2*size_inst:3*size_inst] = -torch.eye(size_inst)
        layer.W2[cfg.idx_buffer:cfg.idx_buffer+size_inst, 3*size_inst:4*size_inst] = torch.eye(size_inst)
        
        return layer
    
    def _init_read_mem(self) -> TransformerLayer:
        """Initialize Read Memory layer (2 heads)."""
        cfg = self.cfg
        size_mem = cfg.N
        size_pos_enc = cfg.logn
        
        layer = TransformerLayer(cfg, cfg.logn, 8 * size_mem, num_heads=2)
        
        # Head 1: Read mem[a]
        layer.Q1[:, cfg.idx_scratch_cmd:cfg.idx_scratch_cmd+size_pos_enc] = torch.eye(size_pos_enc)
        layer.Q1[:, cfg.idx_pos_enc:cfg.idx_pos_enc+size_pos_enc] = torch.eye(size_pos_enc)
        layer.K1 = layer.Q1.clone()
        
        layer.V1[cfg.idx_buffer:cfg.idx_buffer+size_mem, cfg.idx_memory:cfg.idx_memory+size_mem] = torch.eye(size_mem)
        layer.V1[cfg.idx_buffer:cfg.idx_buffer+size_mem, cfg.idx_scratchpad:cfg.idx_scratchpad+size_mem] = torch.eye(size_mem)
        
        # Head 2: Read mem[b]
        layer.Q2[:, cfg.idx_scratch_cmd+size_pos_enc:cfg.idx_scratch_cmd+2*size_pos_enc] = torch.eye(size_pos_enc)
        layer.Q2[:, cfg.idx_pos_enc:cfg.idx_pos_enc+size_pos_enc] = torch.eye(size_pos_enc)
        layer.K2 = layer.Q2.clone()
        
        layer.V2[cfg.idx_buffer+size_mem:cfg.idx_buffer+2*size_mem, cfg.idx_memory:cfg.idx_memory+size_mem] = torch.eye(size_mem)
        layer.V2[cfg.idx_buffer+size_mem:cfg.idx_buffer+2*size_mem, cfg.idx_scratchpad+size_mem:cfg.idx_scratchpad+2*size_mem] = torch.eye(size_mem)
        
        # FFN
        large_const = 100.0
        
        # For mem[a]
        layer.W1[:size_mem, cfg.idx_scratchpad:cfg.idx_scratchpad+size_mem] = -2 * torch.eye(size_mem)
        layer.W1[:size_mem, cfg.idx_buffer:cfg.idx_buffer+size_mem] = 2 * torch.eye(size_mem)
        layer.W1[:size_mem, -1] = large_const
        layer.W1[size_mem:2*size_mem, cfg.idx_scratchpad:cfg.idx_scratchpad+size_mem] = 2 * torch.eye(size_mem)
        layer.W1[size_mem:2*size_mem, cfg.idx_buffer:cfg.idx_buffer+size_mem] = -2 * torch.eye(size_mem)
        layer.W1[size_mem:2*size_mem, -1] = large_const
        layer.W1[2*size_mem:3*size_mem, cfg.idx_buffer:cfg.idx_buffer+size_mem] = torch.eye(size_mem)
        layer.W1[3*size_mem:4*size_mem, cfg.idx_buffer:cfg.idx_buffer+size_mem] = -torch.eye(size_mem)
        
        # For mem[b]
        offset = 4 * size_mem
        layer.W1[offset:offset+size_mem, cfg.idx_scratchpad+size_mem:cfg.idx_scratchpad+2*size_mem] = -2 * torch.eye(size_mem)
        layer.W1[offset:offset+size_mem, cfg.idx_buffer+size_mem:cfg.idx_buffer+2*size_mem] = 2 * torch.eye(size_mem)
        layer.W1[offset:offset+size_mem, -1] = large_const
        layer.W1[offset+size_mem:offset+2*size_mem, cfg.idx_scratchpad+size_mem:cfg.idx_scratchpad+2*size_mem] = 2 * torch.eye(size_mem)
        layer.W1[offset+size_mem:offset+2*size_mem, cfg.idx_buffer+size_mem:cfg.idx_buffer+2*size_mem] = -2 * torch.eye(size_mem)
        layer.W1[offset+size_mem:offset+2*size_mem, -1] = large_const
        layer.W1[offset+2*size_mem:offset+3*size_mem, cfg.idx_buffer+size_mem:cfg.idx_buffer+2*size_mem] = torch.eye(size_mem)
        layer.W1[offset+3*size_mem:offset+4*size_mem, cfg.idx_buffer+size_mem:cfg.idx_buffer+2*size_mem] = -torch.eye(size_mem)
        
        layer.b1[:2*size_mem, :] = -large_const
        layer.b1[offset:offset+2*size_mem, :] = -large_const
        
        layer.W2[cfg.idx_scratchpad:cfg.idx_scratchpad+size_mem, :size_mem] = torch.eye(size_mem)
        layer.W2[cfg.idx_scratchpad:cfg.idx_scratchpad+size_mem, size_mem:2*size_mem] = -torch.eye(size_mem)
        layer.W2[cfg.idx_buffer:cfg.idx_buffer+size_mem, 2*size_mem:3*size_mem] = -torch.eye(size_mem)
        layer.W2[cfg.idx_buffer:cfg.idx_buffer+size_mem, 3*size_mem:4*size_mem] = torch.eye(size_mem)
        layer.W2[cfg.idx_scratchpad+size_mem:cfg.idx_scratchpad+2*size_mem, offset:offset+size_mem] = torch.eye(size_mem)
        layer.W2[cfg.idx_scratchpad+size_mem:cfg.idx_scratchpad+2*size_mem, offset+size_mem:offset+2*size_mem] = -torch.eye(size_mem)
        layer.W2[cfg.idx_buffer+size_mem:cfg.idx_buffer+2*size_mem, offset+2*size_mem:offset+3*size_mem] = -torch.eye(size_mem)
        layer.W2[cfg.idx_buffer+size_mem:cfg.idx_buffer+2*size_mem, offset+3*size_mem:offset+4*size_mem] = torch.eye(size_mem)
        
        return layer
    
    def _init_subtract_mem(self) -> List[TransformerLayer]:
        """Initialize 3 subtract memory layers."""
        cfg = self.cfg
        N = cfg.N
        size_mem = N
        
        # Layer 1: Flip bits
        layer1 = TransformerLayer(cfg, cfg.logn, size_mem, num_heads=1)
        layer1.W1[:size_mem, cfg.idx_scratchpad:cfg.idx_scratchpad+size_mem] = -torch.eye(size_mem)
        layer1.W2[cfg.idx_buffer:cfg.idx_buffer+size_mem, :size_mem] = 2 * torch.eye(size_mem)
        layer1.b2[cfg.idx_buffer:cfg.idx_buffer+size_mem, 0] = -1.0
        
        # Layer 2: Add 1 (for two's complement)
        layer2 = TransformerLayer(cfg, cfg.logn, 6 * N, num_heads=1)
        ref = cfg.idx_buffer
        for i in range(1, N + 1):
            for j in range(1, i + 1):
                elem = (2**(j-1)) / 2
                row_base = 6 * (N - i)
                col = ref + N - j
                layer2.W1[row_base, col] = elem
                layer2.W1[row_base + 1, col] = elem
                layer2.W1[row_base + 2, col] = -elem
                layer2.W1[row_base + 3, col] = -elem
                layer2.W1[row_base + 4, col] = elem
                layer2.W1[row_base + 5, col] = elem
            
            s_bias = (2**i - 1) / 2 + 1
            row_base = 6 * (N - i)
            layer2.b1[row_base, 0] = s_bias - 2**(i-1) + 1
            layer2.b1[row_base + 1, 0] = s_bias - 2**(i-1)
            layer2.b1[row_base + 2, 0] = -s_bias + 2**i
            layer2.b1[row_base + 3, 0] = -s_bias + 2**i - 1
            layer2.b1[row_base + 4, 0] = s_bias - 3 * 2**(i-1) + 1
            layer2.b1[row_base + 5, 0] = s_bias - 3 * 2**(i-1)
            
            layer2.W2[cfg.idx_buffer + size_mem + N - i, row_base:row_base + 6] = 2 * torch.tensor([1, -1, 1, -1, 1, -1], dtype=torch.float32)
        
        layer2.b2[cfg.idx_buffer + size_mem:cfg.idx_buffer + 2*size_mem, 0] = -3.0
        
        # Layer 3: Complete subtraction
        layer3 = TransformerLayer(cfg, cfg.logn, 14 * N, num_heads=1)
        ref = cfg.idx_buffer + size_mem
        ref1 = cfg.idx_scratchpad
        ref2 = cfg.idx_scratchpad + size_mem
        
        for i in range(1, N + 1):
            for j in range(1, i + 1):
                elem = (2**(j-1)) / 2
                row_base = 6 * (N - i)
                layer3.W1[row_base, ref + N - j] = elem
                layer3.W1[row_base + 1, ref + N - j] = elem
                layer3.W1[row_base + 2, ref + N - j] = -elem
                layer3.W1[row_base + 3, ref + N - j] = -elem
                layer3.W1[row_base + 4, ref + N - j] = elem
                layer3.W1[row_base + 5, ref + N - j] = elem
                layer3.W1[row_base, ref2 + N - j] = elem
                layer3.W1[row_base + 1, ref2 + N - j] = elem
                layer3.W1[row_base + 2, ref2 + N - j] = -elem
                layer3.W1[row_base + 3, ref2 + N - j] = -elem
                layer3.W1[row_base + 4, ref2 + N - j] = elem
                layer3.W1[row_base + 5, ref2 + N - j] = elem
            
            s_bias = 2**i - 1
            row_base = 6 * (N - i)
            layer3.b1[row_base, 0] = s_bias - 2**(i-1) + 1
            layer3.b1[row_base + 1, 0] = s_bias - 2**(i-1)
            layer3.b1[row_base + 2, 0] = -s_bias + 2**i
            layer3.b1[row_base + 3, 0] = -s_bias + 2**i - 1
            layer3.b1[row_base + 4, 0] = s_bias - 3 * 2**(i-1) + 1
            layer3.b1[row_base + 5, 0] = s_bias - 3 * 2**(i-1)
            
            layer3.W2[ref2 + N - i, row_base:row_base + 6] = 2 * torch.tensor([1, -1, 1, -1, 1, -1], dtype=torch.float32)
        
        layer3.b2[ref2:ref2 + size_mem, 0] = -3.0
        
        # Zero out buffers
        offset = 6 * N
        layer3.W1[offset:offset + N, cfg.idx_buffer:cfg.idx_buffer + size_mem] = torch.eye(size_mem)
        layer3.W1[offset + N:offset + 2*N, cfg.idx_buffer:cfg.idx_buffer + size_mem] = -torch.eye(size_mem)
        layer3.W1[offset + 2*N:offset + 3*N, cfg.idx_buffer + size_mem:cfg.idx_buffer + 2*size_mem] = torch.eye(size_mem)
        layer3.W1[offset + 3*N:offset + 4*N, cfg.idx_buffer + size_mem:cfg.idx_buffer + 2*size_mem] = -torch.eye(size_mem)
        layer3.W2[cfg.idx_buffer:cfg.idx_buffer + size_mem, offset:offset + N] = -torch.eye(size_mem)
        layer3.W2[cfg.idx_buffer:cfg.idx_buffer + size_mem, offset + N:offset + 2*N] = torch.eye(size_mem)
        layer3.W2[cfg.idx_buffer + size_mem:cfg.idx_buffer + 2*size_mem, offset + 2*N:offset + 3*N] = -torch.eye(size_mem)
        layer3.W2[cfg.idx_buffer + size_mem:cfg.idx_buffer + 2*size_mem, offset + 3*N:offset + 4*N] = torch.eye(size_mem)
        
        offset2 = offset + 4*N
        layer3.W1[offset2:offset2 + N, ref1:ref1 + size_mem] = torch.eye(size_mem)
        layer3.W1[offset2 + N:offset2 + 2*N, ref1:ref1 + size_mem] = -torch.eye(size_mem)
        layer3.W1[offset2 + 2*N:offset2 + 3*N, ref2:ref2 + size_mem] = torch.eye(size_mem)
        layer3.W1[offset2 + 3*N:offset2 + 4*N, ref2:ref2 + size_mem] = -torch.eye(size_mem)
        layer3.W2[ref1:ref1 + size_mem, offset2:offset2 + N] = -torch.eye(size_mem)
        layer3.W2[ref1:ref1 + size_mem, offset2 + N:offset2 + 2*N] = torch.eye(size_mem)
        layer3.W2[ref2:ref2 + size_mem, offset2 + 2*N:offset2 + 3*N] = -torch.eye(size_mem)
        layer3.W2[ref2:ref2 + size_mem, offset2 + 3*N:offset2 + 4*N] = torch.eye(size_mem)
        
        return [layer1, layer2, layer3]
    
    def _init_write_mem(self) -> TransformerLayer:
        """Initialize Write Memory layer."""
        cfg = self.cfg
        size_mem = cfg.N
        size_pos_enc = cfg.logn
        
        layer = TransformerLayer(cfg, cfg.logn, 4 * size_mem, num_heads=1)
        
        layer.Q[:, cfg.idx_scratch_cmd + size_pos_enc:cfg.idx_scratch_cmd + 2*size_pos_enc] = torch.eye(size_pos_enc)
        layer.Q[:, cfg.idx_pos_enc:cfg.idx_pos_enc + size_pos_enc] = torch.eye(size_pos_enc)
        layer.K = layer.Q.clone()
        
        layer.V[cfg.idx_buffer:cfg.idx_buffer + size_mem, cfg.idx_memory:cfg.idx_memory + size_mem] = torch.eye(size_mem)
        layer.V[cfg.idx_buffer:cfg.idx_buffer + size_mem, cfg.idx_scratchpad + size_mem:cfg.idx_scratchpad + 2*size_mem] = torch.eye(size_mem)
        
        large_const = 100.0
        layer.W1[:size_mem, cfg.idx_memory:cfg.idx_memory + size_mem] = -2 * torch.eye(size_mem)
        layer.W1[:size_mem, cfg.idx_buffer:cfg.idx_buffer + size_mem] = 2 * torch.eye(size_mem)
        layer.W1[:size_mem, -1] = -large_const
        layer.W1[size_mem:2*size_mem, cfg.idx_memory:cfg.idx_memory + size_mem] = 2 * torch.eye(size_mem)
        layer.W1[size_mem:2*size_mem, cfg.idx_buffer:cfg.idx_buffer + size_mem] = -2 * torch.eye(size_mem)
        layer.W1[size_mem:2*size_mem, -1] = -large_const
        layer.W1[2*size_mem:3*size_mem, cfg.idx_buffer:cfg.idx_buffer + size_mem] = torch.eye(size_mem)
        layer.W1[3*size_mem:4*size_mem, cfg.idx_buffer:cfg.idx_buffer + size_mem] = -torch.eye(size_mem)
        
        layer.W2[cfg.idx_memory:cfg.idx_memory + size_mem, :size_mem] = torch.eye(size_mem)
        layer.W2[cfg.idx_memory:cfg.idx_memory + size_mem, size_mem:2*size_mem] = -torch.eye(size_mem)
        layer.W2[cfg.idx_buffer:cfg.idx_buffer + size_mem, 2*size_mem:3*size_mem] = -torch.eye(size_mem)
        layer.W2[cfg.idx_buffer:cfg.idx_buffer + size_mem, 3*size_mem:4*size_mem] = torch.eye(size_mem)
        
        return layer
    
    def _init_cond_branch(self) -> List[TransformerLayer]:
        """Initialize 3 conditional branching layers."""
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        size_mem = N
        
        # Layer 1: Compute flag
        layer1 = TransformerLayer(cfg, logn, 2 + 2 * size_mem, num_heads=1)
        ref1_cb = cfg.idx_scratchpad + size_mem
        ref2_cb = cfg.idx_scratchpad
        
        layer1.W1[0, ref1_cb] = 1.0
        layer1.W1[1, ref1_cb:ref1_cb + size_mem] = -1.0
        layer1.W1[2:2 + size_mem, ref1_cb:ref1_cb + size_mem] = torch.eye(size_mem)
        layer1.W1[2 + size_mem:2 + 2*size_mem, ref1_cb:ref1_cb + size_mem] = -torch.eye(size_mem)
        layer1.b1[1, 0] = -N + 1
        
        layer1.W2[ref2_cb, 0] = 1.0
        layer1.W2[ref2_cb, 1] = 1.0
        layer1.W2[ref1_cb:ref1_cb + size_mem, 2:2 + size_mem] = -torch.eye(size_mem)
        layer1.W2[ref1_cb:ref1_cb + size_mem, 2 + size_mem:2 + 2*size_mem] = torch.eye(size_mem)
        
        # Layer 2: Compute PC+1
        layer2 = TransformerLayer(cfg, logn, 8 * logn, num_heads=1)
        unit = logn
        ref_pc = cfg.idx_pc
        
        for i in range(1, unit + 1):
            for j in range(1, i + 1):
                elem = (2**(j-1)) / 2
                row_base = 6 * (unit - i)
                layer2.W1[row_base, ref_pc + unit - j] = elem
                layer2.W1[row_base + 1, ref_pc + unit - j] = elem
                layer2.W1[row_base + 2, ref_pc + unit - j] = -elem
                layer2.W1[row_base + 3, ref_pc + unit - j] = -elem
                layer2.W1[row_base + 4, ref_pc + unit - j] = elem
                layer2.W1[row_base + 5, ref_pc + unit - j] = elem
            
            s_bias = (2**i - 1) / 2 + 1
            row_base = 6 * (unit - i)
            layer2.b1[row_base, 0] = s_bias - 2**(i-1) + 1
            layer2.b1[row_base + 1, 0] = s_bias - 2**(i-1)
            layer2.b1[row_base + 2, 0] = -s_bias + 2**i
            layer2.b1[row_base + 3, 0] = -s_bias + 2**i - 1
            layer2.b1[row_base + 4, 0] = s_bias - 3 * 2**(i-1) + 1
            layer2.b1[row_base + 5, 0] = s_bias - 3 * 2**(i-1)
            
            layer2.W2[ref_pc + unit - i, row_base:row_base + 6] = 2 * torch.tensor([1, -1, 1, -1, 1, -1], dtype=torch.float32)
        
        layer2.b2[ref_pc:ref_pc + unit, 0] = -3.0
        
        offset_pc = 6 * unit
        layer2.W1[offset_pc:offset_pc + unit, ref_pc:ref_pc + unit] = torch.eye(unit)
        layer2.W1[offset_pc + unit:offset_pc + 2*unit, ref_pc:ref_pc + unit] = -torch.eye(unit)
        layer2.W2[ref_pc:ref_pc + unit, offset_pc:offset_pc + unit] = -torch.eye(unit)
        layer2.W2[ref_pc:ref_pc + unit, offset_pc + unit:offset_pc + 2*unit] = torch.eye(unit)
        
        # Layer 3: Select PC
        layer3 = TransformerLayer(cfg, logn, 6 * logn + 2, num_heads=1)
        ref_c = cfg.idx_scratch_cmd + 2 * logn
        ref_flag = cfg.idx_scratchpad
        
        layer3.W1[:unit, ref_c:ref_c + unit] = torch.eye(unit)
        layer3.W1[:unit, ref_flag] = 1.0
        layer3.b1[:unit, 0] = -1.0
        layer3.W1[unit:2*unit, ref_c:ref_c + unit] = -torch.eye(unit)
        layer3.W1[unit:2*unit, ref_flag] = 1.0
        layer3.b1[unit:2*unit, 0] = -1.0
        
        layer3.W1[2*unit:3*unit, ref_pc:ref_pc + unit] = torch.eye(unit)
        layer3.W1[2*unit:3*unit, ref_flag] = -1.0
        layer3.W1[3*unit:4*unit, ref_pc:ref_pc + unit] = -torch.eye(unit)
        layer3.W1[3*unit:4*unit, ref_flag] = -1.0
        
        layer3.W1[4*unit:5*unit, ref_pc:ref_pc + unit] = torch.eye(unit)
        layer3.W1[5*unit:6*unit, ref_pc:ref_pc + unit] = -torch.eye(unit)
        
        layer3.W2[ref_pc:ref_pc + unit, :unit] = torch.eye(unit)
        layer3.W2[ref_pc:ref_pc + unit, unit:2*unit] = -torch.eye(unit)
        layer3.W2[ref_pc:ref_pc + unit, 2*unit:3*unit] = torch.eye(unit)
        layer3.W2[ref_pc:ref_pc + unit, 3*unit:4*unit] = -torch.eye(unit)
        layer3.W2[ref_pc:ref_pc + unit, 4*unit:5*unit] = -torch.eye(unit)
        layer3.W2[ref_pc:ref_pc + unit, 5*unit:6*unit] = torch.eye(unit)
        
        layer3.W1[6*unit, ref_flag] = 1.0
        layer3.W1[6*unit + 1, ref_flag] = -1.0
        layer3.W2[ref_flag, 6*unit] = -1.0
        layer3.W2[ref_flag, 6*unit + 1] = 1.0
        
        return [layer1, layer2, layer3]
    
    def _init_error_correction(self) -> TransformerLayer:
        """Initialize Error Correction layer."""
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        large_const = 100.0
        
        # Extra rows for clearing scratchpad/buffer rows in non-scratchpad columns.
        num_rows = 6 * (N + 4 * logn) + 2 * (cfg.nrows_scratchpad + cfg.nrows_buffer)
        layer = TransformerLayer(cfg, logn, num_rows, num_heads=1)
        
        eps = 0.1
        unit = logn
        ref1_ec = cfg.idx_memory
        ref2_ec = cfg.idx_pc
        
        # Memory correction
        layer.W1[:N, ref1_ec:ref1_ec + N] = torch.eye(N)
        layer.W1[N:2*N, ref1_ec:ref1_ec + N] = torch.eye(N)
        layer.W1[2*N:3*N, ref1_ec:ref1_ec + N] = torch.eye(N)
        layer.W1[3*N:4*N, ref1_ec:ref1_ec + N] = torch.eye(N)
        layer.W1[4*N:5*N, ref1_ec:ref1_ec + N] = -torch.eye(N)
        layer.W1[5*N:6*N, ref1_ec:ref1_ec + N] = torch.eye(N)
        
        layer.b1[:N, :] = 1 - eps
        layer.b1[N:2*N, :] = eps
        layer.b1[2*N:3*N, :] = -eps
        layer.b1[3*N:4*N, :] = -1 + eps
        
        scale = 1 / (1 - 2*eps)
        layer.W2[ref1_ec:ref1_ec + N, :N] = scale * torch.eye(N)
        layer.W2[ref1_ec:ref1_ec + N, N:2*N] = -scale * torch.eye(N)
        layer.W2[ref1_ec:ref1_ec + N, 2*N:3*N] = scale * torch.eye(N)
        layer.W2[ref1_ec:ref1_ec + N, 3*N:4*N] = -scale * torch.eye(N)
        layer.W2[ref1_ec:ref1_ec + N, 4*N:5*N] = torch.eye(N)
        layer.W2[ref1_ec:ref1_ec + N, 5*N:6*N] = -torch.eye(N)
        layer.b2[ref1_ec:ref1_ec + N, :] = -1.0
        
        # PC correction
        offset = 6 * N
        layer.W1[offset:offset + unit, ref2_ec:ref2_ec + unit] = torch.eye(unit)
        layer.W1[offset + unit:offset + 2*unit, ref2_ec:ref2_ec + unit] = torch.eye(unit)
        layer.W1[offset + 2*unit:offset + 3*unit, ref2_ec:ref2_ec + unit] = torch.eye(unit)
        layer.W1[offset + 3*unit:offset + 4*unit, ref2_ec:ref2_ec + unit] = torch.eye(unit)
        layer.W1[offset + 4*unit:offset + 5*unit, ref2_ec:ref2_ec + unit] = -torch.eye(unit)
        layer.W1[offset + 5*unit:offset + 6*unit, ref2_ec:ref2_ec + unit] = torch.eye(unit)
        
        layer.b1[offset:offset + unit, :] = 1 - eps
        layer.b1[offset + unit:offset + 2*unit, :] = eps
        layer.b1[offset + 2*unit:offset + 3*unit, :] = -eps
        layer.b1[offset + 3*unit:offset + 4*unit, :] = -1 + eps

        # Gate PC correction to scratchpad columns only (indicator row = 1)
        layer.W1[offset:offset + 6*unit, -1] = large_const
        layer.b1[offset:offset + 6*unit, :] -= large_const
        
        layer.W2[ref2_ec:ref2_ec + unit, offset:offset + unit] = scale * torch.eye(unit)
        layer.W2[ref2_ec:ref2_ec + unit, offset + unit:offset + 2*unit] = -scale * torch.eye(unit)
        layer.W2[ref2_ec:ref2_ec + unit, offset + 2*unit:offset + 3*unit] = scale * torch.eye(unit)
        layer.W2[ref2_ec:ref2_ec + unit, offset + 3*unit:offset + 4*unit] = -scale * torch.eye(unit)
        layer.W2[ref2_ec:ref2_ec + unit, offset + 4*unit:offset + 5*unit] = torch.eye(unit)
        layer.W2[ref2_ec:ref2_ec + unit, offset + 5*unit:offset + 6*unit] = -torch.eye(unit)
        # Only apply PC bias on scratchpad columns to avoid polluting other columns.
        layer.b2[ref2_ec:ref2_ec + unit, :] = 0.0
        layer.b2[ref2_ec:ref2_ec + unit, :cfg.s] = -1.0

        # Clear PC rows in non-scratchpad columns to prevent read-inst interference.
        mask_offset = offset + 6 * unit
        for i in range(unit):
            r = mask_offset + 2 * i
            layer.W1[r, ref2_ec + i] = 1.0
            layer.W1[r, -1] = -large_const
            layer.b1[r, :] = 0.0
            layer.W2[ref2_ec + i, r] = -1.0

            r = mask_offset + 2 * i + 1
            layer.W1[r, ref2_ec + i] = -1.0
            layer.W1[r, -1] = -large_const
            layer.b1[r, :] = 0.0
            layer.W2[ref2_ec + i, r] = 1.0

        # Clear scratchpad rows in non-scratchpad columns to prevent leakage.
        scratch_offset = mask_offset + 2 * unit
        for i in range(cfg.nrows_scratchpad):
            row_idx = cfg.idx_scratchpad + i
            r = scratch_offset + 2 * i
            layer.W1[r, row_idx] = 1.0
            layer.W1[r, -1] = -large_const
            layer.b1[r, :] = 0.0
            layer.W2[row_idx, r] = -1.0

            r = scratch_offset + 2 * i + 1
            layer.W1[r, row_idx] = -1.0
            layer.W1[r, -1] = -large_const
            layer.b1[r, :] = 0.0
            layer.W2[row_idx, r] = 1.0

        # Clear buffer rows in non-scratchpad columns to prevent instruction fetch contamination.
        buffer_offset = scratch_offset + 2 * cfg.nrows_scratchpad
        for i in range(cfg.nrows_buffer):
            row_idx = cfg.idx_buffer + i
            r = buffer_offset + 2 * i
            layer.W1[r, row_idx] = 1.0
            layer.W1[r, -1] = -large_const
            layer.b1[r, :] = 0.0
            layer.W2[row_idx, r] = -1.0

            r = buffer_offset + 2 * i + 1
            layer.W1[r, row_idx] = -1.0
            layer.W1[r, -1] = -large_const
            layer.b1[r, :] = 0.0
            layer.W2[row_idx, r] = 1.0
        
        return layer
    
    def step(self, X: torch.Tensor, use_softmax: bool = False) -> torch.Tensor:
        """Execute one instruction. use_softmax=True for differentiable mode."""
        for layer in self.layers:
            X = layer.forward(X, use_softmax=use_softmax)
        return X
    
    def run(self, X: torch.Tensor, max_steps: int = 1000) -> Tuple[torch.Tensor, int]:
        """Run until halt (PC=0) or max steps."""
        for step in range(max_steps):
            pc = from_bipolar(X[self.cfg.idx_pc:self.cfg.idx_pc + self.cfg.logn, 0])
            if pc == 0 or pc >= self.cfg.n:
                return X, step
            X = self.step(X)
        return X, max_steps


def init_state(cfg: SUBLEQConfig, memory: List[int], commands: List[Tuple[int, int, int]]) -> torch.Tensor:
    """Initialize state tensor with program."""
    X = torch.zeros(cfg.d_model, cfg.n)
    
    # Scratchpad indicator
    X[-1, :cfg.s] = 1.0
    
    # Position encoding (bipolar) for cols 1+
    for col in range(1, cfg.n):
        X[cfg.idx_pos_enc:cfg.idx_pos_enc + cfg.logn, col] = to_bipolar(col, cfg.logn)
    
    # Memory (bipolar)
    for i, val in enumerate(memory):
        if i >= cfg.m:
            break
        col = cfg.s + i
        X[cfg.idx_memory:cfg.idx_memory + cfg.N, col] = signed_to_bipolar(val, cfg.N)
    
    # Commands
    for i, (a, b, c) in enumerate(commands):
        col = cfg.s + cfg.m + i
        if col >= cfg.n:
            break
        X[:cfg.logn, col] = to_bipolar(a, cfg.logn)
        X[cfg.logn:2*cfg.logn, col] = to_bipolar(b, cfg.logn)
        X[2*cfg.logn:3*cfg.logn, col] = to_bipolar(c, cfg.logn)
    
    # PC = first command location
    pc = cfg.s + cfg.m
    X[cfg.idx_pc:cfg.idx_pc + cfg.logn, 0] = to_bipolar(pc, cfg.logn)
    
    return X


def read_memory(X: torch.Tensor, cfg: SUBLEQConfig) -> List[int]:
    """Read all memory values as integers."""
    memory = []
    for i in range(cfg.m):
        col = cfg.s + i
        bipolar = X[cfg.idx_memory:cfg.idx_memory + cfg.N, col]
        memory.append(signed_from_bipolar(bipolar))
    return memory


def get_pc(X: torch.Tensor, cfg: SUBLEQConfig) -> int:
    """Get current program counter."""
    return from_bipolar(X[cfg.idx_pc:cfg.idx_pc + cfg.logn, 0])
