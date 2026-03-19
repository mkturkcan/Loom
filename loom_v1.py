"""
Extended ISA V4 - Complete Implementation with Gated FFN
=========================================================

Mathematical Foundation:
- For SUBLEQ (a >= 32): scr_sub = buf_a (mem[a]), scr_min = buf_b (mem[b])
- For extended (a < 32): scr_sub and scr_min set based on opcode

Mode Detection (for s=32, logn=6):
- SUBLEQ mode: a[5] = 1 (addresses 32-63)
- Extended mode: a[5] = 0 (opcodes 0-31)

For 6-bit address with s=32:
- bit[0] = a[5] (weight 32) — sole mode discriminator
- a >= 16 means bit[0]=1 OR bit[1]=1
- Bipolar: bit[0] = +1 OR bit[1] = +1

Gate logic:
- SUBLEQ fires when: addr_a[0] + addr_a[1] > 0 (at least one is +1)
- Extended fires when: addr_a[0] + addr_a[1] < 0 (both are -1)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from subleq import (
    SUBLEQConfig, NeuralSUBLEQ, TransformerLayer,
    to_bipolar, from_bipolar, signed_to_bipolar, signed_from_bipolar
)

# Opcodes (0-15 are extended, 16+ are SUBLEQ addresses)
OP_HALT = 0
OP_MOV = 1
OP_ADD = 2
OP_JMP = 3
OP_JZ = 4
OP_JNZ = 5
OP_INC = 6
OP_DEC = 7
OP_SHL = 8
OP_SHR = 9
OP_CMP = 10
OP_LOAD = 11
OP_AND = 12
OP_OR = 13
OP_XOR = 14
OP_SUB = 15
OP_FIND = 16
OP_SWAP = 17
OP_CMOV = 18
OP_MULACC = 19
OP_STORE = 20


@dataclass
class LoomConfig(SUBLEQConfig):
    """Extended config with larger buffer and address tags for indirect access."""

    def __post_init__(self):
        super().__post_init__()
        # Buffer: buf_a (N), buf_b (N), buf_c (N), find_temp (N), load_temp (logn)
        # FIND and LOAD use separate temp regions because both attention heads fire
        # simultaneously in L3 — shared temp causes cross-head interference.
        self.nrows_buffer = 3 * self.N + self.N + self.logn
        # Address tags: N rows storing each memory column's own index (for LOAD)
        self.nrows_tag = self.N
        self.idx_tag = (self.nrows_cmds + self.nrows_memory + self.nrows_scratchpad +
                       self.nrows_pc + self.nrows_pos_enc + self.nrows_buffer)
        self.d_model = (self.nrows_cmds + self.nrows_memory + self.nrows_scratchpad +
                       self.nrows_pc + self.nrows_pos_enc + self.nrows_buffer +
                       self.nrows_tag + self.nrows_indicator)


def opcode_pattern(opcode: int, logn: int) -> torch.Tensor:
    """Convert opcode to bipolar pattern (MSB first)."""
    pattern = torch.zeros(logn)
    for i in range(logn):
        bit = (opcode >> (logn - 1 - i)) & 1
        pattern[i] = 1.0 if bit else -1.0
    return pattern


def create_bipolar_value(val: int, N: int) -> torch.Tensor:
    """Create bipolar representation of signed integer."""
    result = torch.zeros(N)
    if val < 0:
        val = val + (1 << N)  # Two's complement
    for i in range(N):
        bit = (val >> (N - 1 - i)) & 1
        result[i] = 1.0 if bit else -1.0
    return result


class LoomComputer(NeuralSUBLEQ):
    """
    Extended ISA with properly gated Layer 2 FFN.
    
    Key insight: Gate the default buf_a copy to only fire for SUBLEQ mode.
    Extended opcodes set values directly without interference from garbage.
    """
    
    def __init__(self, cfg: SUBLEQConfig):
        if not isinstance(cfg, LoomConfig):
            self.cfg = LoomConfig(s=cfg.s, m=cfg.m, n=cfg.n, N=cfg.N, lam=cfg.lam)
        else:
            self.cfg = cfg
        self.layers = self._build_layers()
    
    def _build_layers(self) -> List[TransformerLayer]:
        layers = []
        layers.append(self._init_read_inst_v4())    # L1: Fetch instruction
        layers.append(self._init_read_mem_v4())     # L2: Read memory + decode
        layers.append(self._init_indirect_read_v4()) # L3: Indirect read + scratchpad correction
        layers.append(self._init_direct_subtract())  # L4: Direct subtract (single layer)
        layers.append(self._init_write_mem_v4())    # L5: Write result (2-head for SWAP)
        layers.extend(self._init_cond_branch_v4())  # L6-7: Branch (flag+PC+1 merged, select)
        layers.append(self._init_error_correction()) # L8: Error correction
        return layers
    
    def _init_read_inst_v4(self) -> TransformerLayer:
        """
        Layer 1: Instruction fetch via symmetric Q=K attention.

        Q=K extracts [PC | pos_enc]. Column 0 has PC set and pos_enc=0;
        instruction columns have pos_enc set and PC=0. The score between
        col 0 and the matching instruction is a 50/50 tie with self-score,
        producing buffer = 0.5 * instruction. FFN snaps to ±1 via sgn().
        """
        cfg = self.cfg
        size_inst = cfg.nrows_cmds
        size_pos_enc = cfg.logn
        
        layer = TransformerLayer(cfg, cfg.logn, 6 * size_inst, num_heads=1)
        
        # Q and K: Extract BOTH PC and position encoding (symmetric Q=K)
        # This works because:
        # - Q @ X[:,0] = PC + pos(0) = PC + all-minus-ones = PC (roughly)  
        # - Q @ X[:,j] = 0 + pos(j) = pos(j) for j > 0 (PC only in col 0)
        # - score[j, 0] = Q@X[:,j] dot Q@X[:,0] = pos(j) dot PC = high when j = PC
        layer.Q[:, cfg.idx_pc:cfg.idx_pc + size_pos_enc] = torch.eye(size_pos_enc)
        layer.Q[:, cfg.idx_pos_enc:cfg.idx_pos_enc + size_pos_enc] = torch.eye(size_pos_enc)
        layer.K = layer.Q.clone()
        
        # V: Read commands to buffer - ONLY from command columns, NOT from scratch_cmd
        layer.V[cfg.idx_buffer:cfg.idx_buffer+size_inst, :cfg.nrows_cmds] = torch.eye(size_inst)
        # REMOVED: layer.V[..., cfg.idx_scratch_cmd:...] = I to prevent contamination
        
        # FFN: Set scratch_cmd = sgn(buffer), clear buffer
        # This handles 0.5-scaled buffer values from 50/50 attention
        #
        # For each instruction bit i:
        #   Row 2i: fires when buffer[i] > 0 (positive instruction bit)
        #   Row 2i+1: fires when buffer[i] < 0 (negative instruction bit)
        #   W2 sets scratch_cmd[i] to +1 or -1 accordingly
        #
        # Also: clear old scratch_cmd and clear buffer
        
        large_const = 100.0
        
        for i in range(size_inst):
            # Row 2i: buffer[i] > 0 case
            r = 2 * i
            layer.W1[r, cfg.idx_buffer + i] = 4.0  # 4*0.5 = 2 when buffer=0.5
            layer.W1[r, -1] = large_const  # indicator column
            layer.b1[r, :] = -large_const  # only fires in scratchpad columns
            
            # Row 2i+1: buffer[i] < 0 case  
            r = 2 * i + 1
            layer.W1[r, cfg.idx_buffer + i] = -4.0  # 4*0.5 = 2 when buffer=-0.5
            layer.W1[r, -1] = large_const
            layer.b1[r, :] = -large_const
        
        # Rows for clearing scratch_cmd (only in scratchpad column)
        base = 2 * size_inst
        for i in range(size_inst):
            # Row base+2i: fires when scratch_cmd[i] = +1 AND indicator=1
            layer.W1[base + 2*i, cfg.idx_scratch_cmd + i] = 1.0
            layer.W1[base + 2*i, -1] = large_const
            layer.b1[base + 2*i, :] = -large_const
            # Row base+2i+1: fires when scratch_cmd[i] = -1 AND indicator=1
            layer.W1[base + 2*i + 1, cfg.idx_scratch_cmd + i] = -1.0
            layer.W1[base + 2*i + 1, -1] = large_const
            layer.b1[base + 2*i + 1, :] = -large_const
        
        # Rows for clearing buffer (only in scratchpad column)
        buf_base = base + 2 * size_inst
        for i in range(size_inst):
            layer.W1[buf_base + 2*i, cfg.idx_buffer + i] = 1.0
            layer.W1[buf_base + 2*i, -1] = large_const
            layer.b1[buf_base + 2*i, :] = -large_const
            layer.W1[buf_base + 2*i + 1, cfg.idx_buffer + i] = -1.0
            layer.W1[buf_base + 2*i + 1, -1] = large_const
            layer.b1[buf_base + 2*i + 1, :] = -large_const
        
        # W2: set scratch_cmd from buffer sign
        # Note: ff1 outputs ~2 (from 4*0.5), so we use weights of ±0.5 to get ±1
        for i in range(size_inst):
            # scratch_cmd[i] += 1 when buffer positive (ff1 output = 2, so 0.5*2 = 1)
            layer.W2[cfg.idx_scratch_cmd + i, 2*i] = 0.5
            # scratch_cmd[i] -= 1 when buffer negative
            layer.W2[cfg.idx_scratch_cmd + i, 2*i + 1] = -0.5
            # Clear old scratch_cmd (subtract current value)
            layer.W2[cfg.idx_scratch_cmd + i, base + 2*i] = -1.0
            layer.W2[cfg.idx_scratch_cmd + i, base + 2*i + 1] = 1.0
        
        # W2: clear buffer (subtract current value)
        for i in range(size_inst):
            layer.W2[cfg.idx_buffer + i, buf_base + 2*i] = -1.0
            layer.W2[cfg.idx_buffer + i, buf_base + 2*i + 1] = 1.0
        
        return layer
    
    def _init_read_mem_v4(self) -> TransformerLayer:
        """
        Layer 2: Read memory with gated routing.
        
        After attention (3 heads):
        - buf_a = mem[a] (garbage for extended mode)
        - buf_b = mem[b]
        - buf_c = mem[c]
        
        FFN routing:
        - SUBLEQ mode (a>=16): scr_sub = buf_a, scr_min = buf_b
        - Extended mode (a<16): scr_sub and scr_min set per opcode
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        
        buf_a = cfg.idx_buffer
        buf_b = cfg.idx_buffer + N
        buf_c = cfg.idx_buffer + 2 * N
        scr_sub = cfg.idx_scratchpad
        scr_min = cfg.idx_scratchpad + N
        addr_a = cfg.idx_scratch_cmd
        bit32 = logn - 6
        bit16 = logn - 5
        mode_bit0 = addr_a + bit32
        mode_bit1 = addr_a + bit16
        bit32 = logn - 6
        bit16 = logn - 5
        mode_bit0 = addr_a + bit32
        mode_bit1 = addr_a + bit16
        # SUBLEQ/extended mode detection uses bits for 32 and 16, not the MSBs.
        bit32 = logn - 6  # 2^5 = 32
        bit16 = logn - 5  # 2^4 = 16
        mode_bit0 = addr_a + bit32
        mode_bit1 = addr_a + bit16
        
        # Need many FFN rows for gated operations
        # Gated copy for SUBLEQ: 4*N for each of buf_a and buf_b (gated by mode)
        # Extended opcodes: vary per opcode
        num_ffn = 80 * N
        layer = TransformerLayer(cfg, logn, num_ffn, num_heads=3)
        
        # === ATTENTION HEADS ===
        
        # Head 1: Read mem[a] -> buf_a
        layer.Q1[:, cfg.idx_scratch_cmd:cfg.idx_scratch_cmd+logn] = torch.eye(logn)
        layer.Q1[:, cfg.idx_pos_enc:cfg.idx_pos_enc+logn] = torch.eye(logn)
        layer.K1 = layer.Q1.clone()
        layer.V1[buf_a:buf_a+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)
        layer.V1[buf_a:buf_a+N, scr_sub:scr_sub+N] = torch.eye(N)
        
        # Head 2: Read mem[b] -> buf_b
        layer.Q2[:, cfg.idx_scratch_cmd+logn:cfg.idx_scratch_cmd+2*logn] = torch.eye(logn)
        layer.Q2[:, cfg.idx_pos_enc:cfg.idx_pos_enc+logn] = torch.eye(logn)
        layer.K2 = layer.Q2.clone()
        layer.V2[buf_b:buf_b+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)
        layer.V2[buf_b:buf_b+N, scr_min:scr_min+N] = torch.eye(N)
        
        # Head 3: Read mem[c] -> buf_c
        layer.Q3[:, cfg.idx_scratch_cmd+2*logn:cfg.idx_scratch_cmd+3*logn] = torch.eye(logn)
        layer.Q3[:, cfg.idx_pos_enc:cfg.idx_pos_enc+logn] = torch.eye(logn)
        layer.K3 = layer.Q3.clone()
        layer.V3[buf_c:buf_c+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)
        
        # === FFN: Gated copy and opcode routing ===
        
        large = 100.0
        row = 0
        
        # ============================================================
        # SUBLEQ MODE: Copy buf_a to scr_sub (only when a >= 16)
        # ============================================================
        # Gate condition: addr_a[0] + addr_a[1] > 0
        # We use ReLU gating: output = ReLU(gate * value)
        # 
        # For each bit i, we need:
        #   scr_sub[i] = buf_a[i] WHEN (addr_a[0] OR addr_a[1])
        #
        # Using the indicator column trick with mode gating:
        #   ReLU(large*(indicator-1) + 2*(buf_a - scr_sub) + mode_weight)
        #
        # For SUBLEQ mode detection:
        #   mode_gate = ReLU(addr_a[0] + 0.5)  # Fires if bit32 is +1 (a >= 32)
        #
        # We'll use a two-stage approach:
        # Stage 1: Compute buf_a to scr_sub with SUBLEQ gating
        # Stage 2: Apply opcode-specific corrections for extended mode

        # --- SUBLEQ gated copy: buf_a -> scr_sub ---
        # Each pair of rows handles one bit, gated by SUBLEQ mode
        for i in range(N):
            # Row for positive contribution (buf_a[i] = +1)
            r = row + 4*i
            # Gate: SUBLEQ mode (addr_a[0] = +1, i.e. a >= 32) AND buf_a[i] = +1
            layer.W1[r, mode_bit0] = 1.0      # Full weight for mode (bit 32)
            layer.W1[r, buf_a + i] = 1.0      # Value contribution
            layer.W1[r, -1] = large           # Indicator
            layer.b1[r, :] = -large           # Cancel indicator for non-scratchpad
            # This row fires when: (mode >= 0) AND (buf_a[i] > 0) in scratchpad column
            # Output ~= 1 when SUBLEQ mode and buf_a[i] = +1
            layer.W2[scr_sub + i, r] = 2.0

            # Row for negative contribution
            r = row + 4*i + 1
            layer.W1[r, mode_bit0] = 1.0
            layer.W1[r, buf_a + i] = -1.0     # Negative contribution
            layer.W1[r, -1] = large
            layer.b1[r, :] = -large
            layer.W2[scr_sub + i, r] = -2.0
            
            # Rows to clear buf_a (for SUBLEQ mode)
            r = row + 4*i + 2
            layer.W1[r, buf_a + i] = 1.0
            layer.W2[buf_a + i, r] = -1.0
            
            r = row + 4*i + 3
            layer.W1[r, buf_a + i] = -1.0
            layer.W2[buf_a + i, r] = 1.0
        row += 4*N
        
        # --- Always copy buf_b -> scr_min (valid for all modes) ---
        for i in range(N):
            r = row + 4*i
            layer.W1[r, scr_min + i] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.W1[r, -1] = large
            layer.b1[r, :] = -large
            layer.W2[scr_min + i, r] = 1.0
            
            r = row + 4*i + 1
            layer.W1[r, scr_min + i] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.W1[r, -1] = large
            layer.b1[r, :] = -large
            layer.W2[scr_min + i, r] = -1.0
            
            r = row + 4*i + 2
            layer.W1[r, buf_b + i] = 1.0
            layer.W2[buf_b + i, r] = -1.0
            
            r = row + 4*i + 3
            layer.W1[r, buf_b + i] = -1.0
            layer.W2[buf_b + i, r] = 1.0
        row += 4*N

        # --- Clear buf_c at MEMORY columns only ---
        # Keep buf_c at scratchpad col 0 (= 0.5*mem[c] from attention) for L5 Head 2.
        # At memory cols (indicator=0): ReLU(±buf_c - large*0) fires → clears buf_c.
        # At scratchpad cols (indicator=1): ReLU(±buf_c - large) ≈ 0 → no effect.
        for i in range(N):
            r = row + 2*i
            layer.W1[r, buf_c + i] = 1.0
            layer.W1[r, -1] = -large
            layer.W2[buf_c + i, r] = -1.0
            r = row + 2*i + 1
            layer.W1[r, buf_c + i] = -1.0
            layer.W1[r, -1] = -large
            layer.W2[buf_c + i, r] = 1.0
        row += 2*N
        
        # ============================================================
        # EXTENDED MODE: Set values directly based on opcode
        # ============================================================
        # For extended mode (a < 16), both addr_a[0] and addr_a[1] are -1
        # Extended mode gate: ReLU(-(addr_a[0] + addr_a[1]) - 1.5) fires when both -1
        # 
        # When extended mode fires, we need to set scr_sub to specific values:
        # - INC (6): scr_sub = -1 (all +1 in bipolar)
        # - DEC (7): scr_sub = +1 (bipolar: [-1]*7 + [+1])
        # - JZ/JNZ (4,5): scr_sub = 0 (all -1)
        # - MOV (1): scr_sub = 0, scr_min = buf_c
        # - ADD (2): scr_sub = -buf_c
        # - SUB (15): scr_sub = buf_c
        
        # --- INC (opcode 6 = 000110): scr_sub = -1 (all +1s in bipolar) ---
        # Gate fires with value 0.5 when opcode matches, so scale W2 by 2
        inc_pat = opcode_pattern(OP_INC, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = inc_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = 2.0  # 0.5 * 2 = +1
        row += 1
        
        # --- DEC (opcode 7 = 000111): scr_sub = +1 ([-1]*7 + [+1]) ---
        dec_pat = opcode_pattern(OP_DEC, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = dec_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N - 1):
            layer.W2[scr_sub + i, r] = -2.0
        layer.W2[scr_sub + N - 1, r] = 2.0
        row += 1
        
        # --- JZ (opcode 4): scr_sub = 0 (all -1s) ---
        jz_pat = opcode_pattern(OP_JZ, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = jz_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # --- JNZ (opcode 5): scr_sub = 0 (all -1s) ---
        jnz_pat = opcode_pattern(OP_JNZ, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = jnz_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # --- JMP (opcode 3): scr_sub = 0 (all -1s) ---
        jmp_pat = opcode_pattern(OP_JMP, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = jmp_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # --- HALT (opcode 0): scr_sub = 0 (all -1s) ---
        halt_pat = opcode_pattern(OP_HALT, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = halt_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # --- MOV (opcode 1): scr_sub = 0 (all -1s), scr_min = buf_c ---
        mov_pat = opcode_pattern(OP_MOV, logn)
        
        # Set scr_sub to all -1s for MOV (value 0)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = mov_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # Replace scr_min with buf_c for MOV
        # scr_min currently has buf_b. We need to add (buf_c - buf_b) correction.
        for i in range(N):
            # Case: buf_c = +, buf_b = - -> add +2
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = mov_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0   # Scaled for ±0.5
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 1.0   # Need opcode match + both conditions
            layer.W2[scr_min + i, r] = 2.0  # Add +2 to flip (this is a correction, not setting)
            
            # Case: buf_c = -, buf_b = + -> add -2
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = mov_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
        row += 2*N
        
        # --- ADD (opcode 2): scr_sub = -buf_c (TWO'S COMPLEMENT) ---
        add_pat = opcode_pattern(OP_ADD, logn)
        
        # Two's complement: -x = ~x + 1
        # We need to compute ~buf_c + 1 using FFN gates
        #
        # For an N-bit value, adding 1 requires carry propagation:
        # Let f[i] = ~buf_c[i] (flipped bits, computed separately)
        # result[N-1] = f[N-1] XOR 1 = ~f[N-1] (LSB always flips)
        # result[N-2] = f[N-2] XOR carry[N-1], where carry[N-1] = f[N-1]
        # result[i] = f[i] XOR carry[i+1], where carry[i+1] = f[i+1] AND carry[i+2]
        #
        # In bipolar: XOR maps to multiplication, AND maps to min(a,b) or (a+b-1)/2
        #
        # Simpler approach using known algebraic identity:
        # For value v with bits b[i], flip(v) = -v-1 in two's complement
        # So -v = flip(v) + 1
        #
        # We can compute this by:
        # 1. Flip all bits (one's complement)
        # 2. Add 1 using carry chain
        #
        # The carry chain for +1:
        # - Start with carry_in = 1
        # - result[N-1] = f[N-1] XOR 1 = NOT f[N-1]
        # - carry[N-2] = f[N-1] AND 1 = f[N-1]
        # - result[N-2] = f[N-2] XOR carry[N-2]
        # - carry[N-3] = f[N-2] AND carry[N-2]
        # - etc.
        #
        # In bipolar:
        # - NOT x = -x
        # - x XOR y = x * y (when x, y ∈ {-1, +1})
        # - x AND y = (x * y + x + y - 1) / 2 with threshold
        #
        # Actually, for +1 addition, there's a simpler pattern:
        # The result of flip(v) + 1 flips all trailing 1s to 0s and first 0 to 1
        # In the flipped value f = ~v:
        # - trailing 0s in v become trailing 1s in f
        # - adding 1 flips these back to 0s and propagates carry
        #
        # Implementation: For each bit position, we need to know if all bits
        # to the right (in the flipped value) are +1.
        #
        # Let's use row-by-row gates:
        # Row for bit i: fires if opcode=ADD AND flipped_bit[i]=-1 AND all bits j>i are +1
        
        # First pass: set scr_sub to flipped buf_c (one's complement)
        for i in range(N):
            # Flip bit: if buf_c = +1, set -1; if buf_c = -1, set +1
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = add_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, 0] = -logn
            layer.W2[scr_sub + i, r] = -1.0
            
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = add_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.b1[r, 0] = -logn
            layer.W2[scr_sub + i, r] = 1.0
        row += 2*N
        
        # Second pass: add 1 using carry propagation.
        # Carry-in for bit i occurs when all lower bits of buf_c are -1 (i.e., 0 in binary),
        # because f = ~buf_c has trailing +1s in those positions.
        # For each bit we must flip based on its current value:
        #   buf_c[i] = +1 (f_i = -1) -> add +2 (flip to +1)
        #   buf_c[i] = -1 (f_i = +1) -> add -2 (flip to -1)
        for i in range(N - 1, -1, -1):
            lower_bits = N - 1 - i
            # Case 1: buf_c[i] = +1 -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = add_pat[j].item()
            for j in range(i + 1, N):
                layer.W1[r, buf_c + j] = -2.0  # Require lower bits = -1
            layer.W1[r, buf_c + i] = 2.0       # Require buf_c[i] = +1
            layer.b1[r, :] = -(logn + lower_bits + 1) + 1.0
            layer.W2[scr_sub + i, r] = 2.0
            row += 1
            
            # Case 2: buf_c[i] = -1 -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = add_pat[j].item()
            for j in range(i + 1, N):
                layer.W1[r, buf_c + j] = -2.0  # Require lower bits = -1
            layer.W1[r, buf_c + i] = -2.0      # Require buf_c[i] = -1
            layer.b1[r, :] = -(logn + lower_bits + 1) + 1.0
            layer.W2[scr_sub + i, r] = -2.0
            row += 1
        
        # --- SUB (opcode 15): scr_sub = buf_c ---
        sub_pat = opcode_pattern(OP_SUB, logn)
        
        # Set scr_sub = buf_c, from 0 baseline
        for i in range(N):
            # When buf_c[i] = +1 → result should be +1 → add +1
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = sub_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, 0] = -logn
            layer.W2[scr_sub + i, r] = 1.0
            
            # When buf_c[i] = -1 → result should be -1 → add -1
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = sub_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.b1[r, 0] = -logn
            layer.W2[scr_sub + i, r] = -1.0
        row += 2*N

        # --- SHL (opcode 8): Shift Left by 1 ---
        # mem[b] <<= 1. Bit ordering: index 0 = MSB, index N-1 = LSB.
        # SHL: result[i] = input[i+1] for i=0..N-2, result[N-1] = -1 (zero bit)
        # We set scr_sub = 0, scr_min = shifted(buf_b).
        # scr_min already has buf_b. We correct it to shifted version.
        # Correction: scr_min[i] needs to become buf_b[i+1] instead of buf_b[i].
        # delta[i] = buf_b[i+1] - buf_b[i] for i < N-1
        # delta[N-1] = -1 - buf_b[N-1]
        shl_pat = opcode_pattern(OP_SHL, logn)

        # Set scr_sub = 0 (all -1s) for pass-through
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = shl_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Correction: for i < N-1, add (buf_b[i+1] - buf_b[i]) to scr_min[i]
        for i in range(N - 1):
            # Case: buf_b[i+1]=+1, buf_b[i]=-1 -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = shl_pat[j].item()
            layer.W1[r, buf_b + i + 1] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0
            row += 1

            # Case: buf_b[i+1]=-1, buf_b[i]=+1 -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = shl_pat[j].item()
            layer.W1[r, buf_b + i + 1] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
            row += 1

        # For LSB (i=N-1): set to -1 (zero). Correction = -1 - buf_b[N-1].
        # If buf_b[N-1] = +1: subtract 2. If buf_b[N-1] = -1: no change.
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = shl_pat[j].item()
        layer.W1[r, buf_b + N - 1] = 2.0
        layer.b1[r, 0] = -logn
        layer.W2[scr_min + N - 1, r] = -2.0
        row += 1

        # --- SHR (opcode 9): Arithmetic Shift Right by 1 ---
        # mem[b] >>= 1 (arithmetic, preserves sign).
        # SHR: result[i] = input[i-1] for i=1..N-1, result[0] = input[0] (sign preserved)
        # scr_min has buf_b. Correct: scr_min[i] should become buf_b[i-1] for i>0.
        # Correction: delta[i] = buf_b[i-1] - buf_b[i] for i > 0
        # delta[0] = 0 (sign bit stays)
        shr_pat = opcode_pattern(OP_SHR, logn)

        # Set scr_sub = 0 (all -1s) for pass-through
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = shr_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Correction: for i > 0, add (buf_b[i-1] - buf_b[i]) to scr_min[i]
        for i in range(1, N):
            # Case: buf_b[i-1]=+1, buf_b[i]=-1 -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = shr_pat[j].item()
            layer.W1[r, buf_b + i - 1] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0
            row += 1

            # Case: buf_b[i-1]=-1, buf_b[i]=+1 -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = shr_pat[j].item()
            layer.W1[r, buf_b + i - 1] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
            row += 1

        # --- CMP (opcode 10 = BLTZ): scr_sub = 0 (all -1s) ---
        # CMP(b, c): if mem[b] < 0, PC = c; else PC = PC+1.  mem[b] unchanged.
        # scr_sub = 0 means subtract phase is no-op (result = scr_min = buf_b = original mem[b]).
        # Write layer writes buf_b back to mem[b] — identity, no modification.
        cmp_pat = opcode_pattern(OP_CMP, logn)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = cmp_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # --- LOAD (opcode 11): mem[b] = mem[mem[c]] ---
        # Instruction: (OP_LOAD, target_addr, pointer_addr)
        # Head 2 reads mem[b] -> buf_b (target, overwritten)
        # Head 3 reads mem[c] -> buf_c (pointer value)
        # Indirect read layer uses pointer to read mem[pointer] -> scr_min
        #
        # IMPORTANT: All LOAD FFN rows use indicator gating to restrict
        # firing to scratchpad columns only (indicator=1). Without this,
        # memory columns' cmd rows can accidentally match the opcode pattern
        # (from L1/L2 attention spillover), causing temp to be set in memory
        # columns and corrupting L3's indirect read attention.
        load_pat = opcode_pattern(OP_LOAD, logn)
        indicator = cfg.d_model - 1  # indicator row: 1 for scratchpad, 0 for memory

        # Set scr_sub = 0 (all -1s) for LOAD
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = load_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Zero scr_min for LOAD (undo "always copy" which sets scr_min = 2*buf_b).
        # buf_b = ±0.5 (from 50/50 attention), always-copy gives scr_min = ±1.0.
        # Need ∓1.0 correction to zero it.
        # ReLU output is 0.5 when gated row fires, so W2 = ±2.0 gives ±1.0.
        find_temp = cfg.idx_buffer + 3 * N      # FIND temp: N rows
        load_temp = cfg.idx_buffer + 3 * N + N  # LOAD temp: logn rows
        for i in range(N):
            # Row A: LOAD AND buf_b[i] = +1 → subtract 1 from scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = load_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 0.5
            layer.W2[scr_min + i, r] = -2.0
            row += 1

            # Row B: LOAD AND buf_b[i] = -1 → add 1 to scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = load_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 0.5
            layer.W2[scr_min + i, r] = 2.0
            row += 1

        # Convert buf_c (N-bit pointer value) → pointer_pos_enc (logn bits) in load_temp.
        # L3 uses Q=K symmetric attention reading load_temp + pos_enc to select
        # column s+pointer_value. The conversion maps:
        #   s = 2^k, column s+p has pos_enc bits:
        #     bit_position > k: constant 0 → bipolar -1
        #     bit_position == k: constant 1 → bipolar +1
        #     bit_position < k: bit of p → copy from buf_c[N-1-bit_position]
        #
        # Indicator gating ensures only scratchpad columns fire.
        k = int(np.log2(cfg.s))  # s = 2^k

        # Row 1: Set constant bits in load_temp (all constants in one row).
        # For pos_enc index i: bit_position = logn-1-i
        #   bit_position > k → load_temp[i] = -1 (W2 = -2.0)
        #   bit_position == k → load_temp[i] = +1 (W2 = +2.0)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = load_pat[j].item()
        layer.W1[r, indicator] = float(logn)
        layer.b1[r, :] = -2.0 * logn + 0.5
        # ReLU output when LOAD+indicator: logn + logn - 2*logn + 0.5 = 0.5
        for i in range(logn):
            bit_position = logn - 1 - i
            if bit_position > k:
                layer.W2[load_temp + i, r] = -2.0  # 0.5 * -2.0 = -1.0
            elif bit_position == k:
                layer.W2[load_temp + i, r] = 2.0   # 0.5 * 2.0 = +1.0
        row += 1

        # Copy bits: for bit_position < k, copy buf_c[N-1-bit_position] → load_temp[i]
        for i in range(logn):
            bit_position = logn - 1 - i
            if bit_position >= k:
                continue  # handled by constant row above
            src_idx = N - 1 - bit_position  # buf_c index for this bit of p

            # Row A: LOAD AND indicator AND buf_c[src] > 0 → load_temp[i] = +1
            # buf_c = ±0.5 from 50/50 attention; ReLU output = 0.5 when fires
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = load_pat[j].item()
            layer.W1[r, buf_c + src_idx] = 2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[load_temp + i, r] = 2.0  # 0.5 * 2.0 = +1.0
            row += 1

            # Row B: LOAD AND indicator AND buf_c[src] < 0 → load_temp[i] = -1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = load_pat[j].item()
            layer.W1[r, buf_c + src_idx] = -2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[load_temp + i, r] = -2.0  # 0.5 * -2.0 = -1.0
            row += 1

        # --- FIND (opcode 16): Content-addressable memory search ---
        # FIND(dest, value_addr): mem[dest] = index i where mem[i] == mem[value_addr]
        # Head 2 reads mem[b] -> buf_b (destination, overwritten)
        # Head 3 reads mem[c] -> buf_c (search value)
        # L3 head 2 does content-addressable search using temp as query.
        find_pat = opcode_pattern(OP_FIND, logn)

        # Set scr_sub = 0 (all -1s) for FIND
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = find_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Zero scr_min for FIND (undo "always copy" buf_b → scr_min)
        for i in range(N):
            # Row A: FIND AND buf_b[i] = +1 → subtract 1 from scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = find_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 0.5
            layer.W2[scr_min + i, r] = -2.0
            row += 1

            # Row B: FIND AND buf_b[i] = -1 → add 1 to scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = find_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 0.5
            layer.W2[scr_min + i, r] = 2.0
            row += 1

        # Copy buf_c (search value, N bits) → find_temp[0:N] with indicator gating.
        # buf_c = ±0.5 from attention; we need find_temp = ±1.
        for i in range(N):
            # Row A: FIND AND indicator AND buf_c[i] > 0 → find_temp[i] = +1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = find_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[find_temp + i, r] = 2.0  # 0.5 * 2.0 = +1.0
            row += 1

            # Row B: FIND AND indicator AND buf_c[i] < 0 → find_temp[i] = -1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = find_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[find_temp + i, r] = -2.0  # 0.5 * -2.0 = -1.0
            row += 1

        # --- CMOV (opcode 18): Conditional Move ---
        # CMOV(b, c): if mem[b] < 0, mem[b] = mem[c]; else mem[b] unchanged.
        # The "always copy" already sets scr_min = buf_b. For CMOV when sign is
        # negative (buf_b[0] = +0.5), we replace scr_min with buf_c via correction.
        cmov_pat = opcode_pattern(OP_CMOV, logn)

        # Set scr_sub = 0 (all -1s) for CMOV
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = cmov_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Conditional correction: replace scr_min bits when buf_b[0] = +1 (negative)
        # Special handling for i=0: buf_b+0 == buf_b (sign bit position), so the sign
        # gate weight and dest bit weight alias the same tensor position. We use a
        # combined weight of 4.0 (sign+dest both want positive) for Case B i=0.
        # Case A for i=0 is impossible (would need sign=+ AND dest=-, contradiction).

        # i=0 (sign bit): only Case B needed (buf_c[0]=-, buf_b[0]=+)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = cmov_pat[j].item()
        layer.W1[r, buf_b] = 4.0       # combined: 2.0(sign) + 2.0(dest i=0)
        layer.W1[r, buf_c] = -2.0      # source bit 0 (want negative)
        layer.b1[r, 0] = -logn - 2.5
        layer.W2[scr_min, r] = -4.0    # 0.5 * -4.0 = -2.0
        row += 1

        # i=0 Case A: buf_c[0]=+ AND buf_b[0]=+ → no correction (same value)
        # This is a no-op — skip. (Both signs match, scr_min[0] already correct.)
        # But we need a dummy row to keep the 2-rows-per-bit structure? No, just skip.

        # i>0: normal 3-condition gating (sign + buf_c + buf_b at separate positions)
        for i in range(1, N):
            # Case A: sign=neg AND buf_c[i]=+ AND buf_b[i]=- → add +2 to scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = cmov_pat[j].item()
            layer.W1[r, buf_b] = 2.0       # sign bit (buf_b[0])
            layer.W1[r, buf_c + i] = 2.0   # source bit
            layer.W1[r, buf_b + i] = -2.0  # dest bit (want negative)
            layer.b1[r, 0] = -logn - 2.5
            layer.W2[scr_min + i, r] = 4.0  # 0.5 * 4.0 = +2.0
            row += 1

            # Case B: sign=neg AND buf_c[i]=- AND buf_b[i]=+ → add -2 to scr_min
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = cmov_pat[j].item()
            layer.W1[r, buf_b] = 2.0       # sign bit (buf_b[0])
            layer.W1[r, buf_c + i] = -2.0  # source bit (want negative)
            layer.W1[r, buf_b + i] = 2.0   # dest bit
            layer.b1[r, 0] = -logn - 2.5
            layer.W2[scr_min + i, r] = -4.0  # 0.5 * -4.0 = -2.0
            row += 1

        # --- SWAP (opcode 17): Swap mem[b] and mem[c] ---
        # SWAP(b, c): temp = mem[b]; mem[b] = mem[c]; mem[c] = temp
        #
        # L5 Head 1 writes scr_min → mem[b]  (scr_min = buf_c = mem[c], via MOV correction)
        # L5 Head 2 writes buf_a → mem[c]    (buf_a = buf_b = original mem[b], via copy below)
        #
        # For non-SWAP opcodes, buf_a = 0 after L2's SUBLEQ clear rows,
        # so Head 2's write is naturally a no-op (2*0.5*mem - mem = 0).
        swap_pat = opcode_pattern(OP_SWAP, logn)

        # Set scr_sub = 0 (all -1s)
        # Use b1[:] to gate off ALL non-col-0 columns (addr_a=0 at memory cols).
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = swap_pat[j].item()
        layer.b1[r, :] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # MOV correction: scr_min = buf_c (same pattern as MOV opcode)
        # b1[:] ensures this only fires at col 0 (where opcode pattern lives).
        for i in range(N):
            # Case: buf_c = +, buf_b = - → add +2
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = swap_pat[j].item()
            layer.W1[r, buf_c + i] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, :] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0

            # Case: buf_c = -, buf_b = + → add -2
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = swap_pat[j].item()
            layer.W1[r, buf_c + i] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, :] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
        row += 2*N

        # MOV correction: buf_c ← buf_b (SWAP-gated).
        # After L2 attention, buf_c[col0] = 0.5*mem[c]. For SWAP, replace with
        # 0.5*mem[b] so L5 Head 2 writes mem[b] to addr_c.
        # b1[:] prevents spurious firing at memory columns.
        for i in range(N):
            # Case: buf_b > buf_c → add +1 to buf_c
            r = row + 2*i
            for j in range(logn):
                layer.W1[r, addr_a + j] = swap_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0
            layer.W1[r, buf_c + i] = -2.0
            layer.b1[r, :] = -logn - 1.0
            layer.W2[buf_c + i, r] = 1.0

            # Case: buf_b < buf_c → add -1 to buf_c
            r = row + 2*i + 1
            for j in range(logn):
                layer.W1[r, addr_a + j] = swap_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, :] = -logn - 1.0
            layer.W2[buf_c + i, r] = -1.0
        row += 2*N

        # --- MULACC (opcode 19): Multiply-Accumulate Step ---
        # MULACC(b, c): mem[b] = (mem[b] << 1) + (mem[c] if MSB(mem[b]) else 0)
        # Used for shift-and-add multiplication.
        #
        # Implementation:
        #   scr_min = shifted(buf_b)          (same SHL correction as opcode 8)
        #   scr_sub = -buf_c when buf_b[0]=+1 (MSB set), else 0
        #   L4 computes: scr_min - scr_sub
        #     MSB set:   shifted(buf_b) - (-buf_c) = shifted(buf_b) + buf_c
        #     MSB clear: shifted(buf_b) - 0        = shifted(buf_b)
        mulacc_pat = opcode_pattern(OP_MULACC, logn)

        # Part 1: Clear scr_sub to 0 (all -1s)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = mulacc_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Part 2: SHL correction for scr_min (same pattern as SHL opcode)
        # For i < N-1: delta = buf_b[i+1] - buf_b[i]
        for i in range(N - 1):
            # Case: buf_b[i+1]=+1, buf_b[i]=-1 -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b + i + 1] = 2.0
            layer.W1[r, buf_b + i] = -2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0
            row += 1

            # Case: buf_b[i+1]=-1, buf_b[i]=+1 -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b + i + 1] = -2.0
            layer.W1[r, buf_b + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
            row += 1

        # LSB (i=N-1): set to -1 (zero). If buf_b[N-1]=+1, subtract 2.
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = mulacc_pat[j].item()
        layer.W1[r, buf_b + N - 1] = 2.0
        layer.b1[r, 0] = -logn
        layer.W2[scr_min + N - 1, r] = -2.0
        row += 1

        # Part 3: Conditional two's complement negation of buf_c into scr_sub
        # Only fires when buf_b[0] = +1 (MSB of original value was set).
        # Same structure as ADD's negation but with extra buf_b[0] gate.

        # Pass 1: One's complement (flip buf_c bits) into scr_sub
        # scr_sub was cleared to -1 (0). ~buf_c[i]:
        #   buf_c[i]=+1 -> ~=-1, already -1, no change
        #   buf_c[i]=-1 -> ~=+1, need +2 correction
        for i in range(N):
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b] = 2.0        # gate on MSB
            layer.W1[r, buf_c + i] = -2.0   # fire when buf_c[i] = -1
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_sub + i, r] = 2.0  # 0.5 * 2.0 = +1.0
            row += 1

        # Pass 2: Add 1 carry chain (same logic as ADD's carry chain)
        # For each bit i: flip if all lower bits of buf_c are -1 (trailing zeros
        # in original, trailing ones in ~buf_c, carry propagates through them).
        for i in range(N - 1, -1, -1):
            lower_bits = N - 1 - i
            # Case 1: buf_c[i] = +1 (f[i]=-1, carry flips to +1) -> add +2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b] = 2.0        # gate on MSB
            for j in range(i + 1, N):
                layer.W1[r, buf_c + j] = -2.0  # require lower bits = -1
            layer.W1[r, buf_c + i] = 2.0       # buf_c[i] = +1
            layer.b1[r, :] = -(logn + lower_bits + 2) + 1.0
            layer.W2[scr_sub + i, r] = 2.0
            row += 1

            # Case 2: buf_c[i] = -1 (f[i]=+1, carry flips to -1) -> add -2
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = mulacc_pat[j].item()
            layer.W1[r, buf_b] = 2.0        # gate on MSB
            for j in range(i + 1, N):
                layer.W1[r, buf_c + j] = -2.0  # require lower bits = -1
            layer.W1[r, buf_c + i] = -2.0      # buf_c[i] = -1
            layer.b1[r, :] = -(logn + lower_bits + 2) + 1.0
            layer.W2[scr_sub + i, r] = -2.0
            row += 1

        # --- AND (opcode 12): Bitwise AND ---
        # AND in bipolar: result = +1 if BOTH inputs are +1, else -1
        # Formula: result = sign(buf_b + buf_c - 1)
        # We compute: result = -1 (default) + correction when both positive
        #
        # Default copy sets scr_min = buf_b. We need to compute:
        # result[i] = +1 if buf_b[i]=+1 AND buf_c[i]=+1, else -1
        #
        # Correction from buf_b:
        # - If buf_b=+1, buf_c=+1: keep +1 (no change)
        # - If buf_b=+1, buf_c=-1: change to -1 (subtract 2)
        # - If buf_b=-1, buf_c=+1: keep -1 (no change)
        # - If buf_b=-1, buf_c=-1: keep -1 (no change)
        #
        # So we only need to subtract 2 when buf_b=+1 and buf_c=-1
        and_pat = opcode_pattern(OP_AND, logn)
        
        # First set scr_sub = 0 for pass-through
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = and_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # Correction: subtract 2 when buf_b=+1 and buf_c=-1
        for i in range(N):
            r = row + i
            for j in range(logn):
                layer.W1[r, addr_a + j] = and_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0   # +1 when buf_b = +0.5
            layer.W1[r, buf_c + i] = -2.0  # +1 when buf_c = -0.5
            layer.b1[r, 0] = -logn - 1.0   # Need opcode + buf_b>0 + buf_c<0
            layer.W2[scr_min + i, r] = -2.0  # Flip +1 to -1
        row += N
        
        # --- OR (opcode 13): Bitwise OR ---
        # OR in bipolar: result = +1 if AT LEAST ONE input is +1, else -1
        # Formula: result = sign(buf_b + buf_c + 1)
        #
        # Default copy sets scr_min = buf_b. We need corrections:
        # - If buf_b=+1, buf_c=+1: keep +1 (no change)
        # - If buf_b=+1, buf_c=-1: keep +1 (no change)
        # - If buf_b=-1, buf_c=+1: change to +1 (add 2)
        # - If buf_b=-1, buf_c=-1: keep -1 (no change)
        #
        # So we only need to add 2 when buf_b=-1 and buf_c=+1
        or_pat = opcode_pattern(OP_OR, logn)
        
        # Set scr_sub = 0 for pass-through
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = or_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # Correction: add 2 when buf_b=-1 and buf_c=+1
        for i in range(N):
            r = row + i
            for j in range(logn):
                layer.W1[r, addr_a + j] = or_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0  # +1 when buf_b = -0.5
            layer.W1[r, buf_c + i] = 2.0   # +1 when buf_c = +0.5
            layer.b1[r, 0] = -logn - 1.0   # Need opcode + buf_b<0 + buf_c>0
            layer.W2[scr_min + i, r] = 2.0  # Flip -1 to +1
        row += N
        
        # --- XOR (opcode 14): Bitwise XOR ---
        # XOR in bipolar: result = -buf_b * buf_c
        # result = +1 when signs differ, -1 when signs are same
        #
        # Default copy sets scr_min = buf_b. We need corrections:
        # - If buf_b=+1, buf_c=+1: change to -1 (subtract 2)
        # - If buf_b=+1, buf_c=-1: keep +1 (no change, XOR=1)
        # - If buf_b=-1, buf_c=+1: keep -1 (wait, XOR should be +1!)
        # - If buf_b=-1, buf_c=-1: keep -1 (XOR=0, correct)
        #
        # Correction map:
        # - buf_b=+1, buf_c=+1: subtract 2 (result: -1)
        # - buf_b=-1, buf_c=+1: add 2 (result: +1)
        xor_pat = opcode_pattern(OP_XOR, logn)
        
        # Set scr_sub = 0 for pass-through
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = xor_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1
        
        # Case 1: buf_b=+1, buf_c=+1 -> subtract 2 (XOR gives -1)
        for i in range(N):
            r = row + i
            for j in range(logn):
                layer.W1[r, addr_a + j] = xor_pat[j].item()
            layer.W1[r, buf_b + i] = 2.0
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = -2.0
        row += N
        
        # Case 2: buf_b=-1, buf_c=+1 -> add 2 (XOR gives +1)
        for i in range(N):
            r = row + i
            for j in range(logn):
                layer.W1[r, addr_a + j] = xor_pat[j].item()
            layer.W1[r, buf_b + i] = -2.0
            layer.W1[r, buf_c + i] = 2.0
            layer.b1[r, 0] = -logn - 1.0
            layer.W2[scr_min + i, r] = 2.0
        row += N
        
        # Note: b2 bias NOT used - it would affect SUBLEQ mode too
        # Extended opcodes set values from 0 baseline using ±1 contributions

        # --- STORE (opcode 20): mem[mem[c]] = mem[b] ---
        # Instruction: (OP_STORE, src_addr, ptr_addr)
        # Head 2 reads mem[b] -> buf_b (source value)
        # Head 3 reads mem[c] -> buf_c (pointer value)
        #
        # The always-copy sets scr_min = buf_b = mem[src]. This is the value
        # to write, so we do NOT undo it (unlike LOAD which zeros scr_min).
        # scr_sub = 0 makes L4 a pass-through: result = scr_min = mem[src].
        #
        # The key trick: overwrite scratch_cmd addr_b with the position encoding
        # of column (s + mem[ptr]). This makes L5 Head 1 write to the
        # dereferenced address instead of the literal b field.
        store_pat = opcode_pattern(OP_STORE, logn)
        addr_b_start = cfg.idx_scratch_cmd + logn  # addr_b region in scratch_cmd

        # Set scr_sub = 0 (all -1s)
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = store_pat[j].item()
        layer.b1[r, 0] = -logn + 0.5
        for i in range(N):
            layer.W2[scr_sub + i, r] = -2.0
        row += 1

        # Clear old addr_b bits from scratch_cmd (at scratchpad cols only).
        # addr_b bits are FULL scale ±1 (from L1 instruction decode), not ±0.5.
        # With W1 weight 2.0, contribution = 2.0*1.0 = 2.0 (not 1.0 like buf_c).
        # Bias must be -(2*logn + 3.5) to get ReLU output = 0.5 when match.
        for i in range(logn):
            # Row A: STORE AND indicator AND addr_b[i] = +1 → subtract
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = store_pat[j].item()
            layer.W1[r, addr_b_start + i] = 2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 3.5
            layer.W2[addr_b_start + i, r] = -2.0  # 0.5 * -2.0 = -1.0
            row += 1

            # Row B: STORE AND indicator AND addr_b[i] = -1 → add
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = store_pat[j].item()
            layer.W1[r, addr_b_start + i] = -2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 3.5
            layer.W2[addr_b_start + i, r] = 2.0  # 0.5 * 2.0 = +1.0
            row += 1

        # Write new addr_b = position encoding of column (s + pointer_value).
        # Same bit layout as LOAD's load_temp conversion, but targeting addr_b.
        k = int(np.log2(cfg.s))  # s = 2^k

        # Constants row: set bits for positions > k to -1, position k to +1
        r = row
        for j in range(logn):
            layer.W1[r, addr_a + j] = store_pat[j].item()
        layer.W1[r, indicator] = float(logn)
        layer.b1[r, :] = -2.0 * logn + 0.5
        for i in range(logn):
            bit_position = logn - 1 - i
            if bit_position > k:
                layer.W2[addr_b_start + i, r] = -2.0
            elif bit_position == k:
                layer.W2[addr_b_start + i, r] = 2.0
        row += 1

        # Copy variable bits from buf_c to addr_b
        for i in range(logn):
            bit_position = logn - 1 - i
            if bit_position >= k:
                continue
            src_idx = N - 1 - bit_position

            # Row A: STORE AND indicator AND buf_c[src] > 0 → addr_b[i] = +1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = store_pat[j].item()
            layer.W1[r, buf_c + src_idx] = 2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[addr_b_start + i, r] = 2.0
            row += 1

            # Row B: STORE AND indicator AND buf_c[src] < 0 → addr_b[i] = -1
            r = row
            for j in range(logn):
                layer.W1[r, addr_a + j] = store_pat[j].item()
            layer.W1[r, buf_c + src_idx] = -2.0
            layer.W1[r, indicator] = float(logn) + 2.0
            layer.b1[r, :] = -2.0 * logn - 2.5
            layer.W2[addr_b_start + i, r] = -2.0
            row += 1

        return layer
    
    def _init_write_mem_v4(self) -> TransformerLayer:
        """Write Memory layer with 2 heads (V4 extension for SWAP).

        Head 1 (all opcodes): matches addr_b, writes scr_min → mem[b].
            Same as base _init_write_mem.

        Head 2 (SWAP only active): matches addr_c, writes via find_temp.
            V2 maps 2*buf_c + memory → find_temp.
            buf_c at col 0 = 0.5*mem[c] (non-SWAP) or 0.5*mem[b] (SWAP).
            buf_c at memory cols = 0 (cleared by L2 memory-only clear).
            50/50 at matched addr_c col:
              non-SWAP: find_temp = mem[c] → FFN: 0 (natural no-op)
              SWAP:     find_temp = 0.5*mem[b]+0.5*mem[c] → FFN: mem[b]-mem[c]
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        buf_a = cfg.idx_buffer
        find_temp = cfg.idx_buffer + 3 * N

        # 4N for Head 1 write + 4N for Head 2 write + 2N for buf_c clear = 10N
        num_ffn = 10 * N
        layer = TransformerLayer(cfg, logn, num_ffn, num_heads=2)
        large = 100.0

        # === Head 1: Match addr_b (same as base _init_write_mem) ===
        layer.Q1[:, cfg.idx_scratch_cmd + logn:cfg.idx_scratch_cmd + 2*logn] = torch.eye(logn)
        layer.Q1[:, cfg.idx_pos_enc:cfg.idx_pos_enc + logn] = torch.eye(logn)
        layer.K1 = layer.Q1.clone()

        # V1: memory + scr_min → buf_a
        layer.V1[buf_a:buf_a+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)
        layer.V1[buf_a:buf_a+N, cfg.idx_scratchpad+N:cfg.idx_scratchpad+2*N] = torch.eye(N)

        # === Head 2: Match addr_c (SWAP second write) ===
        layer.Q2[:, cfg.idx_scratch_cmd + 2*logn:cfg.idx_scratch_cmd + 3*logn] = torch.eye(logn)
        layer.Q2[:, cfg.idx_pos_enc:cfg.idx_pos_enc + logn] = torch.eye(logn)
        layer.K2 = layer.Q2.clone()

        # V2: 2*buf_c + memory → find_temp
        # buf_c at col 0 = 0.5*mem[c] (non-SWAP) or 0.5*mem[b] (SWAP).
        # buf_c at memory cols = 0 (cleared by L2 memory-only clear).
        # At matched addr_c: 50/50 gives find_temp = 0.5*(2*buf_c[0]+0) + 0.5*(0+mem[c])
        #   non-SWAP: = 0.5*mem[c] + 0.5*mem[c] = mem[c] → FFN: 0 (no-op)
        #   SWAP:     = 0.5*mem[b] + 0.5*mem[c] → FFN: mem[b]-mem[c] (writes mem[b])
        buf_c = cfg.idx_buffer + 2 * N
        layer.V2[find_temp:find_temp+N, buf_c:buf_c+N] = 2.0 * torch.eye(N)
        layer.V2[find_temp:find_temp+N, cfg.idx_memory:cfg.idx_memory+N] = torch.eye(N)

        # === FFN Section 1: Head 1 write (buf_a → memory at addr_b) ===
        # Same logic as base _init_write_mem
        layer.W1[:N, cfg.idx_memory:cfg.idx_memory+N] = -2 * torch.eye(N)
        layer.W1[:N, buf_a:buf_a+N] = 2 * torch.eye(N)
        layer.W1[:N, -1] = -large
        layer.W1[N:2*N, cfg.idx_memory:cfg.idx_memory+N] = 2 * torch.eye(N)
        layer.W1[N:2*N, buf_a:buf_a+N] = -2 * torch.eye(N)
        layer.W1[N:2*N, -1] = -large
        # buf_a clear
        layer.W1[2*N:3*N, buf_a:buf_a+N] = torch.eye(N)
        layer.W1[3*N:4*N, buf_a:buf_a+N] = -torch.eye(N)

        layer.W2[cfg.idx_memory:cfg.idx_memory+N, :N] = torch.eye(N)
        layer.W2[cfg.idx_memory:cfg.idx_memory+N, N:2*N] = -torch.eye(N)
        layer.W2[buf_a:buf_a+N, 2*N:3*N] = -torch.eye(N)
        layer.W2[buf_a:buf_a+N, 3*N:4*N] = torch.eye(N)

        # === FFN Section 2: Head 2 write (find_temp → memory at addr_c) ===
        off = 4 * N
        layer.W1[off:off+N, cfg.idx_memory:cfg.idx_memory+N] = -2 * torch.eye(N)
        layer.W1[off:off+N, find_temp:find_temp+N] = 2 * torch.eye(N)
        layer.W1[off:off+N, -1] = -large
        layer.W1[off+N:off+2*N, cfg.idx_memory:cfg.idx_memory+N] = 2 * torch.eye(N)
        layer.W1[off+N:off+2*N, find_temp:find_temp+N] = -2 * torch.eye(N)
        layer.W1[off+N:off+2*N, -1] = -large
        # find_temp clear
        layer.W1[off+2*N:off+3*N, find_temp:find_temp+N] = torch.eye(N)
        layer.W1[off+3*N:off+4*N, find_temp:find_temp+N] = -torch.eye(N)

        layer.W2[cfg.idx_memory:cfg.idx_memory+N, off:off+N] = torch.eye(N)
        layer.W2[cfg.idx_memory:cfg.idx_memory+N, off+N:off+2*N] = -torch.eye(N)
        layer.W2[find_temp:find_temp+N, off+2*N:off+3*N] = -torch.eye(N)
        layer.W2[find_temp:find_temp+N, off+3*N:off+4*N] = torch.eye(N)

        # === FFN Section 3: Clear buf_c (prevent accumulation across cycles) ===
        off2 = 8 * N
        layer.W1[off2:off2+N, buf_c:buf_c+N] = torch.eye(N)
        layer.W1[off2+N:off2+2*N, buf_c:buf_c+N] = -torch.eye(N)

        layer.W2[buf_c:buf_c+N, off2:off2+N] = -torch.eye(N)
        layer.W2[buf_c:buf_c+N, off2+N:off2+2*N] = torch.eye(N)

        return layer

    def _init_indirect_read_v4(self) -> TransformerLayer:
        """
        Layer for indirect memory read (LOAD), content-addressable search (FIND),
        and scratchpad correction.

        Head 1 (LOAD): Q=K symmetric on load_temp[0:logn] + pos_enc[0:logn]
        Head 2 (FIND): Q=K symmetric on find_temp[0:N] + idx_memory[0:N]

        FFN: Clears temp buffers + snaps scr_sub/scr_min back to bipolar.
        The correction runs after attention, so it fixes errors from both
        L2's memory read and L3's own LOAD/FIND attention.
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        find_temp = cfg.idx_buffer + 3 * N      # FIND temp: N rows
        load_temp = cfg.idx_buffer + 3 * N + N  # LOAD temp: logn rows
        scr_sub = cfg.idx_scratchpad
        scr_min = cfg.idx_scratchpad + N

        large = 100.0
        num_rows_Q = max(logn, N)
        num_clear = N + logn    # clear find_temp (N) + load_temp (logn)
        num_correct = 12 * N    # 6 rows per bit × 2 targets (scr_sub, scr_min)
        layer = TransformerLayer(cfg, num_rows_Q, 2 * num_clear + num_correct, num_heads=2)

        # Head 1 (LOAD): Q=K on load_temp[0:logn] + pos_enc[0:logn]
        for i in range(logn):
            layer.Q1[i, load_temp + i] = 1.0
            layer.Q1[i, cfg.idx_pos_enc + i] = 1.0
        layer.K1 = layer.Q1.clone()

        # V1: copy memory → scr_min with weight 2.0
        for i in range(N):
            layer.V1[scr_min + i, cfg.idx_memory + i] = 2.0

        # Head 2 (FIND): Q=K on find_temp[0:N] + idx_memory[0:N]
        for i in range(N):
            layer.Q2[i, find_temp + i] = 1.0
            layer.Q2[i, cfg.idx_memory + i] = 1.0
        layer.K2 = layer.Q2.clone()

        # V2: copy tag → scr_min with weight 2.0
        for i in range(N):
            layer.V2[scr_min + i, cfg.idx_tag + i] = 2.0

        # FFN Section 1: Clear find_temp (N rows) and load_temp (logn rows).
        # Indicator-gated to scratchpad columns only.
        r = 0
        for temp_start, temp_size in [(find_temp, N), (load_temp, logn)]:
            for i in range(temp_size):
                layer.W1[r, temp_start + i] = 1.0
                layer.W1[r, -1] = large
                layer.b1[r, :] = -large
                layer.W2[temp_start + i, r] = -1.0
                r += 1

                layer.W1[r, temp_start + i] = -1.0
                layer.W1[r, -1] = large
                layer.b1[r, :] = -large
                layer.W2[temp_start + i, r] = 1.0
                r += 1

        # FFN Section 2: Scratchpad correction — snap scr_sub/scr_min to bipolar.
        eps = 0.1
        scale = 1.0 / (1.0 - 2.0 * eps)

        for target in (scr_sub, scr_min):
            for i in range(N):
                layer.W1[r, target + i] = 1.0
                layer.W1[r + 1, target + i] = 1.0
                layer.W1[r + 2, target + i] = 1.0
                layer.W1[r + 3, target + i] = 1.0
                layer.W1[r + 4, target + i] = -1.0
                layer.W1[r + 5, target + i] = 1.0

                for k in range(6):
                    layer.W1[r + k, -1] = large

                layer.b1[r, :] = 1.0 - eps - large
                layer.b1[r + 1, :] = eps - large
                layer.b1[r + 2, :] = -eps - large
                layer.b1[r + 3, :] = -1.0 + eps - large
                layer.b1[r + 4, :] = -large
                layer.b1[r + 5, :] = -large

                layer.W2[target + i, r] = scale
                layer.W2[target + i, r + 1] = -scale
                layer.W2[target + i, r + 2] = scale
                layer.W2[target + i, r + 3] = -scale
                layer.W2[target + i, r + 4] = 1.0
                layer.W2[target + i, r + 5] = -1.0

                r += 6

            layer.b2[target:target + N, :cfg.s] = -1.0

        return layer

    def _init_scratchpad_correction_v4(self) -> TransformerLayer:
        """Snap scr_sub/scr_min back to bipolar in scratchpad columns."""
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn

        layer = TransformerLayer(cfg, logn, 12 * N, num_heads=1)

        eps = 0.1
        scale = 1.0 / (1.0 - 2.0 * eps)
        large = 100.0
        scr_sub = cfg.idx_scratchpad
        scr_min = cfg.idx_scratchpad + N

        row = 0
        for target in (scr_sub, scr_min):
            for i in range(N):
                r = row + 6 * i
                layer.W1[r, target + i] = 1.0
                layer.W1[r + 1, target + i] = 1.0
                layer.W1[r + 2, target + i] = 1.0
                layer.W1[r + 3, target + i] = 1.0
                layer.W1[r + 4, target + i] = -1.0
                layer.W1[r + 5, target + i] = 1.0

                # Gate to scratchpad columns only.
                for k in range(6):
                    layer.W1[r + k, -1] = large

                layer.b1[r, :] = 1.0 - eps - large
                layer.b1[r + 1, :] = eps - large
                layer.b1[r + 2, :] = -eps - large
                layer.b1[r + 3, :] = -1.0 + eps - large
                layer.b1[r + 4, :] = -large
                layer.b1[r + 5, :] = -large

                layer.W2[target + i, r] = scale
                layer.W2[target + i, r + 1] = -scale
                layer.W2[target + i, r + 2] = scale
                layer.W2[target + i, r + 3] = -scale
                layer.W2[target + i, r + 4] = 1.0
                layer.W2[target + i, r + 5] = -1.0

            layer.b2[target:target + N, :cfg.s] = -1.0
            row += 6 * N

        return layer

    def _init_direct_subtract(self) -> TransformerLayer:
        """
        Single-layer direct subtraction: scr_min = scr_min - scr_sub.

        Replaces the 3-layer two's complement approach (flip, add-1, add)
        with a single layer that computes a - b directly using the
        6-threshold-per-bit pattern with:
          - Negated weights for scr_sub (implements bit-flip)
          - s_bias = 2^i (carry_in = 1, completes two's complement)

        The partial binary sum at position i (1-indexed from LSB) is:
          S_i = sum_{j=1}^{i} [bit_a_j + ~bit_b_j] * 2^(j-1) + 1
              = sum_{j=1}^{i} (a_j - b_j) * 2^(j-2) + 2^i

        where a = scr_min (minuend) and b = scr_sub (subtrahend).
        The 6 ReLU thresholds extract the result bit from S_i.
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn

        scr_sub = cfg.idx_scratchpad       # subtrahend (b)
        scr_min = cfg.idx_scratchpad + N    # minuend (a), also output

        # 6*N for addition with borrow chain, 4*N for clearing scr_sub + scr_min
        num_ffn = 10 * N
        layer = TransformerLayer(cfg, logn, num_ffn, num_heads=1)

        # === Section 1: Direct subtraction via 6-threshold pattern ===
        # For each bit position i (1-indexed from LSB):
        #   h = sum_{j=1}^{i} elem_j * (scr_min[N-j] - scr_sub[N-j])
        #   6 ReLU rows threshold h to extract the result bit
        for i in range(1, N + 1):
            row_base = 6 * (N - i)

            # Read bits j=1..i from both operands
            for j in range(1, i + 1):
                elem = (2**(j-1)) / 2
                # scr_min with positive weight (minuend)
                layer.W1[row_base, scr_min + N - j] = elem
                layer.W1[row_base + 1, scr_min + N - j] = elem
                layer.W1[row_base + 2, scr_min + N - j] = -elem
                layer.W1[row_base + 3, scr_min + N - j] = -elem
                layer.W1[row_base + 4, scr_min + N - j] = elem
                layer.W1[row_base + 5, scr_min + N - j] = elem
                # scr_sub with NEGATIVE weight (negation = bit flip)
                layer.W1[row_base, scr_sub + N - j] = -elem
                layer.W1[row_base + 1, scr_sub + N - j] = -elem
                layer.W1[row_base + 2, scr_sub + N - j] = elem
                layer.W1[row_base + 3, scr_sub + N - j] = elem
                layer.W1[row_base + 4, scr_sub + N - j] = -elem
                layer.W1[row_base + 5, scr_sub + N - j] = -elem

            # Biases: s_bias = 2^i (carry_in = 1, vs 2^i - 1 for plain add)
            s_bias = 2**i
            layer.b1[row_base, 0] = s_bias - 2**(i-1) + 1
            layer.b1[row_base + 1, 0] = s_bias - 2**(i-1)
            layer.b1[row_base + 2, 0] = -s_bias + 2**i
            layer.b1[row_base + 3, 0] = -s_bias + 2**i - 1
            layer.b1[row_base + 4, 0] = s_bias - 3 * 2**(i-1) + 1
            layer.b1[row_base + 5, 0] = s_bias - 3 * 2**(i-1)

            # Write result to scr_min
            layer.W2[scr_min + N - i, row_base:row_base + 6] = 2 * torch.tensor(
                [1, -1, 1, -1, 1, -1], dtype=torch.float32)

        # b2 baseline for scr_min (same as original add layer)
        layer.b2[scr_min:scr_min + N, 0] = -3.0

        # === Section 2: Clear scr_sub and scr_min ===
        # The residual connection preserves scr_min's original value.
        # We subtract it (clearing) so only the computed result remains.
        offset = 6 * N

        # Clear scr_sub: scr_sub[i] -= scr_sub[i]
        layer.W1[offset:offset + N, scr_sub:scr_sub + N] = torch.eye(N)
        layer.W1[offset + N:offset + 2*N, scr_sub:scr_sub + N] = -torch.eye(N)
        layer.W2[scr_sub:scr_sub + N, offset:offset + N] = -torch.eye(N)
        layer.W2[scr_sub:scr_sub + N, offset + N:offset + 2*N] = torch.eye(N)

        # Clear scr_min: scr_min[i] -= scr_min[i] (result replaces via Section 1)
        layer.W1[offset + 2*N:offset + 3*N, scr_min:scr_min + N] = torch.eye(N)
        layer.W1[offset + 3*N:offset + 4*N, scr_min:scr_min + N] = -torch.eye(N)
        layer.W2[scr_min:scr_min + N, offset + 2*N:offset + 3*N] = -torch.eye(N)
        layer.W2[scr_min:scr_min + N, offset + 3*N:offset + 4*N] = torch.eye(N)

        return layer

    def _init_cond_branch_v4(self) -> List[TransformerLayer]:
        """Conditional branching with extended opcodes (2 layers).

        Layer 1 (merged): Branch flag computation + PC increment.
          These are independent FFN computations (flag reads scratchpad/opcodes,
          PC+1 reads PC rows) merged into a single layer.
        Layer 2: Select between branch target and PC+1 based on flag.
        """
        cfg = self.cfg
        N = cfg.N
        logn = cfg.logn
        large_const = 100.0

        # Merged layer: branch flag (2+2*N+41=59 rows) + PC+1 (8*logn=80 rows)
        num_ffn_flag = 2 + 2*N + 41  # +1 for OP_STORE suppress
        num_ffn_pc = 8 * logn
        num_ffn = num_ffn_flag + num_ffn_pc
        layer1 = TransformerLayer(cfg, logn, num_ffn, num_heads=1)
        
        ref_result = cfg.idx_scratchpad + N
        ref_flag = cfg.idx_scratchpad
        addr_a = cfg.idx_scratch_cmd
        bit32 = logn - 6
        bit16 = logn - 5
        mode_bit0 = addr_a + bit32
        mode_bit1 = addr_a + bit16
        
        # SUBLEQ mode gate (a >= 32)
        subleq_row = 2 + 2*N
        layer1.W1[subleq_row, mode_bit0] = 1.0
        layer1.b1[subleq_row, 0] = -0.5
        
        # Sign bit
        layer1.W1[0, ref_result] = 1.0
        
        # Zero check
        layer1.W1[1, ref_result:ref_result+N] = -1.0
        layer1.b1[1, 0] = -N + 1
        
        # Clear result
        layer1.W1[2:2+N, ref_result:ref_result+N] = torch.eye(N)
        layer1.W1[2+N:2+2*N, ref_result:ref_result+N] = -torch.eye(N)
        
        jmp_row = subleq_row + 1
        jmp_pat = opcode_pattern(OP_JMP, logn)
        for i in range(logn):
            layer1.W1[jmp_row, addr_a + i] = jmp_pat[i].item()
        layer1.b1[jmp_row, 0] = -logn + 0.5
        
        halt_row = jmp_row + 1
        halt_pat = opcode_pattern(OP_HALT, logn)
        for i in range(logn):
            layer1.W1[halt_row, addr_a + i] = halt_pat[i].item()
        layer1.b1[halt_row, 0] = -logn + 0.5
        
        # Suppress SUBLEQ for opcodes 4-7 (JZ, JNZ, INC, DEC)
        suppress_row = halt_row + 1
        for i in range(logn - 3):
            layer1.W1[suppress_row, addr_a + i] = -1.0
        layer1.W1[suppress_row, addr_a + (logn - 3)] = 1.0
        layer1.b1[suppress_row, 0] = -(logn - 3)
        
        # Suppress SUBLEQ for AND (opcode 12)
        suppress_and_row = suppress_row + 1
        and_pat = opcode_pattern(OP_AND, logn)
        for i in range(logn):
            layer1.W1[suppress_and_row, addr_a + i] = and_pat[i].item()
        layer1.b1[suppress_and_row, 0] = -logn + 1.0
        
        # Suppress SUBLEQ for OR (opcode 13)
        suppress_or_row = suppress_and_row + 1
        or_pat = opcode_pattern(OP_OR, logn)
        for i in range(logn):
            layer1.W1[suppress_or_row, addr_a + i] = or_pat[i].item()
        layer1.b1[suppress_or_row, 0] = -logn + 1.0
        
        # Suppress SUBLEQ for XOR (opcode 14)
        suppress_xor_row = suppress_or_row + 1
        xor_pat = opcode_pattern(OP_XOR, logn)
        for i in range(logn):
            layer1.W1[suppress_xor_row, addr_a + i] = xor_pat[i].item()
        layer1.b1[suppress_xor_row, 0] = -logn + 1.0
        
        # Suppress SUBLEQ for MOV (opcode 1) - MOV also shouldn't branch
        suppress_mov_row = suppress_xor_row + 1
        mov_pat = opcode_pattern(OP_MOV, logn)
        for i in range(logn):
            layer1.W1[suppress_mov_row, addr_a + i] = mov_pat[i].item()
        layer1.b1[suppress_mov_row, 0] = -logn + 1.0
        
        # Suppress SUBLEQ for ADD (opcode 2) - ADD shouldn't branch
        suppress_add_row = suppress_mov_row + 1
        add_pat = opcode_pattern(OP_ADD, logn)
        for i in range(logn):
            layer1.W1[suppress_add_row, addr_a + i] = add_pat[i].item()
        layer1.b1[suppress_add_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for SUB (opcode 15) - SUB shouldn't branch
        suppress_sub_row = suppress_add_row + 1
        sub_pat = opcode_pattern(OP_SUB, logn)
        for i in range(logn):
            layer1.W1[suppress_sub_row, addr_a + i] = sub_pat[i].item()
        layer1.b1[suppress_sub_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for SHL (opcode 8)
        suppress_shl_row = suppress_sub_row + 1
        shl_pat = opcode_pattern(OP_SHL, logn)
        for i in range(logn):
            layer1.W1[suppress_shl_row, addr_a + i] = shl_pat[i].item()
        layer1.b1[suppress_shl_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for SHR (opcode 9)
        suppress_shr_row = suppress_shl_row + 1
        shr_pat = opcode_pattern(OP_SHR, logn)
        for i in range(logn):
            layer1.W1[suppress_shr_row, addr_a + i] = shr_pat[i].item()
        layer1.b1[suppress_shr_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for LOAD (opcode 11)
        suppress_load_row = suppress_shr_row + 1
        load_pat = opcode_pattern(OP_LOAD, logn)
        for i in range(logn):
            layer1.W1[suppress_load_row, addr_a + i] = load_pat[i].item()
        layer1.b1[suppress_load_row, 0] = -logn + 1.0

        # JZ: branch if zero
        jz_row = suppress_load_row + 1
        jz_pat = opcode_pattern(OP_JZ, logn)
        for i in range(logn):
            layer1.W1[jz_row, addr_a + i] = jz_pat[i].item()
        layer1.W1[jz_row, ref_result:ref_result+N] = -1.0
        layer1.b1[jz_row, 0] = -(logn + N) + 0.5
        
        # JNZ: branch if not zero
        jnz_row = jz_row + 1
        jnz_pat = opcode_pattern(OP_JNZ, logn)
        for i in range(logn):
            layer1.W1[jnz_row, addr_a + i] = jnz_pat[i].item() * 10.0
        layer1.W1[jnz_row, ref_result:ref_result+N] = 1.0
        layer1.b1[jnz_row, 0] = -(logn * 10 - N + 0.5)

        # CMP (BLTZ): suppress zero_check contribution so only negative branches.
        # Default ref_flag = sign_bit(row0) + zero_check(row1).
        # For CMP: value<0 → 1+0=1 (branch ✓), value=0 → 0+1=1 (wrong!), value>0 → 0+0=0 (✓).
        # This row fires when opcode=CMP AND value is zero (all bits = -1).
        # W2 = -1.0 cancels the zero_check's +1 contribution.
        cmp_zero_row = jnz_row + 1
        cmp_pat = opcode_pattern(OP_CMP, logn)
        for i in range(logn):
            layer1.W1[cmp_zero_row, addr_a + i] = cmp_pat[i].item()
        layer1.W1[cmp_zero_row, ref_result:ref_result+N] = -1.0
        layer1.b1[cmp_zero_row, 0] = -(logn + N) + 0.5

        # Suppress SUBLEQ for FIND (opcode 16)
        suppress_find_row = cmp_zero_row + 1
        find_pat = opcode_pattern(OP_FIND, logn)
        for i in range(logn):
            layer1.W1[suppress_find_row, addr_a + i] = find_pat[i].item()
        layer1.b1[suppress_find_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for SWAP (opcode 17)
        suppress_swap_row = suppress_find_row + 1
        swap_pat = opcode_pattern(OP_SWAP, logn)
        for i in range(logn):
            layer1.W1[suppress_swap_row, addr_a + i] = swap_pat[i].item()
        layer1.b1[suppress_swap_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for CMOV (opcode 18)
        suppress_cmov_row = suppress_swap_row + 1
        cmov_pat = opcode_pattern(OP_CMOV, logn)
        for i in range(logn):
            layer1.W1[suppress_cmov_row, addr_a + i] = cmov_pat[i].item()
        layer1.b1[suppress_cmov_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for MULACC (opcode 19)
        suppress_mulacc_row = suppress_cmov_row + 1
        mulacc_pat = opcode_pattern(OP_MULACC, logn)
        for i in range(logn):
            layer1.W1[suppress_mulacc_row, addr_a + i] = mulacc_pat[i].item()
        layer1.b1[suppress_mulacc_row, 0] = -logn + 1.0

        # Suppress SUBLEQ for STORE (opcode 20)
        suppress_store_row = suppress_mulacc_row + 1
        store_pat = opcode_pattern(OP_STORE, logn)
        for i in range(logn):
            layer1.W1[suppress_store_row, addr_a + i] = store_pat[i].item()
        layer1.b1[suppress_store_row, 0] = -logn + 1.0

        # Combine outputs
        layer1.W2[ref_flag, 0] = 1.0
        layer1.W2[ref_flag, 1] = 1.0
        layer1.W2[ref_flag, jmp_row] = 2.0
        layer1.W2[ref_flag, halt_row] = 2.0
        layer1.W2[ref_flag, suppress_row] = -1.0
        layer1.W2[ref_flag, suppress_and_row] = -1.0
        layer1.W2[ref_flag, suppress_or_row] = -1.0
        layer1.W2[ref_flag, suppress_xor_row] = -1.0
        layer1.W2[ref_flag, suppress_mov_row] = -1.0
        layer1.W2[ref_flag, suppress_add_row] = -1.0
        layer1.W2[ref_flag, suppress_sub_row] = -1.0
        layer1.W2[ref_flag, suppress_shl_row] = -1.0
        layer1.W2[ref_flag, suppress_shr_row] = -1.0
        layer1.W2[ref_flag, suppress_load_row] = -1.0
        layer1.W2[ref_flag, suppress_find_row] = -1.0
        layer1.W2[ref_flag, suppress_swap_row] = -1.0
        layer1.W2[ref_flag, suppress_cmov_row] = -1.0
        layer1.W2[ref_flag, suppress_mulacc_row] = -1.0
        layer1.W2[ref_flag, suppress_store_row] = -1.0
        layer1.W2[ref_flag, jz_row] = 2.0
        layer1.W2[ref_flag, jnz_row] = 2.0
        layer1.W2[ref_flag, cmp_zero_row] = -2.0
        
        layer1.W2[ref_result:ref_result+N, 2:2+N] = -torch.eye(N)
        layer1.W2[ref_result:ref_result+N, 2+N:2+2*N] = torch.eye(N)

        # --- PC+1 (merged into same layer, offset by num_ffn_flag) ---
        ref_pc = cfg.idx_pc
        pc_off = num_ffn_flag  # FFN row offset for PC+1 section

        for i in range(1, logn + 1):
            for j in range(1, i + 1):
                elem = (2**(j-1)) / 2.0
                rb = pc_off + 6 * (logn - i)
                layer1.W1[rb:rb+6, ref_pc + logn - j] = torch.tensor([elem, elem, -elem, -elem, elem, elem])
            s_bias = (2**i - 1) / 2.0 + 1
            rb = pc_off + 6 * (logn - i)
            layer1.b1[rb:rb+6, 0] = torch.tensor([
                s_bias - 2**(i-1) + 1, s_bias - 2**(i-1),
                -s_bias + 2**i, -s_bias + 2**i - 1,
                s_bias - 3*2**(i-1) + 1, s_bias - 3*2**(i-1)
            ])
            layer1.W2[ref_pc + logn - i, rb:rb+6] = 2*torch.tensor([1,-1,1,-1,1,-1], dtype=torch.float32)
        layer1.b2[ref_pc:ref_pc+logn, 0] = -3.0

        off = pc_off + 6*logn
        layer1.W1[off:off+logn, ref_pc:ref_pc+logn] = torch.eye(logn)
        layer1.W1[off+logn:off+2*logn, ref_pc:ref_pc+logn] = -torch.eye(logn)
        layer1.W2[ref_pc:ref_pc+logn, off:off+logn] = -torch.eye(logn)
        layer1.W2[ref_pc:ref_pc+logn, off+logn:off+2*logn] = torch.eye(logn)

        # L8: Select PC
        layer2 = TransformerLayer(cfg, logn, 6*logn + 2, num_heads=1)
        ref_c = cfg.idx_scratch_cmd + 2*logn

        layer2.W1[:logn, ref_c:ref_c+logn] = torch.eye(logn)
        layer2.W1[:logn, ref_flag] = 1.0
        layer2.b1[:logn, 0] = -1.0
        layer2.W1[logn:2*logn, ref_c:ref_c+logn] = -torch.eye(logn)
        layer2.W1[logn:2*logn, ref_flag] = 1.0
        layer2.b1[logn:2*logn, 0] = -1.0

        layer2.W1[2*logn:3*logn, ref_pc:ref_pc+logn] = torch.eye(logn)
        layer2.W1[2*logn:3*logn, ref_flag] = -1.0
        layer2.W1[3*logn:4*logn, ref_pc:ref_pc+logn] = -torch.eye(logn)
        layer2.W1[3*logn:4*logn, ref_flag] = -1.0

        layer2.W1[4*logn:5*logn, ref_pc:ref_pc+logn] = torch.eye(logn)
        layer2.W1[5*logn:6*logn, ref_pc:ref_pc+logn] = -torch.eye(logn)

        layer2.W2[ref_pc:ref_pc+logn, :logn] = torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, logn:2*logn] = -torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, 2*logn:3*logn] = torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, 3*logn:4*logn] = -torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, 4*logn:5*logn] = -torch.eye(logn)
        layer2.W2[ref_pc:ref_pc+logn, 5*logn:6*logn] = torch.eye(logn)

        layer2.W1[6*logn, ref_flag] = 1.0
        layer2.W1[6*logn + 1, ref_flag] = -1.0
        layer2.W2[ref_flag, 6*logn] = -1.0
        layer2.W2[ref_flag, 6*logn + 1] = 1.0

        # Gate cond-branch FFN rows to scratchpad columns only to prevent
        # PC/scratchpad leakage into command/memory columns.
        for layer in (layer1, layer2):
            layer.W1[:, -1] = large_const
            layer.b1[:, :] -= large_const

        return [layer1, layer2]


def init_state(cfg, memory, commands):
    """Initialize state tensor."""
    X = torch.zeros(cfg.d_model, cfg.n)
    X[-1, :cfg.s] = 1.0

    # Position encoding for columns 1 to n-1
    # IMPORTANT: Column 0 intentionally has pos_enc = [0,0,...,0]
    # This ensures Q @ X[:,0] = PC only, without pos(0) interference
    for col in range(1, cfg.n):
        X[cfg.idx_pos_enc:cfg.idx_pos_enc + cfg.logn, col] = to_bipolar(col, cfg.logn)

    for i, val in enumerate(memory):
        if i >= cfg.m:
            break
        col = cfg.s + i
        X[cfg.idx_memory:cfg.idx_memory + cfg.N, col] = signed_to_bipolar(val, cfg.N)

    # Address tags: each memory column stores its own index (for LOAD indirect access)
    if hasattr(cfg, 'idx_tag'):
        for i in range(cfg.m):
            col = cfg.s + i
            X[cfg.idx_tag:cfg.idx_tag + cfg.N, col] = signed_to_bipolar(i, cfg.N)

    for i, (a, b, c) in enumerate(commands):
        col = cfg.s + cfg.m + i
        if col >= cfg.n:
            break
        X[:cfg.logn, col] = to_bipolar(a, cfg.logn)
        X[cfg.logn:2*cfg.logn, col] = to_bipolar(b, cfg.logn)
        X[2*cfg.logn:3*cfg.logn, col] = to_bipolar(c, cfg.logn)

    pc = cfg.s + cfg.m
    X[cfg.idx_pc:cfg.idx_pc + cfg.logn, 0] = to_bipolar(pc, cfg.logn)

    return X


def read_memory(X, cfg):
    """Read memory as integers."""
    memory = []
    for i in range(cfg.m):
        col = cfg.s + i
        bipolar = X[cfg.idx_memory:cfg.idx_memory + cfg.N, col]
        memory.append(signed_from_bipolar(bipolar))
    return memory


def get_pc(X, cfg):
    """Get PC value."""
    return from_bipolar(X[cfg.idx_pc:cfg.idx_pc + cfg.logn, 0])


# Backwards-compatibility aliases
ExtendedConfigV4 = LoomConfig
ExtendedNeuralComputerV4 = LoomComputer
init_state_v4 = init_state
read_memory_v4 = read_memory
get_pc_v4 = get_pc


if __name__ == "__main__":
    # Quick sanity check
    cfg = LoomConfig(s=32, m=8, n=64, N=8)
    ext = LoomComputer(cfg)
    
    print("V4 Implementation Ready")
    print(f"Config: s={cfg.s}, m={cfg.m}, n={cfg.n}, N={cfg.N}, logn={cfg.logn}")
    print(f"d_model = {cfg.d_model}")
    print(f"Buffer rows = {cfg.nrows_buffer}")
