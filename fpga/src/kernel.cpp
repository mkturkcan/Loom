/*
 * Vitis HLS Kernel for Neural SUBLEQ V4 Transformer.
 *
 * v2: n=1024, argmax attention (no n×n matrix), halt detection.
 *
 * Design:
 *   - All weights loaded into on-chip BRAM at kernel start
 *   - State matrix X[d_model][n] in URAM, processed in-place
 *   - Argmax attention: streaming top-2 per key row, no n×n storage
 *   - No hls::expf needed (argmax replaces softmax)
 *   - Halt detection: checks PC=0 between steps, exits early
 *   - Program-independent: weights are universal, any compiled program
 *     can be loaded via the state DDR buffer without recompilation
 *
 * Resource estimate (Alveo U200, 44 MB on-chip):
 *   X[155][1024]     = 635 KB (URAM)
 *   VX_acc[155][1024] = 635 KB (URAM)
 *   ff[640][1024]     = 2.6 MB (URAM)
 *   QX,KX[10][1024]  = 80 KB (BRAM)
 *   Weights (sparse)  = ~300 KB (BRAM)
 *   Total: ~4 MB — fits easily (was 40 MB with softmax attn matrix)
 */

#include <cmath>
#include <cstring>
#include <ap_int.h>

// ---- Model constants (compile-time for HLS) ----
// V4 config: s=32, m=160, n=1024, N=8, logn=10
// Standard 8-layer architecture (no merges for reliability)
// STORE opcode enables n=1024 (sudoku compiles to 284 instructions, fits in 832 slots)
// Argmax attention eliminates the n×n matrix that caused synthesis OOM
static const int D_MODEL    = 155;
static const int N_SEQ      = 1024;
static const int N_LAYERS   = 8;
static const int MAX_Q_ROWS = 10;    // logn=10
static const int MAX_W1     = 640;   // max FFN hidden (L2)
static const int MAX_HEADS  = 4;
static const int MAX_NNZ    = 5000;  // L2 W1 has ~4004 nnz with STORE opcode
static const int LOGN       = 10;
static const int IDX_PC     = 84;    // PC row start for this config

static const int STATE_SIZE = D_MODEL * N_SEQ;  // 155 * 1024 = 158,720

// Argmax tie threshold: scores within 1.0 are considered tied (50/50 split)
static const float TIE_THRESH = 1.0f;

// ---- Sparse COO entry ----
struct COOEntry {
    short row, col;
    float val;
};

// ---- Compact bias: 3 values per row ----
struct CompactBias {
    float col0, scr, mem;
};

// ---- Per-head weight storage ----
struct HeadWeights {
    COOEntry Q[MAX_NNZ], K[MAX_NNZ], V[MAX_NNZ];
    int q_nnz, k_nnz, v_nnz;
    int q_rows;
};

// ---- Per-layer weight storage ----
struct LayerWeights {
    int num_heads, q_rows, w1_rows;
    HeadWeights heads[MAX_HEADS];
    COOEntry W1[MAX_NNZ], W2[MAX_NNZ];
    int w1_nnz, w2_nnz;
    CompactBias b1[MAX_W1];
    CompactBias b2[D_MODEL];
};


// ---- Sparse matvec column ----
static void sparse_matvec_col(
    const COOEntry* entries, int nnz,
    const float* X, float* out, int col_j
) {
    for (int e = 0; e < nnz; e++) {
#pragma HLS PIPELINE II=1
        out[entries[e].row * N_SEQ + col_j] +=
            entries[e].val * X[entries[e].col * N_SEQ + col_j];
    }
}

// ---- Argmax attention (replaces softmax dim=0) ----
// For each source column i: find top-2 target columns j that maximize
// score[i,j] = KX[:,i]·QX[:,j], distribute VX[:,i] to those targets.
// Weight 1.0 (unique max) or 0.5 (tied). No n×n storage — O(n) memory.
// With Q=K (symmetric scores), equivalent to column-wise argmax.
static void argmax_attention(
    const float* QX, const float* KX,  // [q_rows × N_SEQ]
    int q_rows,
    const COOEntry* V, int v_nnz,
    const float* X,      // [D_MODEL × N_SEQ]
    float* VX_acc         // [D_MODEL × N_SEQ], accumulated output
) {
    for (int i = 0; i < N_SEQ; i++) {
        // Check if KX column i is zero (no-op head for this position)
        bool is_zero = true;
        for (int q = 0; q < q_rows; q++) {
            if (KX[q * N_SEQ + i] != 0.0f) { is_zero = false; break; }
        }
        if (is_zero) continue;

        // Find top-2 query columns for key row i
        float best1_v = -1e30f, best2_v = -1e30f;
        int best1_j = 0, best2_j = 0;

        for (int j = 0; j < N_SEQ; j++) {
#pragma HLS PIPELINE II=1
            float score = 0.0f;
            for (int q = 0; q < q_rows; q++) {
                score += KX[q * N_SEQ + i] * QX[q * N_SEQ + j];
            }
            if (score > best1_v) {
                best2_v = best1_v; best2_j = best1_j;
                best1_v = score; best1_j = j;
            } else if (score > best2_v) {
                best2_v = score; best2_j = j;
            }
        }

        // Distribute V @ X[:,i] to winning columns
        bool tied = (best1_v - best2_v) < TIE_THRESH;
        float w1 = tied ? 0.5f : 1.0f;

        for (int e = 0; e < v_nnz; e++) {
#pragma HLS PIPELINE II=1
            float vx = V[e].val * X[V[e].col * N_SEQ + i];
            VX_acc[V[e].row * N_SEQ + best1_j] += w1 * vx;
            if (tied) {
                VX_acc[V[e].row * N_SEQ + best2_j] += 0.5f * vx;
            }
        }
    }
}

// ---- Add compact bias ----
static void add_bias_compact(
    float* out, int rows, int s, const CompactBias* bias
) {
    for (int r = 0; r < rows; r++) {
        out[r * N_SEQ + 0] += bias[r].col0;
        for (int j = 1; j < s; j++) {
#pragma HLS PIPELINE II=1
            out[r * N_SEQ + j] += bias[r].scr;
        }
        for (int j = s; j < N_SEQ; j++) {
#pragma HLS PIPELINE II=1
            out[r * N_SEQ + j] += bias[r].mem;
        }
    }
}

// ---- ReLU ----
static void relu(float* data, int count) {
    for (int i = 0; i < count; i++) {
#pragma HLS PIPELINE II=1
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

// ---- Extract PC from state (bipolar) ----
static int get_pc(const float* X) {
    int pc = 0;
    for (int b = 0; b < LOGN; b++) {
        if (X[(IDX_PC + b) * N_SEQ] > 0.0f)
            pc |= (1 << (LOGN - 1 - b));
    }
    return pc;
}


/*
 * Top-level kernel.
 *
 * Runs up to num_steps transformer steps, halting early if PC=0.
 * Returns actual steps executed via the return value.
 */
extern "C" {

void transformer_kernel(
    const float* weights_buf,
    float* state_buf,
    int num_steps,
    int weights_size,
    int d_model,
    int n_seq,
    int n_layers,
    float lam
) {
#pragma HLS INTERFACE m_axi port=weights_buf offset=slave bundle=gmem0 depth=131072
#pragma HLS INTERFACE m_axi port=state_buf   offset=slave bundle=gmem1 depth=158720
#pragma HLS INTERFACE s_axilite port=weights_buf
#pragma HLS INTERFACE s_axilite port=state_buf
#pragma HLS INTERFACE s_axilite port=num_steps
#pragma HLS INTERFACE s_axilite port=weights_size
#pragma HLS INTERFACE s_axilite port=d_model
#pragma HLS INTERFACE s_axilite port=n_seq
#pragma HLS INTERFACE s_axilite port=n_layers
#pragma HLS INTERFACE s_axilite port=lam
#pragma HLS INTERFACE s_axilite port=return

    // ---- Load state from DDR ----
    static float X[STATE_SIZE];
#pragma HLS BIND_STORAGE variable=X type=ram_2p impl=uram

    for (int i = 0; i < STATE_SIZE; i++) {
#pragma HLS PIPELINE II=1
        X[i] = state_buf[i];
    }

    // ---- Load weights from DDR ----
    static LayerWeights layers[N_LAYERS];
#pragma HLS BIND_STORAGE variable=layers type=ram_2p impl=uram

    int wptr = 0;
    int s = (int)weights_buf[wptr++];
    int nl = (int)weights_buf[wptr++];

    for (int li = 0; li < nl && li < N_LAYERS; li++) {
        LayerWeights& lw = layers[li];
        lw.num_heads = (int)weights_buf[wptr++];
        lw.q_rows    = (int)weights_buf[wptr++];
        lw.w1_rows   = (int)weights_buf[wptr++];

        for (int h = 0; h < lw.num_heads && h < MAX_HEADS; h++) {
            HeadWeights& hw = lw.heads[h];
            hw.q_rows = lw.q_rows;
            hw.q_nnz = (int)weights_buf[wptr++];
            for (int e = 0; e < hw.q_nnz && e < MAX_NNZ; e++) {
                hw.Q[e].row = (short)weights_buf[wptr++];
                hw.Q[e].col = (short)weights_buf[wptr++];
                hw.Q[e].val = weights_buf[wptr++];
            }
            hw.k_nnz = (int)weights_buf[wptr++];
            for (int e = 0; e < hw.k_nnz && e < MAX_NNZ; e++) {
                hw.K[e].row = (short)weights_buf[wptr++];
                hw.K[e].col = (short)weights_buf[wptr++];
                hw.K[e].val = weights_buf[wptr++];
            }
            hw.v_nnz = (int)weights_buf[wptr++];
            for (int e = 0; e < hw.v_nnz && e < MAX_NNZ; e++) {
                hw.V[e].row = (short)weights_buf[wptr++];
                hw.V[e].col = (short)weights_buf[wptr++];
                hw.V[e].val = weights_buf[wptr++];
            }
        }

        lw.w1_nnz = (int)weights_buf[wptr++];
        for (int e = 0; e < lw.w1_nnz && e < MAX_NNZ; e++) {
            lw.W1[e].row = (short)weights_buf[wptr++];
            lw.W1[e].col = (short)weights_buf[wptr++];
            lw.W1[e].val = weights_buf[wptr++];
        }
        int b1_rows = (int)weights_buf[wptr++];
        for (int r = 0; r < b1_rows && r < MAX_W1; r++) {
            lw.b1[r].col0 = weights_buf[wptr++];
            lw.b1[r].scr  = weights_buf[wptr++];
            lw.b1[r].mem  = weights_buf[wptr++];
        }
        lw.w2_nnz = (int)weights_buf[wptr++];
        for (int e = 0; e < lw.w2_nnz && e < MAX_NNZ; e++) {
            lw.W2[e].row = (short)weights_buf[wptr++];
            lw.W2[e].col = (short)weights_buf[wptr++];
            lw.W2[e].val = weights_buf[wptr++];
        }
        int b2_rows = (int)weights_buf[wptr++];
        for (int r = 0; r < b2_rows && r < D_MODEL; r++) {
            lw.b2[r].col0 = weights_buf[wptr++];
            lw.b2[r].scr  = weights_buf[wptr++];
            lw.b2[r].mem  = weights_buf[wptr++];
        }
    }

    // ---- Run transformer steps with halt detection ----
    static float QX[MAX_Q_ROWS * N_SEQ];
    static float KX[MAX_Q_ROWS * N_SEQ];
    static float VX_acc[STATE_SIZE];
    static float ff_hidden[MAX_W1 * N_SEQ];
#pragma HLS BIND_STORAGE variable=QX       type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=KX       type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=VX_acc   type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=ff_hidden type=ram_2p impl=uram

    int steps_done = 0;

    for (int step = 0; step < num_steps; step++) {
        // Check PC before each step
        if (get_pc(X) == 0) break;

        for (int li = 0; li < nl && li < N_LAYERS; li++) {
            const LayerWeights& lw = layers[li];
            int q_rows = lw.q_rows;

            // ---- Attention (argmax) ----
            for (int i = 0; i < STATE_SIZE; i++) {
#pragma HLS PIPELINE II=1
                VX_acc[i] = 0.0f;
            }

            for (int h = 0; h < lw.num_heads; h++) {
                const HeadWeights& hw = lw.heads[h];

                // Skip no-op heads (Q=K=V all zero)
                if (hw.q_nnz == 0 && hw.k_nnz == 0 && hw.v_nnz == 0) continue;

                // QX = Q @ X
                for (int i = 0; i < q_rows * N_SEQ; i++) {
#pragma HLS PIPELINE II=1
                    QX[i] = 0.0f;
                }
                for (int j = 0; j < N_SEQ; j++) {
                    sparse_matvec_col(hw.Q, hw.q_nnz, X, QX, j);
                }

                // KX = K @ X
                for (int i = 0; i < q_rows * N_SEQ; i++) {
#pragma HLS PIPELINE II=1
                    KX[i] = 0.0f;
                }
                for (int j = 0; j < N_SEQ; j++) {
                    sparse_matvec_col(hw.K, hw.k_nnz, X, KX, j);
                }

                // Argmax attention (no n×n matrix!)
                argmax_attention(QX, KX, q_rows, hw.V, hw.v_nnz, X, VX_acc);
            }

            // Residual
            for (int i = 0; i < STATE_SIZE; i++) {
#pragma HLS PIPELINE II=1
                X[i] += VX_acc[i];
            }

            // ---- FFN ----
            for (int i = 0; i < lw.w1_rows * N_SEQ; i++) {
#pragma HLS PIPELINE II=1
                ff_hidden[i] = 0.0f;
            }
            for (int j = 0; j < N_SEQ; j++) {
                sparse_matvec_col(lw.W1, lw.w1_nnz, X, ff_hidden, j);
            }
            add_bias_compact(ff_hidden, lw.w1_rows, s, lw.b1);
            relu(ff_hidden, lw.w1_rows * N_SEQ);

            for (int j = 0; j < N_SEQ; j++) {
                sparse_matvec_col(lw.W2, lw.w2_nnz, ff_hidden, X, j);
            }
            add_bias_compact(X, d_model, s, lw.b2);
        }

        steps_done++;
    }

    // ---- Write state back to DDR ----
    for (int i = 0; i < STATE_SIZE; i++) {
#pragma HLS PIPELINE II=1
        state_buf[i] = X[i];
    }

}

} // extern "C"
