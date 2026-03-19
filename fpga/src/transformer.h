#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* ============================================================
 * Neural SUBLEQ V4 Transformer — FPGA-targeted implementation
 *
 * Key properties exploited:
 *   - Q/K/V/W1/W2 are >99% sparse (only ~8K nonzeros total)
 *   - Biases are compact: 2 values per row (col0 + broadcast)
 *   - lambda=10 softmax is nearly one-hot → use hard attention (argmax)
 *   - All weight values are multiples of 0.25
 * ============================================================ */

// ---- Configuration ----
struct Config {
    int d_model;   // 155 or 164
    int n;         // 1024 or 2048
    int s;         // 32 (scratchpad columns)
    int m;         // 64 or 224 (memory columns)
    int N;         // 8 (value bits)
    int logn;      // 10 or 11
    float lam;     // 10.0
    int idx_pc;    // row index where PC bits start
    int idx_memory; // row index where memory bits start
};

// ---- Sparse COO entry ----
struct COOEntry {
    int16_t row, col;
    float val;
};

// ---- Dense bias: full rows × cols matrix ----
struct BiasDense {
    int rows;
    int cols;
    std::vector<float> data;  // [rows × cols], row-major
};

// ---- Attention head ----
struct AttentionHead {
    int q_rows;
    std::vector<COOEntry> Q, K, V;
};

// ---- One transformer layer ----
struct Layer {
    int num_heads;
    int q_rows;
    int w1_rows;
    std::vector<AttentionHead> heads;

    // FFN
    std::vector<COOEntry> W1;
    BiasDense b1;
    std::vector<COOEntry> W2;
    BiasDense b2;
};

// ---- Full model ----
struct Model {
    Config cfg;
    std::vector<Layer> layers;
};


/* ============================================================
 * Binary file I/O
 * ============================================================ */

static inline void read_exact(FILE* f, void* buf, size_t n) {
    if (fread(buf, 1, n, f) != n) {
        fprintf(stderr, "Error: unexpected end of file\n");
        exit(1);
    }
}

static inline int32_t read_i32(FILE* f) {
    int32_t v; read_exact(f, &v, 4); return v;
}

static inline float read_f32(FILE* f) {
    float v; read_exact(f, &v, 4); return v;
}

static std::vector<COOEntry> read_sparse_coo(FILE* f) {
    int32_t nnz = read_i32(f);
    std::vector<COOEntry> entries(nnz);
    for (int i = 0; i < nnz; i++) {
        read_exact(f, &entries[i].row, 2);
        read_exact(f, &entries[i].col, 2);
        read_exact(f, &entries[i].val, 4);
    }
    return entries;
}

static BiasDense read_bias_dense(FILE* f) {
    BiasDense b;
    b.rows = read_i32(f);
    b.cols = read_i32(f);
    b.data.resize(b.rows * b.cols);
    read_exact(f, b.data.data(), b.rows * b.cols * sizeof(float));
    return b;
}

static Config load_config(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    Config cfg;
    cfg.d_model = read_i32(f);
    cfg.n = read_i32(f);
    cfg.s = read_i32(f);
    cfg.m = read_i32(f);
    cfg.N = read_i32(f);
    cfg.logn = read_i32(f);
    cfg.lam = read_f32(f);
    // Try reading extended fields (idx_pc, idx_memory) — fall back to computed values
    if (fread(&cfg.idx_pc, 4, 1, f) == 1) {
        cfg.idx_memory = read_i32(f);
    } else {
        // Legacy config without idx_pc: compute from formula
        int nrows_cmds = 3 * cfg.logn;
        int nrows_memory = cfg.N;
        int nrows_scratchpad = 3 * cfg.logn + 2 * cfg.N;
        cfg.idx_memory = nrows_cmds;
        cfg.idx_pc = nrows_cmds + nrows_memory + nrows_scratchpad;
    }
    fclose(f);
    return cfg;
}

static Model load_model(const char* config_path, const char* weights_path) {
    Model model;
    model.cfg = load_config(config_path);

    FILE* f = fopen(weights_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", weights_path); exit(1); }

    int num_layers = read_i32(f);
    model.layers.resize(num_layers);

    for (int li = 0; li < num_layers; li++) {
        Layer& layer = model.layers[li];
        layer.num_heads = read_i32(f);
        layer.q_rows = read_i32(f);
        layer.w1_rows = read_i32(f);

        layer.heads.resize(layer.num_heads);
        for (int h = 0; h < layer.num_heads; h++) {
            layer.heads[h].q_rows = layer.q_rows;
            layer.heads[h].Q = read_sparse_coo(f);
            layer.heads[h].K = read_sparse_coo(f);
            layer.heads[h].V = read_sparse_coo(f);
        }

        layer.W1 = read_sparse_coo(f);
        layer.b1 = read_bias_dense(f);
        layer.W2 = read_sparse_coo(f);
        layer.b2 = read_bias_dense(f);
    }

    fclose(f);
    return model;
}

static std::vector<float> load_state(const char* path, int* d_out, int* n_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    *d_out = read_i32(f);
    *n_out = read_i32(f);
    int total = (*d_out) * (*n_out);
    std::vector<float> data(total);
    read_exact(f, data.data(), total * sizeof(float));
    fclose(f);
    return data;
}


/* ============================================================
 * Core computation
 *
 * State X is row-major: X[row * n + col]
 * ============================================================ */

// Sparse matrix × dense matrix column: out[row] += val * X[col * stride + j]
// Used for Q@X, K@X, V@X, W1@X, W2@ff
static void sparse_matvec_add(
    const std::vector<COOEntry>& sparse,
    const float* X, int n,       // X is [rows_X × n], row-major
    float* out, int out_stride,  // out is [rows_out × out_stride]
    int col_j                    // which column of X to process
) {
    for (const auto& e : sparse) {
        out[e.row * out_stride + col_j] += e.val * X[e.col * n + col_j];
    }
}

// Apply dense bias: out[r][j] += bias.data[r * cols + j]
static void add_bias_dense(
    float* out, int rows, int n,
    const BiasDense& bias
) {
    for (int r = 0; r < rows; r++) {
        for (int j = 0; j < n; j++) {
            out[r * n + j] += bias.data[r * bias.cols + j];
        }
    }
}

// Compute attention weights using actual softmax (matches Python exactly).
// scores[i][j] = dot(KX[:, i], QX[:, j])
// attn_weights = softmax(lam * scores, dim=0)  (column-wise softmax, standard convention)
// Each column j sums to 1: column j selects which source row i to read from.
// Then VX_acc[:, j] += sum_{i} V @ X[:, i] * attn_weights[i, j]
static void compute_attention_weights(
    const float* QX, const float* KX,  // both [q_rows × n]
    int q_rows, int n, float lam,
    float* attn  // [n × n] output, row-major
) {
    // Compute scores
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float score = 0.0f;
            for (int q = 0; q < q_rows; q++) {
                score += KX[q * n + i] * QX[q * n + j];
            }
            attn[i * n + j] = lam * score;
        }
    }
    // Column-wise softmax (dim=0): normalize each column j
    for (int j = 0; j < n; j++) {
        float col_max = -1e30f;
        for (int i = 0; i < n; i++) {
            if (attn[i * n + j] > col_max) col_max = attn[i * n + j];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < n; i++) {
            attn[i * n + j] = expf(attn[i * n + j] - col_max);
            sum_exp += attn[i * n + j];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < n; i++) {
            attn[i * n + j] *= inv_sum;
        }
    }
}

// Argmax attention: streaming top-2 per source column, no n×n matrix.
// For each source column i, find the top-2 target columns j that maximize
// score[i,j] = KX[:,i] · QX[:,j], then distribute VX[:,i] to those targets.
// With Q=K (symmetric scores), this is equivalent to column-wise argmax (dim=0).
static void argmax_attention(
    const float* QX, const float* KX,  // [q_rows × n]
    int q_rows, int n,
    const std::vector<COOEntry>& V,
    const float* X,     // [d × n]
    float* VX_acc       // [d × n]
) {
    const float TIE_THRESH = 1.0f;

    for (int i = 0; i < n; i++) {
        // Check if KX column i is zero
        bool is_zero = true;
        for (int q = 0; q < q_rows; q++) {
            if (KX[q * n + i] != 0.0f) { is_zero = false; break; }
        }
        if (is_zero) continue;

        // Find top-2
        float best1_v = -1e30f, best2_v = -1e30f;
        int best1_j = 0, best2_j = 0;
        for (int j = 0; j < n; j++) {
            float score = 0.0f;
            for (int q = 0; q < q_rows; q++) {
                score += KX[q * n + i] * QX[q * n + j];
            }
            if (score > best1_v) {
                best2_v = best1_v; best2_j = best1_j;
                best1_v = score; best1_j = j;
            } else if (score > best2_v) {
                best2_v = score; best2_j = j;
            }
        }

        bool tied = (best1_v - best2_v) < TIE_THRESH;
        float w1 = tied ? 0.5f : 1.0f;

        for (const auto& e : V) {
            float vx = e.val * X[e.col * n + i];
            VX_acc[e.row * n + best1_j] += w1 * vx;
            if (tied) {
                VX_acc[e.row * n + best2_j] += 0.5f * vx;
            }
        }
    }
}

// Transformer step using argmax attention (no n×n matrix)
static void transformer_step_argmax(const Model& model, float* X) {
    const Config& cfg = model.cfg;
    const int d = cfg.d_model;
    const int n = cfg.n;

    std::vector<float> QX, KX, VX_acc, ff_hidden;
    int max_q_rows = 0, max_w1_rows = 0;
    for (const auto& layer : model.layers) {
        if (layer.q_rows > max_q_rows) max_q_rows = layer.q_rows;
        if (layer.w1_rows > max_w1_rows) max_w1_rows = layer.w1_rows;
    }
    QX.resize(max_q_rows * n);
    KX.resize(max_q_rows * n);
    VX_acc.resize(d * n);
    ff_hidden.resize(max_w1_rows * n);

    for (int li = 0; li < (int)model.layers.size(); li++) {
        const Layer& layer = model.layers[li];
        int q_rows = layer.q_rows;

        memset(VX_acc.data(), 0, d * n * sizeof(float));

        for (int h = 0; h < layer.num_heads; h++) {
            const AttentionHead& head = layer.heads[h];
            // Skip no-op heads
            if (head.Q.empty() && head.K.empty() && head.V.empty()) continue;

            memset(QX.data(), 0, q_rows * n * sizeof(float));
            for (int j = 0; j < n; j++)
                sparse_matvec_add(head.Q, X, n, QX.data(), n, j);

            memset(KX.data(), 0, q_rows * n * sizeof(float));
            for (int j = 0; j < n; j++)
                sparse_matvec_add(head.K, X, n, KX.data(), n, j);

            argmax_attention(QX.data(), KX.data(), q_rows, n, head.V, X, VX_acc.data());
        }

        for (int i = 0; i < d * n; i++) X[i] += VX_acc[i];

        // FFN
        memset(ff_hidden.data(), 0, layer.w1_rows * n * sizeof(float));
        for (int j = 0; j < n; j++)
            sparse_matvec_add(layer.W1, X, n, ff_hidden.data(), n, j);
        add_bias_dense(ff_hidden.data(), layer.w1_rows, n, layer.b1);
        for (int i = 0; i < layer.w1_rows * n; i++)
            if (ff_hidden[i] < 0.0f) ff_hidden[i] = 0.0f;
        for (int j = 0; j < n; j++)
            sparse_matvec_add(layer.W2, ff_hidden.data(), n, X, n, j);
        add_bias_dense(X, d, n, layer.b2);
    }
}

// One full transformer step using softmax: process all layers sequentially
static void transformer_step(const Model& model, float* X) {
    const Config& cfg = model.cfg;
    const int d = cfg.d_model;
    const int n = cfg.n;

    // Scratch buffers (allocated once, reused per layer)
    std::vector<float> QX, KX, VX_acc, ff_hidden, attn_weights;

    int max_q_rows = 0, max_w1_rows = 0;
    for (const auto& layer : model.layers) {
        if (layer.q_rows > max_q_rows) max_q_rows = layer.q_rows;
        if (layer.w1_rows > max_w1_rows) max_w1_rows = layer.w1_rows;
    }
    QX.resize(max_q_rows * n);
    KX.resize(max_q_rows * n);
    VX_acc.resize(d * n);
    ff_hidden.resize(max_w1_rows * n);
    attn_weights.resize((size_t)n * n);

    for (int li = 0; li < (int)model.layers.size(); li++) {
        const Layer& layer = model.layers[li];
        int q_rows = layer.q_rows;

        // ---- Attention ----
        // attn = X + sum_h(V_h @ X @ softmax(lam * X^T K_h^T Q_h X, dim=0))
        memset(VX_acc.data(), 0, d * n * sizeof(float));

        for (int h = 0; h < layer.num_heads; h++) {
            const AttentionHead& head = layer.heads[h];

            // QX = Q @ X  (q_rows × n, sparse)
            memset(QX.data(), 0, q_rows * n * sizeof(float));
            for (int j = 0; j < n; j++) {
                sparse_matvec_add(head.Q, X, n, QX.data(), n, j);
            }

            // KX = K @ X  (q_rows × n, sparse)
            memset(KX.data(), 0, q_rows * n * sizeof(float));
            for (int j = 0; j < n; j++) {
                sparse_matvec_add(head.K, X, n, KX.data(), n, j);
            }

            // Compute softmax attention weights (n × n)
            compute_attention_weights(QX.data(), KX.data(), q_rows, n,
                                      cfg.lam, attn_weights.data());

            // VX_acc[:, j] += sum_i V @ X[:, i] * attn_weights[i, j]
            // = V @ (X @ attn_weights)[:, j]
            for (int j = 0; j < n; j++) {
                for (const auto& e : head.V) {
                    float sum = 0.0f;
                    for (int i = 0; i < n; i++) {
                        sum += attn_weights[i * n + j] * X[e.col * n + i];
                    }
                    VX_acc[e.row * n + j] += e.val * sum;
                }
            }
        }

        // X = X + VX_acc (residual connection)
        for (int i = 0; i < d * n; i++) X[i] += VX_acc[i];

        // ---- FFN ----
        // ff_hidden = ReLU(W1 @ X + b1)
        memset(ff_hidden.data(), 0, layer.w1_rows * n * sizeof(float));
        for (int j = 0; j < n; j++) {
            sparse_matvec_add(layer.W1, X, n, ff_hidden.data(), n, j);
        }
        add_bias_dense(ff_hidden.data(), layer.w1_rows, n, layer.b1);

        // ReLU
        for (int i = 0; i < layer.w1_rows * n; i++) {
            if (ff_hidden[i] < 0.0f) ff_hidden[i] = 0.0f;
        }

        // X += W2 @ ff_hidden + b2
        for (int j = 0; j < n; j++) {
            sparse_matvec_add(layer.W2, ff_hidden.data(), n, X, n, j);
        }
        add_bias_dense(X, d, n, layer.b2);
    }
}

#endif // TRANSFORMER_H
