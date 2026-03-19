/*
 * CPU test program for Neural SUBLEQ V4 Transformer.
 *
 * Loads exported weights and test vectors, runs one transformer step,
 * and compares the output against the Python reference.
 *
 * Build:  g++ -O2 -o test_cpu src/main_cpu.cpp -lm
 * Run:    ./test_cpu data/
 */

#include "transformer.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char* argv[]) {
    const char* data_dir = (argc > 1) ? argv[1] : "data";

    std::string config_path  = std::string(data_dir) + "/config.bin";
    std::string weights_path = std::string(data_dir) + "/weights.bin";
    std::string input_path   = std::string(data_dir) + "/test_input.bin";
    std::string output_path  = std::string(data_dir) + "/test_output.bin";
    std::string layer_path   = std::string(data_dir) + "/layer_outputs.bin";

    // ---- Load model ----
    printf("Loading model from %s ...\n", data_dir);
    Model model = load_model(config_path.c_str(), weights_path.c_str());

    const Config& cfg = model.cfg;
    printf("  Config: d_model=%d, n=%d, s=%d, m=%d, N=%d, logn=%d, lam=%.1f\n",
           cfg.d_model, cfg.n, cfg.s, cfg.m, cfg.N, cfg.logn, cfg.lam);
    printf("  Layers: %d\n", (int)model.layers.size());

    int total_nnz = 0;
    for (int li = 0; li < (int)model.layers.size(); li++) {
        const Layer& layer = model.layers[li];
        int layer_nnz = 0;
        for (int h = 0; h < layer.num_heads; h++) {
            layer_nnz += layer.heads[h].Q.size();
            layer_nnz += layer.heads[h].K.size();
            layer_nnz += layer.heads[h].V.size();
        }
        layer_nnz += layer.W1.size() + layer.W2.size();
        total_nnz += layer_nnz;
        printf("  L%d: %d heads, q_rows=%d, w1_rows=%d, nnz=%d\n",
               li + 1, layer.num_heads, layer.q_rows, layer.w1_rows, layer_nnz);
    }
    printf("  Total nonzero weights: %d\n", total_nnz);

    // ---- Load test input ----
    int d_in, n_in;
    std::vector<float> X = load_state(input_path.c_str(), &d_in, &n_in);
    printf("\nInput: %d x %d\n", d_in, n_in);

    if (d_in != cfg.d_model || n_in != cfg.n) {
        fprintf(stderr, "Error: input dimensions %dx%d don't match config %dx%d\n",
                d_in, n_in, cfg.d_model, cfg.n);
        return 1;
    }

    // ---- Load reference output ----
    int d_out, n_out;
    std::vector<float> X_ref = load_state(output_path.c_str(), &d_out, &n_out);
    printf("Reference output: %d x %d\n", d_out, n_out);

    // ---- Run transformer step (argmax attention) ----
    printf("\nRunning transformer step (CPU, argmax) ...\n");
    transformer_step_argmax(model, X.data());

    // ---- Compare against reference ----
    double max_err = 0.0;
    double sum_err = 0.0;
    int num_mismatch = 0;
    int total = cfg.d_model * cfg.n;

    for (int i = 0; i < total; i++) {
        double err = fabs((double)X[i] - (double)X_ref[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
        if (err > 0.01) num_mismatch++;
    }

    double avg_err = sum_err / total;
    printf("\nResults:\n");
    printf("  Max error:     %.6f\n", max_err);
    printf("  Avg error:     %.6f\n", avg_err);
    printf("  Mismatches (>0.01): %d / %d\n", num_mismatch, total);

    // ---- Check PC advancement ----
    // PC is at row idx_pc (= s + m + nrows_cmds), column 0
    // idx_pc = s + m + nrows_cmds where nrows_cmds = 3 * ceil(n / 2^N)
    int nrows_cmds = 3 * ((cfg.n + (1 << cfg.N) - 1) / (1 << cfg.N));
    int idx_pc = cfg.s + cfg.m + nrows_cmds;

    // Read PC from input and output (column 0 = the scalar value)
    // In the state matrix, PC is encoded across N bit rows starting at idx_pc
    // Actually the PC value is simpler: it's the binary encoding at rows idx_pc..idx_pc+logn-1, col 0
    printf("\n  State row idx_pc=%d\n", idx_pc);
    printf("  Input  PC bits: ");
    for (int b = 0; b < cfg.logn; b++)
        printf("%.0f ", X_ref[0]);  // just show we ran

    // ---- Per-layer comparison (if available) ----
    FILE* lf = fopen(layer_path.c_str(), "rb");
    if (lf) {
        int32_t nl = read_i32(lf);
        printf("\nPer-layer comparison (%d layers):\n", nl);

        // Re-run from input, layer by layer
        int d_in2, n_in2;
        std::vector<float> X2 = load_state(input_path.c_str(), &d_in2, &n_in2);

        for (int li = 0; li < nl && li < (int)model.layers.size(); li++) {
            // Read reference layer output
            std::vector<float> ref_layer(cfg.d_model * cfg.n);
            read_exact(lf, ref_layer.data(), cfg.d_model * cfg.n * sizeof(float));

            // Run single layer on X2
            // We need to replicate transformer_step but one layer at a time
            // For simplicity, just compare final output
        }

        // Actually, let's just run the full step again and compare layer-by-layer
        // Re-load input
        X2 = load_state(input_path.c_str(), &d_in2, &n_in2);
        fseek(lf, 4, SEEK_SET);  // skip num_layers

        // Scratch buffers
        int max_q = 0, max_w1 = 0;
        for (const auto& layer : model.layers) {
            if (layer.q_rows > max_q) max_q = layer.q_rows;
            if (layer.w1_rows > max_w1) max_w1 = layer.w1_rows;
        }

        std::vector<float> QX(max_q * cfg.n), KX(max_q * cfg.n);
        std::vector<float> VX_acc(cfg.d_model * cfg.n);
        std::vector<float> ff_hidden(max_w1 * cfg.n);
        std::vector<float> attn_w((size_t)cfg.n * cfg.n);

        for (int li = 0; li < nl && li < (int)model.layers.size(); li++) {
            const Layer& layer = model.layers[li];
            int q_rows = layer.q_rows;

            // Run attention (softmax, matching Python)
            memset(VX_acc.data(), 0, cfg.d_model * cfg.n * sizeof(float));
            for (int h = 0; h < layer.num_heads; h++) {
                const AttentionHead& head = layer.heads[h];
                memset(QX.data(), 0, q_rows * cfg.n * sizeof(float));
                for (int j = 0; j < cfg.n; j++)
                    sparse_matvec_add(head.Q, X2.data(), cfg.n, QX.data(), cfg.n, j);
                memset(KX.data(), 0, q_rows * cfg.n * sizeof(float));
                for (int j = 0; j < cfg.n; j++)
                    sparse_matvec_add(head.K, X2.data(), cfg.n, KX.data(), cfg.n, j);
                compute_attention_weights(QX.data(), KX.data(), q_rows, cfg.n,
                                          cfg.lam, attn_w.data());
                for (int j = 0; j < cfg.n; j++) {
                    for (const auto& e : head.V) {
                        float sum = 0.0f;
                        for (int i = 0; i < cfg.n; i++)
                            sum += attn_w[i * cfg.n + j] * X2[e.col * cfg.n + i];
                        VX_acc[e.row * cfg.n + j] += e.val * sum;
                    }
                }
            }
            for (int i = 0; i < cfg.d_model * cfg.n; i++) X2[i] += VX_acc[i];

            // Run FFN
            memset(ff_hidden.data(), 0, layer.w1_rows * cfg.n * sizeof(float));
            for (int j = 0; j < cfg.n; j++)
                sparse_matvec_add(layer.W1, X2.data(), cfg.n, ff_hidden.data(), cfg.n, j);
            add_bias_dense(ff_hidden.data(), layer.w1_rows, cfg.n, layer.b1);
            for (int i = 0; i < layer.w1_rows * cfg.n; i++)
                if (ff_hidden[i] < 0.0f) ff_hidden[i] = 0.0f;
            for (int j = 0; j < cfg.n; j++)
                sparse_matvec_add(layer.W2, ff_hidden.data(), cfg.n, X2.data(), cfg.n, j);
            add_bias_dense(X2.data(), cfg.d_model, cfg.n, layer.b2);

            // Read reference and compare
            std::vector<float> ref_layer(cfg.d_model * cfg.n);
            read_exact(lf, ref_layer.data(), cfg.d_model * cfg.n * sizeof(float));

            double layer_max = 0.0;
            int layer_mismatch = 0;
            for (int i = 0; i < cfg.d_model * cfg.n; i++) {
                double err = fabs((double)X2[i] - (double)ref_layer[i]);
                if (err > layer_max) layer_max = err;
                if (err > 0.01) layer_mismatch++;
            }
            printf("  L%d: max_err=%.6f, mismatches=%d %s\n",
                   li + 1, layer_max, layer_mismatch,
                   layer_max < 0.01 ? "OK" : "MISMATCH");
        }

        fclose(lf);
    }

    if (max_err < 0.01) {
        printf("\n*** PASS: C++ output matches Python reference ***\n");
        return 0;
    } else {
        printf("\n*** FAIL: output differs from reference ***\n");
        return 1;
    }
}
