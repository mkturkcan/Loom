/*
 * Multi-step transformer benchmark.
 *
 * Runs the V4 transformer for N steps and reports timing.
 * Used to measure throughput (steps/second) for the paper.
 *
 * Build:  g++ -O2 -o bench src/bench.cpp -lm
 * Run:    ./bench data/ 1000
 */

#include "transformer.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

// Extract PC value from state (bipolar encoding)
static int get_pc(const float* X, const Config& cfg, int idx_pc) {
    int pc = 0;
    for (int b = 0; b < cfg.logn; b++) {
        int bit = (X[( idx_pc + b) * cfg.n + 0] > 0.0f) ? 1 : 0;
        pc |= (bit << (cfg.logn - 1 - b));
    }
    return pc;
}

int main(int argc, char* argv[]) {
    const char* data_dir = (argc > 1) ? argv[1] : "data";
    int max_steps = (argc > 2) ? atoi(argv[2]) : 1000;
    int print_every = (argc > 3) ? atoi(argv[3]) : 100;

    std::string config_path  = std::string(data_dir) + "/config.bin";
    std::string weights_path = std::string(data_dir) + "/weights.bin";
    std::string input_path   = std::string(data_dir) + "/test_input.bin";

    // ---- Load model ----
    printf("Loading model from %s ...\n", data_dir);
    Model model = load_model(config_path.c_str(), weights_path.c_str());
    const Config& cfg = model.cfg;
    printf("  Config: d_model=%d, n=%d, lam=%.1f, idx_pc=%d\n",
           cfg.d_model, cfg.n, cfg.lam, cfg.idx_pc);

    int idx_pc = cfg.idx_pc;

    // ---- Load input state ----
    int d_in, n_in;
    std::vector<float> X = load_state(input_path.c_str(), &d_in, &n_in);
    printf("  Input: %d x %d\n", d_in, n_in);

    int total = cfg.d_model * cfg.n;

    // ---- Keep a copy for halt detection ----
    std::vector<float> X_prev(total);

    // ---- Run transformer steps ----
    printf("\nRunning %d steps...\n", max_steps);
    int pc0 = get_pc(X.data(), cfg, idx_pc);
    printf("  Initial PC: %d\n", pc0);

    auto t_start = std::chrono::high_resolution_clock::now();

    int step;
    bool halted = false;
    for (step = 0; step < max_steps; step++) {
        // Save state for halt detection
        memcpy(X_prev.data(), X.data(), total * sizeof(float));

        // Run one transformer step (argmax attention — no n×n matrix)
        transformer_step_argmax(model, X.data());

        int pc = get_pc(X.data(), cfg, idx_pc);

        if (step < 5 || (print_every > 0 && (step + 1) % print_every == 0)) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double rate = (step + 1) / elapsed;
            printf("  Step %d: PC=%d (%.1f steps/sec)\n", step + 1, pc, rate);
        }

        // Check for halt: state unchanged
        double max_diff = 0.0;
        for (int i = 0; i < total; i++) {
            double d = fabs((double)X[i] - (double)X_prev[i]);
            if (d > max_diff) max_diff = d;
        }
        if (max_diff < 1e-4) {
            printf("  HALTED at step %d (max_diff=%.6f)\n", step + 1, max_diff);
            halted = true;
            step++;
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // ---- Report ----
    printf("\n========================================\n");
    printf("Benchmark results:\n");
    printf("  Steps:        %d%s\n", step, halted ? " (halted)" : "");
    printf("  Total time:   %.3f seconds\n", elapsed);
    printf("  Steps/sec:    %.1f\n", step / elapsed);
    printf("  ms/step:      %.3f\n", elapsed * 1000.0 / step);

    int pc_final = get_pc(X.data(), cfg, idx_pc);
    printf("  Final PC:     %d\n", pc_final);
    printf("========================================\n");

    // ---- Print some memory values for verification ----
    printf("\nMemory dump (first 32 values):\n  ");
    int idx_mem = cfg.idx_memory;
    for (int i = 0; i < 32 && i < cfg.m; i++) {
        int mem_col = cfg.s + i;  // memory columns start after scratchpad
        // Decode signed bipolar: N value bits at idx_memory rows, MSB first
        int magnitude = 0;
        for (int b = 0; b < cfg.N; b++) {
            int bit = (X[(idx_mem + b) * cfg.n + mem_col] > 0.0f) ? 1 : 0;
            magnitude |= (bit << (cfg.N - 1 - b));
        }
        // Sign: if highest bit is set, it's negative (two's complement style)
        int val = magnitude;
        if (magnitude >= (1 << (cfg.N - 1)))
            val = magnitude - (1 << cfg.N);
        printf("%d ", val);
    }
    printf("\n");

    return halted ? 0 : 1;
}
