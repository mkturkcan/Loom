/*
 * OpenCL host program for Neural SUBLEQ V4 Transformer FPGA kernel.
 *
 * v2: n=1024, argmax attention, halt detection.
 *
 * Test plan:
 *   1. Single-step INC:  ./host <xclbin> data_v2 1
 *      Expected: PC 192->193, mem[0] 5->6
 *   2. Multi-step halt:  ./host <xclbin> data_v2 50
 *      Expected: halts at step 2 (INC then HALT)
 *   3. Benchmark:        ./host <xclbin> data_v2 1000000
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include "transformer.h"


static std::vector<float> load_flat_weights(const char* path, int* total_floats) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *total_floats = (int)(size / sizeof(float));
    std::vector<float> buf(*total_floats);
    read_exact(f, buf.data(), size);
    fclose(f);
    return buf;
}

static cl::Device find_xilinx_device() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto& platform : platforms) {
        std::string name = platform.getInfo<CL_PLATFORM_NAME>();
        if (name.find("Xilinx") != std::string::npos) {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (!devices.empty()) {
                printf("Found Xilinx device: %s\n",
                       devices[0].getInfo<CL_DEVICE_NAME>().c_str());
                return devices[0];
            }
        }
    }
    fprintf(stderr, "Error: no Xilinx FPGA device found\n");
    exit(1);
}

static std::vector<unsigned char> load_xclbin(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { fprintf(stderr, "Cannot open xclbin: %s\n", path); exit(1); }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<unsigned char> buf(size);
    file.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

static int get_pc(const float* X, const Config& cfg) {
    int pc = 0;
    for (int b = 0; b < cfg.logn; b++) {
        int bit = (X[(cfg.idx_pc + b) * cfg.n + 0] > 0.0f) ? 1 : 0;
        pc |= (bit << (cfg.logn - 1 - b));
    }
    return pc;
}

static int get_mem(const float* X, const Config& cfg, int i) {
    int mem_col = cfg.s + i;
    int magnitude = 0;
    for (int b = 0; b < cfg.N; b++) {
        int bit = (X[(cfg.idx_memory + b) * cfg.n + mem_col] > 0.0f) ? 1 : 0;
        magnitude |= (bit << (cfg.N - 1 - b));
    }
    if (magnitude >= (1 << (cfg.N - 1)))
        magnitude -= (1 << cfg.N);
    return magnitude;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <xclbin_path> [data_dir] [num_steps]\n", argv[0]);
        return 1;
    }

    const char* xclbin_path = argv[1];
    const char* data_dir = (argc > 2) ? argv[2] : "data_v2";
    int num_steps = (argc > 3) ? atoi(argv[3]) : 1;

    std::string config_path  = std::string(data_dir) + "/config.bin";
    std::string weights_path = std::string(data_dir) + "/weights_flat.bin";
    std::string input_path   = std::string(data_dir) + "/test_input.bin";

    // ---- Load config ----
    Config cfg = load_config(config_path.c_str());
    printf("Config: d_model=%d, n=%d, s=%d, m=%d, lam=%.1f\n",
           cfg.d_model, cfg.n, cfg.s, cfg.m, cfg.lam);
    printf("  idx_pc=%d, idx_memory=%d, logn=%d\n", cfg.idx_pc, cfg.idx_memory, cfg.logn);

    int state_size = cfg.d_model * cfg.n;

    // ---- Load weights ----
    int weights_total;
    std::vector<float> weights = load_flat_weights(weights_path.c_str(), &weights_total);
    printf("Weights: %d floats (%.1f KB)\n", weights_total, weights_total * 4.0f / 1024);

    // ---- Load test input ----
    int d_in, n_in;
    std::vector<float> X = load_state(input_path.c_str(), &d_in, &n_in);
    printf("Input: %d x %d\n", d_in, n_in);

    if (d_in != cfg.d_model || n_in != cfg.n) {
        fprintf(stderr, "Error: input %dx%d != config %dx%d\n", d_in, n_in, cfg.d_model, cfg.n);
        return 1;
    }

    int pc0 = get_pc(X.data(), cfg);
    int mem0_before = get_mem(X.data(), cfg, 0);
    printf("Initial: PC=%d, mem[0]=%d\n", pc0, mem0_before);

    // ---- Depth-matched buffer allocation ----
    // Buffers must be at least as large as kernel's HLS depth pragmas
    const int W_DEPTH = 131072;   // matches kernel depth pragma for weights_buf
    const int S_DEPTH = 158720;   // matches kernel depth pragma for state_buf

    std::vector<float> w_big(W_DEPTH, 0.0f);
    std::copy(weights.begin(), weights.end(), w_big.begin());

    std::vector<float> s_big(S_DEPTH, 0.0f);
    std::copy(X.begin(), X.end(), s_big.begin());

    // ---- Setup OpenCL ----
    printf("\nSetting up FPGA...\n");

    cl::Device device = find_xilinx_device();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::vector<unsigned char> xclbin = load_xclbin(xclbin_path);
    cl::Program::Binaries binaries;
    binaries.push_back(xclbin);
    std::vector<cl::Device> devices{device};

    cl_int err;
    cl::Program program(context, devices, binaries, nullptr, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error programming FPGA: %d\n", err); return 1; }

    cl::Kernel kernel(program, "transformer_kernel", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error creating kernel: %d\n", err); return 1; }

    // ---- Allocate device buffers ----
    cl::Buffer weights_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           W_DEPTH * sizeof(float), w_big.data(), &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error alloc weights: %d\n", err); return 1; }

    cl::Buffer state_buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         S_DEPTH * sizeof(float), s_big.data(), &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "Error alloc state: %d\n", err); return 1; }

    // ---- Set kernel arguments ----
    int n_layers = 8;
    kernel.setArg(0, weights_buf);
    kernel.setArg(1, state_buf);
    kernel.setArg(2, num_steps);
    kernel.setArg(3, weights_total);
    kernel.setArg(4, cfg.d_model);
    kernel.setArg(5, cfg.n);
    kernel.setArg(6, n_layers);
    kernel.setArg(7, cfg.lam);

    // ---- Execute ----
    printf("\nRunning %d transformer step(s) on FPGA...\n", num_steps);
    auto t_start = std::chrono::high_resolution_clock::now();

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
    queue.finish();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // ---- Read back ----
    std::vector<float> X_out(state_size);
    queue.enqueueReadBuffer(state_buf, CL_TRUE, 0,
                            state_size * sizeof(float), X_out.data());

    int pc_final = get_pc(X_out.data(), cfg);
    int mem0_after = get_mem(X_out.data(), cfg, 0);

    // ---- Report ----
    printf("\n========================================\n");
    printf("FPGA Results:\n");
    printf("  Requested steps: %d\n", num_steps);
    printf("  Total time:      %.3f seconds\n", elapsed);
    if (elapsed > 0) {
        printf("  Steps/sec:       %.1f\n", num_steps / elapsed);
        printf("  ms/step:         %.3f\n", elapsed * 1000.0 / num_steps);
    }
    printf("  PC:  %d -> %d\n", pc0, pc_final);
    printf("  mem[0]: %d -> %d\n", mem0_before, mem0_after);
    printf("========================================\n");

    // ---- Verification (for test_input = INC 5) ----
    if (num_steps == 1 && mem0_before == 5) {
        bool pass = (mem0_after == 6 && pc_final == pc0 + 1);
        printf("\n*** INC TEST: %s ***\n", pass ? "PASS" : "FAIL");
        if (!pass) {
            printf("  Expected: mem[0]=6, PC=%d\n", pc0 + 1);
            printf("  Got:      mem[0]=%d, PC=%d\n", mem0_after, pc_final);
        }
    }

    // ---- Memory dump ----
    printf("\nMemory[0..31]: ");
    for (int i = 0; i < 32 && i < cfg.m; i++) {
        printf("%d ", get_mem(X_out.data(), cfg, i));
    }
    printf("\n");

    return 0;
}
