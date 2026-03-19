# Neural SUBLEQ V4 Transformer — FPGA Implementation

Run the V4 transformer on a Xilinx Alveo U200 FPGA.

## Architecture (v2 — March 2025)

- **8-layer transformer**, d_model=155, n=1024 (standard V4, no merges)
- **Argmax attention**: no n×n matrix, no exp(), streaming top-2 + tie detection
- **STORE opcode**: indirect write enables n=1024 (sudoku compiles to 284 instructions)
- **8,624 nonzero weights** (99.9% sparse) — all multiples of 0.25
- **Program-independent**: weights are universal, any compiled program runs without recompilation
- On-chip: ~4 MB (was 40 MB with softmax) — fits easily in U200's 44 MB

### Key results
- **Hardware synthesis**: 1h47m (was 7h+ OOM with n=2048 softmax)
- **INC TEST: PASS** on real FPGA hardware (PC: 192→193, mem[0]: 5→6)
- **sw_emu verified**: all operations correct including STORE + halt detection

## FPGA Machine Setup (Ubuntu 22.04, Alveo U200)

### Prerequisites
- XRT 2.14 (Xilinx Runtime)
- Vitis 2022.2
- Alveo U200 shell: xilinx_u200_gen3x16_xdma_base_2

See detailed install instructions in `NEXT_STEPS.md` or the Xilinx documentation.

### IMPORTANT: Verify card health before any hardware test

```bash
source /opt/xilinx/xrt/setup.sh
xbutil validate -d 0000:ca:00.1 -r verify
```

If this fails with "Bus error":
1. Cold reboot: `sudo ipmitool power cycle` (warm reboot is NOT enough)
2. If still failing: reflash shell with `xbmgmt program --base --device 0000:ca:00.0`
3. Cold reboot again (mandatory after flash)
4. May need TWO flash+reboot cycles from GOLDEN state

**Do NOT proceed with hardware tests until `xbutil validate` shows PASSED.**

## Quick Start

### Step 1: Transfer files to the FPGA machine

```bash
scp -r fpga/ user@fpga-machine:~/neural_subleq_fpga/
```

### Step 2: Export weights (on dev machine with PyTorch)

```bash
cd extended_isa
python fpga/export_weights_v2.py
```

This generates `data_v2/` with config, weights, and test vectors for n=1024.

### Step 3: Build and run CPU verification

```bash
cd ~/neural_subleq_fpga
make cpu DATA_DIR=data_v2
```

### Step 4: Software emulation

```bash
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/2022.2/settings64.sh

make sw_emu
XCL_EMULATION_MODE=sw_emu ./build/host build/transformer_kernel_sw_emu.xclbin data_v2 1
# Expected: PC: 192 -> 193, mem[0]: 5 -> 6, INC TEST: PASS
```

### Step 5: Hardware synthesis and run

```bash
# Verify card health first!
xbutil validate -d 0000:ca:00.1 -r verify

# Build (1-2 hours)
make hw DATA_DIR=data_v2

# Build host
make build/host

# Single step test
./build/host build/transformer_kernel_hw.xclbin data_v2 1
# Expected: PC: 192 -> 193, mem[0]: 5 -> 6

# Multi-step with halt detection
./build/host build/transformer_kernel_hw.xclbin data_v2 50

# Benchmark
./build/host build/transformer_kernel_hw.xclbin data_v2 100000
```

## File Structure

```
fpga/
├── data_v2/                   # v2 binary files (n=1024, STORE opcode)
│   ├── config.bin
│   ├── weights.bin            # Structured (for CPU test)
│   ├── weights_flat.bin       # Flat 137 KB (for FPGA kernel)
│   ├── test_input.bin         # INC 5 test state
│   └── test_output.bin        # Python reference output
├── src/
│   ├── transformer.h          # CPU implementation (softmax + argmax)
│   ├── main_cpu.cpp           # CPU test (argmax, compares vs Python)
│   ├── bench.cpp              # Multi-step benchmark (argmax)
│   ├── kernel.cpp             # Vitis HLS kernel (argmax, n=1024)
│   └── host.cpp               # OpenCL host program
├── export_weights_v2.py       # Export for v2 config
├── Makefile
├── README.md
└── NEXT_STEPS.md              # Detailed status + known issues
```

## Known Issues

- **BRAM vs URAM for weights**: `LayerWeights` struct with MAX_NNZ=5000 is ~4.5 MB.
  Must use `impl=uram` (not `impl=bram` which is only ~1 MB). Using BRAM causes
  silent corruption and Bus errors at runtime.
- **MAX_NNZ**: STORE opcode increased L2's W1 to ~4004 nonzeros. MAX_NNZ must be ≥5000.
- **Card health**: Alveo U200 can become unresponsive after failed runs. Always validate
  before hardware tests. Cold reboot (not warm) required to recover.
- **No printf in HLS**: `printf`/`fflush` work in sw_emu but cause synthesis failure
  for hardware builds. Remove before `make hw`.
- **First run overhead**: ~3 seconds for first kernel invocation (FPGA programming).
  Subsequent calls are faster.
- **No Ctrl+C during runs**: Killing the host while the kernel is running deadlocks the
  FPGA compute unit. Recovery requires `xbutil reset` or cold reboot. Let runs complete
  or set a reasonable `num_steps` limit.
- **ONNX FP16 breaks multi-step**: The V4 architecture relies on exact bipolar ±1 values
  and precise thresholds (6-threshold carry chain, eps=0.1 error correction). FP16's
  10-bit mantissa causes accumulated errors that flip bits after ~3-5 steps. INT8 also
  doesn't work via ONNX's generic quantization. FP32 is required for correctness.

## Performance Results (March 2026)

### Measured benchmarks

| Platform | Per-step | Notes |
|---|---|---|
| Python CPU (PyTorch) | ~1000 ms | Reference implementation |
| C++ CPU (argmax) | ~250 ms | `bench.cpp`, single-threaded |
| **FPGA (U200, sequential)** | **~3000 ms** | Sequential columns, no DSP parallelism |
| ONNX WASM (browser) | ~50 ms | `sudoku_merged.onnx`, 146×512 |
| ONNX WebGPU (RTX 4080) | ~10 ms | Same model, GPU-accelerated |

### Why the FPGA is 300x slower than the GPU

The current HLS kernel processes columns **sequentially**:
```
for each layer (8):
  for each head (6 active):
    for i = 0..1023:          // each key column
      for j = 0..1023:        // scan all query columns
        score += KX[q][i] * QX[q][j]   // 10 multiply-adds
```

This is 6 × 1024 × 1024 × 10 = **63M multiply-adds** for attention alone, done
one at a time at 300 MHz = ~0.2 seconds per layer × 8 layers = 1.6 seconds.
Plus sparse matvec and FFN adds ~1.4 seconds. Total: ~3 seconds/step.

The RTX 4080 does the same computation as **matrix multiplies** across 9,728 CUDA
cores simultaneously, completing in ~10 ms.

### Path to competitive FPGA performance

To match GPU speed, the kernel needs **column-parallel processing**:

1. **Array partitioning**: partition state X across P=64 processing elements
2. **Broadcast sparse weights**: each COO entry goes to all PEs simultaneously
3. **DSP utilization**: currently 0 DSPs used (all LUT-based float). INT16 weights
   × INT8 state fits in one DSP48E2 (27×18→48 bit MAC)
4. **Estimated P=64**: 64 DSPs per matvec, ~3 ms/step at 300 MHz
5. **With INT16 quantization**: weights fit INT16 (27 unique values, max ±256×4=1024),
   state fits INT8 (bipolar ±1). Would use 128 of 6,840 DSPs (2%).

### Value of current FPGA implementation

Despite being slow, the current implementation proves:
- ✅ Neural SUBLEQ computer runs on real silicon (correctness verified)
- ✅ Argmax attention works in hardware (no n×n matrix, no exp())
- ✅ STORE opcode functions correctly on FPGA
- ✅ Halt detection works (PC=0 exits early)
- ✅ Design synthesizes in ~2 hours and fits easily on U200 (~4 MB of 44 MB)
- ✅ Program-independent: change the state buffer to run any compiled program

### On-chip resource usage

```
URAM:  ~160 of 960 blocks  (state + weights + compute buffers)
BRAM:  ~224 of 2160 blocks
DSP:   0 of 6840 (!)       (floating-point in LUTs, not DSP slices)
LUT:   ~38K                 (floating-point arithmetic)
FF:    ~24K                 (pipeline registers)
Freq:  ~230 MHz estimated   (meets 300 MHz after P&R)
```

The 0 DSP usage and all-LUT floating point is the main performance issue.
A fixed-point INT16 kernel using DSP48E2 slices would be fundamentally faster.
