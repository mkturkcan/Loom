# Neural SUBLEQ V4 Transformer — FPGA Build System
#
# Targets:
#   cpu       — Build and run CPU verification test
#   export    — Export weights from Python (run on dev machine)
#   export_flat — Export flat weight buffer for FPGA kernel
#   sw_emu    — Software emulation (Vitis)
#   hw_emu    — Hardware emulation (Vitis, cycle-accurate)
#   hw        — Full hardware build (Vitis, several hours)
#   run_sw    — Run software emulation
#   run_hw_emu — Run hardware emulation
#   clean     — Remove build artifacts

CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall
DATA_DIR = data

# ---- Vitis settings ----
PLATFORM ?= xilinx_u200_gen3x16_xdma_2_202110_1
VPP      = v++
VPP_FLAGS = --platform $(PLATFORM) --save-temps

# Host compile flags (Xilinx OpenCL)
XILINX_XRT   ?= /opt/xilinx/xrt
HOST_CXX      = g++
HOST_CXXFLAGS = -O2 -std=c++17 -Wall \
                -I$(XILINX_XRT)/include \
                -I src/
HOST_LDFLAGS  = -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt

.PHONY: all cpu bench export export_flat sw_emu hw_emu hw run_sw run_hw_emu clean

all: cpu

# ===========================================================
# Phase 1 — CPU verification & benchmarking
# ===========================================================

build/test_cpu: src/main_cpu.cpp src/transformer.h | build
	$(CXX) $(CXXFLAGS) -o $@ src/main_cpu.cpp -lm

build/bench: src/bench.cpp src/transformer.h | build
	$(CXX) $(CXXFLAGS) -o $@ src/bench.cpp -lm

bench: build/bench
	@echo "========================================="
	@echo "Running benchmark..."
	@echo "========================================="
	./build/bench $(DATA_DIR) $(STEPS)

cpu: build/test_cpu
	@echo "========================================="
	@echo "Running CPU verification test..."
	@echo "========================================="
	./build/test_cpu $(DATA_DIR)

build:
	mkdir -p build

# ===========================================================
# Phase 2 — Vitis HLS builds
# ===========================================================

# ---- Software emulation (fast, functional) ----
build/transformer_kernel_sw_emu.xclbin: src/kernel.cpp | build
	$(VPP) $(VPP_FLAGS) -t sw_emu -c -k transformer_kernel \
		-o build/transformer_kernel_sw_emu.xo src/kernel.cpp
	$(VPP) $(VPP_FLAGS) -t sw_emu -l \
		-o $@ build/transformer_kernel_sw_emu.xo

sw_emu: build/transformer_kernel_sw_emu.xclbin build/host
	@echo "Software emulation build complete."

# ---- Hardware emulation (slow, cycle-accurate) ----
build/transformer_kernel_hw_emu.xclbin: src/kernel.cpp | build
	$(VPP) $(VPP_FLAGS) -t hw_emu -c -k transformer_kernel \
		-o build/transformer_kernel_hw_emu.xo src/kernel.cpp
	$(VPP) $(VPP_FLAGS) -t hw_emu -l \
		-o $@ build/transformer_kernel_hw_emu.xo

hw_emu: build/transformer_kernel_hw_emu.xclbin build/host
	@echo "Hardware emulation build complete."

# ---- Full hardware synthesis ----
build/transformer_kernel_hw.xclbin: src/kernel.cpp | build
	$(VPP) $(VPP_FLAGS) -t hw -c -k transformer_kernel \
		-o build/transformer_kernel_hw.xo src/kernel.cpp
	$(VPP) $(VPP_FLAGS) -t hw -l \
		-o $@ build/transformer_kernel_hw.xo

hw: build/transformer_kernel_hw.xclbin build/host
	@echo "Hardware build complete."

# ---- Host program ----
build/host: src/host.cpp src/transformer.h | build
	$(HOST_CXX) $(HOST_CXXFLAGS) -o $@ src/host.cpp $(HOST_LDFLAGS)

# ===========================================================
# Run targets
# ===========================================================

emconfig.json:
	emconfigutil --platform $(PLATFORM)

run_sw: build/transformer_kernel_sw_emu.xclbin build/host emconfig.json
	@echo "========================================="
	@echo "Running software emulation..."
	@echo "========================================="
	XCL_EMULATION_MODE=sw_emu ./build/host build/transformer_kernel_sw_emu.xclbin $(DATA_DIR)

run_hw_emu: build/transformer_kernel_hw_emu.xclbin build/host
	@echo "========================================="
	@echo "Running hardware emulation..."
	@echo "========================================="
	XCL_EMULATION_MODE=hw_emu ./build/host build/transformer_kernel_hw_emu.xclbin $(DATA_DIR)

run_hw: build/transformer_kernel_hw.xclbin build/host
	@echo "========================================="
	@echo "Running on FPGA hardware..."
	@echo "========================================="
	./build/host build/transformer_kernel_hw.xclbin $(DATA_DIR)

# ===========================================================
# Export
# ===========================================================

export:
	cd .. && python fpga/export_weights.py

export_flat:
	cd .. && python fpga/export_weights_flat.py

# ===========================================================
# Clean
# ===========================================================

clean:
	rm -rf build *.log *.jou .Xil _x
