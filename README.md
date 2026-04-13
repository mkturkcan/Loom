# Loom: A Scalable Computer Architecture for Looped Transformers

A general-purpose computer implemented as a looped transformer with analytically derived weights. No training data, no gradient descent. Programs are written in C, compiled to a 21-opcode ISA, and executed as iterated matrix multiplications through 8 fixed-weight transformer layers.

**[Live Demos](https://mkturkcan.github.io/Loom/demos/)** | **[Paper (PDF)](https://arxiv.org/abs/2604.08816)** | **[Hugging Face](https://huggingface.co/mehmetkeremturkcan/Loom)**

## Highlights

- **21 opcodes in 8 layers.** Extends the single-instruction (SUBLEQ) baseline of Giannou et al. from 1 opcode in 10 layers to 21 opcodes in 8 layers.
- **Every weight is analytical.** No training, no gradient descent, no data. The weights are derived from the ISA specification.
- **Argmax attention.** Replaces softmax with numerically exact top-2 selection. Programs run for millions of steps without drift.
- **C compiler.** Write C, compile to ISA, run on the transformer. Both Python and in-browser JavaScript compilers included.
- **STORE opcode.** Indirect memory write reduces the 9x9 Sudoku solver from 1,085 to 284 instructions.
- **Three scales.** 146x512 (8 MB), 155x1024 (15 MB), 164x2048 (28 MB). Same architecture, same compiler.
- **FPGA verified.** Synthesized and tested on a Xilinx Alveo U200.
- **140+ tests pass.** Opcodes, integration, compiled C programs.

## Quick Start

```bash
pip install torch numpy
python -c "
from loom_v1 import LoomConfig, LoomComputer, init_state, read_memory, get_pc, OP_INC, OP_HALT
from subleq import signed_from_bipolar
import torch

cfg = LoomConfig(s=32, m=8, n=64, N=8)
comp = LoomComputer(cfg)
X = init_state(cfg, [5,0,0,0,0,0,0,0], [(OP_INC, cfg.s, 0), (OP_HALT, 0, 0)])

with torch.no_grad():
    while get_pc(X, cfg) != 0:
        X = comp.step(X)

print('mem[0] =', read_memory(X, cfg)[0])  # 6
"
```

## Run the Tests

```bash
python test_v4.py                          # 42 opcode tests
python test_v4_standard.py                 # 29 standard variant tests
python -m pytest test_swap.py -q           # 19 swap + regression tests
python -m pytest test_comprehensive.py -v  # 50 compiled C tests
```

## Run the Demos

```bash
cd demos && python -m http.server 8000
# Open http://localhost:8000
```

All demos run client-side via ONNX Runtime WebGPU. No server computation.

## Architecture

```
L1   Fetch instruction at PC            Attention: Q=K on PC + pos_enc
L2   Read operands + decode opcode      3-head attention, opcode-gated FFN
L3   Indirect read + correction         LOAD/FIND attention + snap to +/-1
L4   Subtract (direct borrow chain)     Single-layer subtraction, 6 thresholds/bit
L5   Write result to memory             Attention-based memory store
L6   Branch flag + PC increment         Merged: condition check + PC+1
L7   Branch select                      Mux: branch target or PC+1
L8   Error correction                   Clamp all values to +/-1
```

Every forward pass executes one instruction. The entire machine state fits in a single fixed-size tensor. The weights are program-independent: any compiled program runs without changing the model.

## Instruction Set

| Op | Mnemonic | Operation |
|---|---|---|
| 0 | HALT | Stop execution |
| 1 | MOV | mem[b] = mem[c] |
| 2 | ADD | mem[b] += mem[c] |
| 3-5 | JMP/JZ/JNZ | Unconditional and conditional branches |
| 6-7 | INC/DEC | Increment and decrement |
| 8-9 | SHL/SHR | Bitwise shifts |
| 10 | CMP | Branch if negative |
| 11 | LOAD | mem[b] = mem[mem[c]] (indirect read) |
| 12-14 | AND/OR/XOR | Bitwise operations |
| 15 | SUB | mem[b] -= mem[c] |
| 16 | FIND | Content-addressable search |
| 17 | SWAP | Atomic swap |
| 18 | CMOV | Conditional move |
| 19 | MULACC | Multiply-accumulate step |
| 20 | STORE | mem[mem[c]] = mem[b] (indirect write) |
| >=32 | SUBLEQ | Subtract and branch if nonpositive |

## File Structure

```
loom_v1.py               Main computer: 8-layer transformer, 21 opcodes
subleq.py                Base transformer layer with argmax attention
c_compiler.py            C-to-ISA compiler
loom_v1_standard.py      Broadcast-bias + argmax variant
test_*.py                140+ tests (all pass)
demos/                   Browser demos (ONNX WebGPU, no server)
  index.html             Demo hub
  sorting_viz.html       Sorting with 3D architecture visualization
  debugger_viz.html      C debugger with architecture visualization
  ...
fpga/                    FPGA implementation (Alveo U200)
technical_report.tex     Paper
refs.bib                 Bibliography
```

## Citation

```bibtex
@misc{turkcan2026loomscalableanalyticalneural,
      title={Loom: A Scalable Analytical Neural Computer Architecture}, 
      author={Mehmet Kerem Turkcan},
      year={2026},
      eprint={2604.08816},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.08816}, 
}
```

## License

See LICENSE file.
