"""
Extended ISA Snake Game - Full Mechanics
=======================================

A complete snake implementation compiled into the Extended ISA program.
All game logic (movement, growth, collisions, food spawn) runs through
hardcoded transformer weights with no branching in the GPU forward pass.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extended_isa_v4 import (
    ExtendedNeuralComputerV4, ExtendedConfigV4,
    init_state_v4, read_memory_v4, get_pc_v4,
    OP_HALT, OP_MOV, OP_ADD, OP_SUB, OP_JMP, OP_JZ, OP_JNZ, OP_INC, OP_DEC,
    OP_AND, OP_XOR,
)
from subleq import signed_to_bipolar, signed_from_bipolar


@dataclass(frozen=True)
class SnakeLayout:
    ZERO: int = 0
    ONE: int = 1
    TWO: int = 2
    THREE: int = 3
    FOUR: int = 4
    GRID: int = 5
    GRID_M1: int = 6
    CONST7: int = 7
    NEG1: int = 8
    SIGN_MASK: int = 9
    RNG_A: int = 10
    RNG_B: int = 11
    RNG_C: int = 12
    RNG_D: int = 13

    HEAD_X: int = 14
    HEAD_Y: int = 15
    DIR: int = 16
    INPUT: int = 17
    FOOD_X: int = 18
    FOOD_Y: int = 19
    SCORE: int = 20
    LENGTH: int = 21
    HEAD_IDX: int = 22
    TAIL_IDX: int = 23
    ATE: int = 24
    GAME_OVER: int = 25
    FRAME: int = 26
    RAND: int = 27

    TEMP0: int = 28
    TEMP1: int = 29
    TEMP2: int = 30
    TEMP3: int = 31

    BODY_START: int = 32
    MAX_LEN: int = 8


class SnakeExtendedConfig(ExtendedConfigV4):
    """Extended config for the full Snake program."""

    def __init__(self):
        super().__init__(s=32, m=64, n=1024, N=8)


def _body_x(layout: SnakeLayout, idx: int) -> int:
    return layout.BODY_START + 2 * idx


def _body_y(layout: SnakeLayout, idx: int) -> int:
    return layout.BODY_START + 2 * idx + 1


def compile_snake_program() -> Tuple[ExtendedConfigV4, List[int], List[Tuple[int, int, int]], Dict[str, int]]:
    """
    Compile the full snake game into Extended ISA instructions.

    Returns:
        cfg, memory, commands, meta
    """
    cfg = SnakeExtendedConfig()
    layout = SnakeLayout()
    s = cfg.s
    m = cfg.m
    cmd_start = s + m

    def addr(mem_idx: int) -> int:
        return s + mem_idx

    # ===== Initialize memory =====
    mem = [0] * m

    # Constants
    mem[layout.ZERO] = 0
    mem[layout.ONE] = 1
    mem[layout.TWO] = 2
    mem[layout.THREE] = 3
    mem[layout.FOUR] = 4
    mem[layout.GRID] = 8
    mem[layout.GRID_M1] = 7
    mem[layout.CONST7] = 7
    mem[layout.NEG1] = -1
    mem[layout.SIGN_MASK] = -128  # 0x80
    mem[layout.RNG_A] = 13
    mem[layout.RNG_B] = 109
    mem[layout.RNG_C] = 37
    mem[layout.RNG_D] = -75  # 0xB5 = 181 unsigned

    # Game state
    mem[layout.HEAD_X] = 3
    mem[layout.HEAD_Y] = 3
    mem[layout.DIR] = 1  # right
    mem[layout.INPUT] = 4  # 4 = no input
    mem[layout.FOOD_X] = 5
    mem[layout.FOOD_Y] = 3
    mem[layout.SCORE] = 0
    mem[layout.LENGTH] = 1
    mem[layout.HEAD_IDX] = 1
    mem[layout.TAIL_IDX] = 0
    mem[layout.ATE] = 0
    mem[layout.GAME_OVER] = 0
    mem[layout.FRAME] = 0
    mem[layout.RAND] = 42

    # Body initialization: fill with -1 sentinel
    for i in range(layout.MAX_LEN):
        mem[_body_x(layout, i)] = -1
        mem[_body_y(layout, i)] = -1
    mem[_body_x(layout, 0)] = mem[layout.HEAD_X]
    mem[_body_y(layout, 0)] = mem[layout.HEAD_Y]

    commands: List[List[int]] = []
    labels: Dict[str, int] = {}

    def label(name: str) -> None:
        labels[name] = len(commands)

    def emit(op: int, b: int, c: int) -> None:
        commands.append([op, b, c])

    def mov(dest: int, src: int) -> None:
        emit(OP_MOV, addr(dest), addr(src))

    def add(dest: int, src: int) -> None:
        emit(OP_ADD, addr(dest), addr(src))

    def sub(dest: int, src: int) -> None:
        emit(OP_SUB, addr(dest), addr(src))

    def inc(dest: int) -> None:
        emit(OP_INC, addr(dest), 0)

    def dec(dest: int) -> None:
        emit(OP_DEC, addr(dest), 0)

    def band(dest: int, src: int) -> None:
        emit(OP_AND, addr(dest), addr(src))

    def bxor(dest: int, src: int) -> None:
        emit(OP_XOR, addr(dest), addr(src))

    def jz(mem_idx: int, target: str) -> None:
        emit(OP_JZ, addr(mem_idx), target)  # target patched later

    def jnz(mem_idx: int, target: str) -> None:
        emit(OP_JNZ, addr(mem_idx), target)

    def jmp(target: str) -> None:
        emit(OP_JMP, 0, target)

    def halt() -> None:
        emit(OP_HALT, 0, 0)

    bit_masks = {
        0: layout.ONE,
        1: layout.TWO,
        2: layout.FOUR,
    }

    def emit_dispatch(idx_var: int, prefix: str, leaf_fn, after_label: str) -> None:
        def rec(bit: int, base: int) -> None:
            if bit < 0:
                leaf_fn(base)
                jmp(after_label)
                return
            zero_label = f"{prefix}_b{bit}_0_{base}"
            one_label = f"{prefix}_b{bit}_1_{base}"
            mov(layout.TEMP0, idx_var)
            band(layout.TEMP0, bit_masks[bit])
            jz(layout.TEMP0, zero_label)
            label(one_label)
            rec(bit - 1, base | (1 << bit))
            label(zero_label)
            rec(bit - 1, base)

        rec(2, 0)
        label(after_label)

    # ===== Program =====

    label("LOOP")
    jnz(layout.GAME_OVER, "HALT")
    inc(layout.FRAME)

    # Input handling (ignore if input == 4 or opposite dir)
    mov(layout.TEMP0, layout.INPUT)
    sub(layout.TEMP0, layout.FOUR)
    jz(layout.TEMP0, "SKIP_INPUT")

    # temp1 = dir + 2
    mov(layout.TEMP1, layout.DIR)
    inc(layout.TEMP1)
    inc(layout.TEMP1)
    # temp2 = temp1 - 4
    mov(layout.TEMP2, layout.TEMP1)
    sub(layout.TEMP2, layout.FOUR)
    # temp3 = sign(temp2)
    mov(layout.TEMP3, layout.TEMP2)
    band(layout.TEMP3, layout.SIGN_MASK)
    jnz(layout.TEMP3, "OPP_READY")
    # wrap: temp1 = temp2 (temp1 - 4)
    mov(layout.TEMP1, layout.TEMP2)
    label("OPP_READY")
    mov(layout.TEMP2, layout.INPUT)
    sub(layout.TEMP2, layout.TEMP1)
    jz(layout.TEMP2, "SKIP_INPUT")
    mov(layout.DIR, layout.INPUT)
    label("SKIP_INPUT")

    # Movement
    mov(layout.TEMP0, layout.DIR)
    jz(layout.TEMP0, "MOVE_UP")
    sub(layout.TEMP0, layout.ONE)
    jz(layout.TEMP0, "MOVE_RIGHT")
    sub(layout.TEMP0, layout.ONE)
    jz(layout.TEMP0, "MOVE_DOWN")

    # MOVE_LEFT
    dec(layout.HEAD_X)
    jmp("AFTER_MOVE")
    label("MOVE_UP")
    dec(layout.HEAD_Y)
    jmp("AFTER_MOVE")
    label("MOVE_RIGHT")
    inc(layout.HEAD_X)
    jmp("AFTER_MOVE")
    label("MOVE_DOWN")
    inc(layout.HEAD_Y)
    label("AFTER_MOVE")

    # Boundary checks: negative or >= grid
    mov(layout.TEMP0, layout.HEAD_X)
    band(layout.TEMP0, layout.SIGN_MASK)
    jnz(layout.TEMP0, "COLLIDE")
    mov(layout.TEMP0, layout.HEAD_Y)
    band(layout.TEMP0, layout.SIGN_MASK)
    jnz(layout.TEMP0, "COLLIDE")

    mov(layout.TEMP0, layout.HEAD_X)
    sub(layout.TEMP0, layout.GRID)
    band(layout.TEMP0, layout.SIGN_MASK)
    jz(layout.TEMP0, "COLLIDE")

    mov(layout.TEMP0, layout.HEAD_Y)
    sub(layout.TEMP0, layout.GRID)
    band(layout.TEMP0, layout.SIGN_MASK)
    jz(layout.TEMP0, "COLLIDE")

    # ATE flag
    mov(layout.ATE, layout.ZERO)
    mov(layout.TEMP0, layout.HEAD_X)
    sub(layout.TEMP0, layout.FOOD_X)
    jnz(layout.TEMP0, "NO_EAT")
    mov(layout.TEMP0, layout.HEAD_Y)
    sub(layout.TEMP0, layout.FOOD_Y)
    jnz(layout.TEMP0, "NO_EAT")
    mov(layout.ATE, layout.ONE)
    label("NO_EAT")

    # Clear tail if not eating
    jnz(layout.ATE, "SKIP_TAIL_CLEAR")

    def emit_tail_clear(idx: int) -> None:
        mov(_body_x(layout, idx), layout.NEG1)
        mov(_body_y(layout, idx), layout.NEG1)

    emit_dispatch(layout.TAIL_IDX, "TAIL", emit_tail_clear, "AFTER_TAIL")

    inc(layout.TAIL_IDX)
    band(layout.TAIL_IDX, layout.CONST7)

    label("SKIP_TAIL_CLEAR")

    # Self-collision check (compare against all body segments)
    for i in range(layout.MAX_LEN):
        mov(layout.TEMP0, _body_x(layout, i))
        sub(layout.TEMP0, layout.HEAD_X)
        jnz(layout.TEMP0, f"COLL_SKIP_{i}")
        mov(layout.TEMP0, _body_y(layout, i))
        sub(layout.TEMP0, layout.HEAD_Y)
        jz(layout.TEMP0, "COLLIDE")
        label(f"COLL_SKIP_{i}")

    # Write new head at head_idx
    def emit_head_write(idx: int) -> None:
        mov(_body_x(layout, idx), layout.HEAD_X)
        mov(_body_y(layout, idx), layout.HEAD_Y)

    emit_dispatch(layout.HEAD_IDX, "HEAD", emit_head_write, "AFTER_HEAD")

    inc(layout.HEAD_IDX)
    band(layout.HEAD_IDX, layout.CONST7)

    # If ate: grow and respawn food (with collision check)
    jz(layout.ATE, "SKIP_EAT")
    inc(layout.LENGTH)
    inc(layout.SCORE)

    # Spawn food at random position, retry if it overlaps any body segment
    label("RESPAWN_FOOD")
    add(layout.RAND, layout.RNG_A)
    bxor(layout.RAND, layout.RNG_B)
    mov(layout.FOOD_X, layout.RAND)
    band(layout.FOOD_X, layout.GRID_M1)

    add(layout.RAND, layout.RNG_C)
    bxor(layout.RAND, layout.RNG_D)
    mov(layout.FOOD_Y, layout.RAND)
    band(layout.FOOD_Y, layout.GRID_M1)

    # Check food against head
    mov(layout.TEMP0, layout.FOOD_X)
    sub(layout.TEMP0, layout.HEAD_X)
    jnz(layout.TEMP0, "FOOD_NOT_HEAD")
    mov(layout.TEMP0, layout.FOOD_Y)
    sub(layout.TEMP0, layout.HEAD_Y)
    jz(layout.TEMP0, "RESPAWN_FOOD")
    label("FOOD_NOT_HEAD")

    # Check food against each body segment (unrolled, 8 segments)
    for idx in range(layout.MAX_LEN):
        ok_label = f"FOOD_NOT_BODY_{idx}"
        mov(layout.TEMP0, layout.FOOD_X)
        sub(layout.TEMP0, _body_x(layout, idx))
        jnz(layout.TEMP0, ok_label)
        mov(layout.TEMP0, layout.FOOD_Y)
        sub(layout.TEMP0, _body_y(layout, idx))
        jz(layout.TEMP0, "RESPAWN_FOOD")
        label(ok_label)

    label("SKIP_EAT")

    jmp("LOOP")

    label("COLLIDE")
    mov(layout.GAME_OVER, layout.ONE)
    label("HALT")
    halt()

    # Patch labels
    for i, cmd in enumerate(commands):
        if isinstance(cmd[2], str):
            label_name = cmd[2]
            if label_name not in labels:
                raise ValueError(f"Undefined label: {label_name}")
            cmd[2] = cmd_start + labels[label_name]

    if len(commands) > cfg.n - cmd_start:
        raise ValueError(f"Program too large: {len(commands)} commands for n={cfg.n}")

    meta = {
        "cmd_start": cmd_start,
        "loop_addr": cmd_start + labels["LOOP"],
        "halt_addr": cmd_start + labels["HALT"],
    }

    return cfg, mem, [tuple(c) for c in commands], meta


def move_computer_to_device(computer: ExtendedNeuralComputerV4, device: torch.device):
    """Move all layer weights to the given device."""
    for layer in computer.layers:
        for attr in ['Q', 'K', 'V', 'Q1', 'K1', 'V1', 'Q2', 'K2', 'V2', 'Q3', 'K3', 'V3']:
            if hasattr(layer, attr):
                setattr(layer, attr, getattr(layer, attr).to(device))
        layer.W1 = layer.W1.to(device)
        layer.b1 = layer.b1.to(device)
        layer.W2 = layer.W2.to(device)
        layer.b2 = layer.b2.to(device)


def run_tick(
    computer: ExtendedNeuralComputerV4,
    cfg: ExtendedConfigV4,
    X: torch.Tensor,
    loop_addr: int,
    max_steps: int = 5000,
    max_seconds: float = 180.0,
    trace: bool = False,
    trace_every: int = 200,
    trace_limit: int = 50,
    watch: List[int] | None = None,
) -> torch.Tensor:
    """Run until the next loop iteration or halt, with safety limits."""
    steps = 0
    start = time.monotonic()
    trace_count = 0
    pc_history: List[int] = []
    if watch is None:
        watch = [
            SnakeLayout.HEAD_X,
            SnakeLayout.HEAD_Y,
            SnakeLayout.DIR,
            SnakeLayout.INPUT,
            SnakeLayout.FOOD_X,
            SnakeLayout.FOOD_Y,
            SnakeLayout.LENGTH,
            SnakeLayout.SCORE,
            SnakeLayout.ATE,
            SnakeLayout.GAME_OVER,
            SnakeLayout.HEAD_IDX,
            SnakeLayout.TAIL_IDX,
            SnakeLayout.RAND,
        ]

    def _read_mem_indices(indices: List[int]) -> Dict[int, int]:
        values = {}
        X_cpu = X.cpu() if X.is_cuda else X
        for idx in indices:
            col = cfg.s + idx
            values[idx] = signed_from_bipolar(
                X_cpu[cfg.idx_memory:cfg.idx_memory + cfg.N, col]
            )
        return values

    with torch.no_grad():
        while True:
            if steps >= max_steps:
                raise RuntimeError(
                    f"run_tick exceeded max_steps={max_steps}; "
                    f"last_pc={pc_history[-1] if pc_history else None}; "
                    f"pc_tail={pc_history[-16:]}"
                )
            if time.monotonic() - start >= max_seconds:
                raise RuntimeError(
                    f"run_tick exceeded max_seconds={max_seconds}; "
                    f"last_pc={pc_history[-1] if pc_history else None}; "
                    f"pc_tail={pc_history[-16:]}"
                )
            X_cpu = X.cpu() if X.is_cuda else X
            pc = get_pc_v4(X_cpu, cfg)
            pc_history.append(pc)
            if pc == 0:
                return X
            if trace and steps % trace_every == 0 and trace_count < trace_limit:
                snapshot = _read_mem_indices(watch)
                print(f"[tick] step={steps} pc={pc} mem={snapshot}")
                trace_count += 1
            X = computer.step(X)
            steps += 1
            X_cpu = X.cpu() if X.is_cuda else X
            pc = get_pc_v4(X_cpu, cfg)
            if steps > 0 and pc == loop_addr:
                return X


def test_snake():
    """Quick sanity test for the full snake program."""
    print("=" * 60)
    print("Testing Full Extended ISA Snake")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg, memory, commands, meta = compile_snake_program()
    computer = ExtendedNeuralComputerV4(cfg)
    if device.type == "cuda":
        move_computer_to_device(computer, device)
    X = init_state_v4(cfg, memory, commands).to(device)

    print(f"Commands: {len(commands)} (n={cfg.n}, cmd_start={meta['cmd_start']})")
    print(f"Initial: head=({memory[14]},{memory[15]}), food=({memory[18]},{memory[19]})")

    for step in range(5):
        X = run_tick(computer, cfg, X, meta["loop_addr"])
        mem = read_memory_v4(X.cpu(), cfg)
        print(
            f"Tick {step:2d}: head=({mem[14]},{mem[15]}), "
            f"len={mem[21]}, score={mem[20]}, over={mem[25]}, pc={get_pc_v4(X.cpu(), cfg)}"
        )
        if mem[25] != 0:
            break

    return cfg, memory, commands


def export_snake() -> None:
    """Export config/state and ONNX for the snake program."""
    print("\n" + "=" * 60)
    print("Exporting Full Snake Program")
    print("=" * 60)

    cfg, memory, commands, meta = compile_snake_program()
    X = init_state_v4(cfg, memory, commands)

    output_dir = os.path.dirname(__file__)
    mirror_dir = os.path.join(output_dir, "snake_game")

    config = {
        "s": cfg.s,
        "m": cfg.m,
        "n": cfg.n,
        "N": cfg.N,
        "logn": cfg.logn,
        "d_model": cfg.d_model,
        "idx_memory": cfg.idx_memory,
        "idx_pc": cfg.idx_pc,
        "head_x_addr": 14,
        "head_y_addr": 15,
        "direction_addr": 16,
        "input_addr": 17,
        "food_x_addr": 18,
        "food_y_addr": 19,
        "score_addr": 20,
        "length_addr": 21,
        "head_idx_addr": 22,
        "tail_idx_addr": 23,
        "ate_addr": 24,
        "game_over_addr": 25,
        "frame_addr": 26,
        "rand_addr": 27,
        "body_start_addr": 32,
        "max_length": 8,
        "grid_size": 8,
        "loop_addr": meta["loop_addr"],
        "halt_addr": meta["halt_addr"],
        "isa_type": "extended",
    }

    for target_dir in [output_dir, mirror_dir]:
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        with open(os.path.join(target_dir, "initial_state.json"), "w") as f:
            json.dump({"state": X.tolist(), "memory": memory}, f)

    print("Saved config.json and initial_state.json")

    # Export single-step ONNX model
    class Module(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.computer = ExtendedNeuralComputerV4(cfg)

        def forward(self, x):
            return self.computer.step(x)

    model = Module(cfg)
    model.eval()

    onnx_path = os.path.join(output_dir, "snake.onnx")
    torch.onnx.export(
        model,
        X,
        onnx_path,
        export_params=True,
        opset_version=18,
        input_names=["state"],
        output_names=["new_state"],
    )

    # Downgrade IR version for browser compatibility
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx_model.ir_version = 9
    onnx.save(onnx_model, onnx_path)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Saved snake.onnx ({size_mb:.1f} MB)")


if __name__ == "__main__":
    test_snake()
    export_snake()
