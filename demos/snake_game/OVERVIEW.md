# Neural Transformer Snake Game - Overview

## What This Is

This project implements a **complete Snake game running entirely on hardcoded transformer weights** - no training involved. The game logic (movement, collisions, food spawning, growth) executes through a neural network that functions as a programmable computer.

## Key Concepts

### Looped Transformer Computer

A transformer network configured to act as a programmable CPU:

- **State Matrix**: 121 × 1024 tensor representing memory, program counter, and scratchpad
- **10 Transformer Layers**: Execute one instruction per pass (fetch → decode → execute → writeback)
- **No Branching in Forward Pass**: All conditional logic handled through softmax gating

### Extended ISA (Instruction Set Architecture)

13 operations + base SUBLEQ:

| Opcode | Instruction | Description |
|--------|-------------|-------------|
| 0 | HALT | Stop execution |
| 1 | MOV | Copy value |
| 2 | ADD | Addition |
| 3 | JMP | Unconditional jump |
| 4 | JZ | Jump if zero |
| 5 | JNZ | Jump if not zero |
| 6 | INC | Increment by 1 |
| 7 | DEC | Decrement by 1 |
| 12 | AND | Bitwise AND |
| 13 | OR | Bitwise OR |
| 14 | XOR | Bitwise XOR |
| 15 | SUB | Subtraction |
| ≥16 | SUBLEQ | Subtract and branch if ≤0 |

### Snake Game Implementation

The game compiles to ~400 Extended ISA instructions handling:

- **Input Processing**: Direction changes with opposite-direction rejection
- **Movement**: 4-way movement with boundary collision
- **Collision Detection**: Wall boundaries + self-collision against 16-segment body
- **Food System**: PRNG-based food spawning that avoids snake body
- **Circular Buffer**: Head/tail indices wrap for efficient body tracking

## File Structure

```
snake_game/
├── snake_extended.py      # Game logic → Extended ISA compiler
├── export_onnx.py         # ONNX export utilities
├── test_snake_extended.py # Game tests
├── snake.onnx             # Exported model (4.5 MB)
├── config.json            # Architecture configuration
├── initial_state.json     # Starting game state
└── index.html             # Web interface (ONNX.js)
```

## Memory Layout (64 slots)

| Addresses | Purpose |
|-----------|---------|
| 0-13 | Constants (0,1,2,3,4,8,7,15,-1,128,RNG) |
| 14-27 | Game state (head, direction, food, score, etc.) |
| 28-31 | Temporary variables |
| 32-63 | Body segments (16 × 2 for x,y pairs) |

## Configuration (config.json)

```json
{
  "d_model": 121,        // Feature dimension (tensor rows)
  "n": 1024,             // Total columns
  "m": 64,               // Memory slots
  "N": 8,                // Bits per value
  "grid_size": 8,        // 8×8 play area
  "head_x_addr": 14,     // Memory addresses for game state
  "head_y_addr": 15,
  "direction_addr": 16,
  "input_addr": 17,
  "food_x_addr": 18,
  "food_y_addr": 19,
  "score_addr": 20,
  "game_over_addr": 25
}
```

## How It Works

1. **State Initialization**: Load initial_state.json (121×1024 tensor + memory array)
2. **Input Injection**: Write direction (0=up, 1=right, 2=down, 3=left, 4=none) to input_addr
3. **Tick Execution**: Run ONNX model in a loop until PC returns to loop_addr
4. **State Reading**: Extract game state from memory addresses
5. **Rendering**: Draw 8×8 grid with snake and food

## ONNX Model Interface

**Input**: `state` tensor (121 × 1024, float32)
**Output**: `new_state` tensor (same shape)

Each model invocation executes one Extended ISA instruction. A full game tick requires hundreds of invocations.

## Controls

- Arrow keys or WASD for direction
- Snake cannot reverse into itself
- Game ends on wall or self collision
