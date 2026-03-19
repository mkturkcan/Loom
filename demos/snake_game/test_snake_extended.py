import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from extended_isa_v4 import ExtendedNeuralComputerV4, init_state_v4, read_memory_v4
from snake_extended import (
    compile_snake_program,
    run_tick,
    SnakeLayout,
    _body_x,
    _body_y,
)

MAX_TICK_STEPS = 5000
MAX_SECONDS = 180.0


def _build_state():
    cfg, memory, commands, meta = compile_snake_program()
    comp = ExtendedNeuralComputerV4(cfg)
    return cfg, memory, commands, meta, comp


def _set_body(memory, layout, segments):
    for i in range(layout.MAX_LEN):
        if i < len(segments):
            memory[_body_x(layout, i)] = segments[i][0]
            memory[_body_y(layout, i)] = segments[i][1]
        else:
            memory[_body_x(layout, i)] = -1
            memory[_body_y(layout, i)] = -1


def _run_once(cfg, memory, commands, meta, comp):
    X = init_state_v4(cfg, memory, commands)
    X = run_tick(comp, cfg, X, meta["loop_addr"], max_steps=MAX_TICK_STEPS, max_seconds=MAX_SECONDS)
    return read_memory_v4(X, cfg)


def test_move_right():
    cfg, memory, commands, meta, comp = _build_state()
    layout = SnakeLayout()
    memory[layout.INPUT] = 4  # no input
    result = _run_once(cfg, memory, commands, meta, comp)
    assert result[layout.HEAD_X] == 4
    assert result[layout.HEAD_Y] == 3
    assert result[layout.DIR] == 1
    assert result[layout.SCORE] == 0
    assert result[layout.LENGTH] == 1
    assert result[layout.GAME_OVER] == 0


def test_turn_up():
    cfg, memory, commands, meta, comp = _build_state()
    layout = SnakeLayout()
    memory[layout.INPUT] = 0  # up
    result = _run_once(cfg, memory, commands, meta, comp)
    assert result[layout.DIR] == 0
    assert result[layout.HEAD_X] == 3
    assert result[layout.HEAD_Y] == 2


def test_ignore_opposite():
    cfg, memory, commands, meta, comp = _build_state()
    layout = SnakeLayout()
    memory[layout.INPUT] = 3  # left (opposite of right)
    result = _run_once(cfg, memory, commands, meta, comp)
    assert result[layout.DIR] == 1
    assert result[layout.HEAD_X] == 4
    assert result[layout.HEAD_Y] == 3


def test_eat_food():
    cfg, memory, commands, meta, comp = _build_state()
    layout = SnakeLayout()
    memory[layout.FOOD_X] = 4
    memory[layout.FOOD_Y] = 3
    result = _run_once(cfg, memory, commands, meta, comp)
    assert result[layout.HEAD_X] == 4
    assert result[layout.HEAD_Y] == 3
    assert result[layout.SCORE] == 1
    assert result[layout.LENGTH] == 2


def test_wall_collision():
    cfg, memory, commands, meta, comp = _build_state()
    layout = SnakeLayout()
    memory[layout.HEAD_X] = 7
    memory[layout.HEAD_Y] = 3
    memory[layout.DIR] = 1  # right
    _set_body(memory, layout, [(7, 3)])
    result = _run_once(cfg, memory, commands, meta, comp)
    assert result[layout.GAME_OVER] == 1


def test_self_collision():
    cfg, memory, commands, meta, comp = _build_state()
    layout = SnakeLayout()
    memory[layout.HEAD_X] = 3
    memory[layout.HEAD_Y] = 3
    memory[layout.DIR] = 1  # right
    memory[layout.LENGTH] = 2
    memory[layout.HEAD_IDX] = 2
    memory[layout.TAIL_IDX] = 0
    _set_body(memory, layout, [(3, 3), (4, 3)])
    # Place food away so we don't skip tail clear.
    memory[layout.FOOD_X] = 7
    memory[layout.FOOD_Y] = 7
    result = _run_once(cfg, memory, commands, meta, comp)
    assert result[layout.GAME_OVER] == 1


def run_all():
    tests = [
        test_move_right,
        test_turn_up,
        test_ignore_opposite,
        test_eat_food,
        test_wall_collision,
        test_self_collision,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All snake tests passed")


if __name__ == "__main__":
    run_all()
