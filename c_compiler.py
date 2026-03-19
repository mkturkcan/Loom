"""
C-to-Extended-ISA Compiler
==========================

Compiles a subset of C to Extended ISA V4 instructions that run on the
neural SUBLEQ transformer computer.

Supported C subset:
  - int variables (N-bit signed, default N=8: -128 to 127)
  - Arithmetic: +, -, &, |, ^, unary -
  - Comparisons: ==, !=, <, >, <=, >=
  - Logical: &&, ||, !
  - Control flow: if/else, while, for
  - Arrays with constant size and constant or variable index
  - Functions (inlined, no recursion)
  - return statement
  - printf(var) emits value to a designated output slot

Limitations:
  - No pointers, structs, malloc, recursion
  - No division or modulo (would need software routines)
  - Use mul(a, b) builtin for multiplication (MULACC-based, N instructions)
  - Signed overflow on comparisons with large value differences
  - Max ~48 user variable slots, ~944 instruction slots
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

from extended_isa_v4 import (
    ExtendedNeuralComputerV4,
    ExtendedConfigV4,
    init_state_v4,
    read_memory_v4,
    get_pc_v4,
    OP_HALT, OP_MOV, OP_ADD, OP_SUB, OP_JMP, OP_JZ, OP_JNZ,
    OP_INC, OP_DEC, OP_AND, OP_OR, OP_XOR,
    OP_SHL, OP_SHR, OP_CMP, OP_LOAD,
    OP_FIND, OP_SWAP, OP_CMOV, OP_MULACC, OP_STORE,
)


# ============================================================
# Lexer
# ============================================================

class TT(Enum):
    INT = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    RETURN = auto()
    VOID = auto()
    IDENT = auto()
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    AMP = auto()
    PIPE = auto()
    CARET = auto()
    TILDE = auto()
    BANG = auto()
    LSHIFT = auto()
    RSHIFT = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    AND = auto()
    OR = auto()
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    SEMI = auto()
    COMMA = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    EOF = auto()


@dataclass
class Token:
    type: TT
    value: str
    line: int


KEYWORDS = {
    'int': TT.INT, 'if': TT.IF, 'else': TT.ELSE,
    'while': TT.WHILE, 'for': TT.FOR, 'return': TT.RETURN,
    'void': TT.VOID,
}

# Ordered longest-first so >= matches before >
OPERATORS = [
    ('==', TT.EQ), ('!=', TT.NEQ), ('<=', TT.LE), ('>=', TT.GE),
    ('<<', TT.LSHIFT), ('>>', TT.RSHIFT),
    ('&&', TT.AND), ('||', TT.OR),
    ('+=', TT.PLUS_ASSIGN), ('-=', TT.MINUS_ASSIGN),
    ('+', TT.PLUS), ('-', TT.MINUS),
    ('&', TT.AMP), ('|', TT.PIPE), ('^', TT.CARET),
    ('~', TT.TILDE), ('!', TT.BANG),
    ('<', TT.LT), ('>', TT.GT), ('=', TT.ASSIGN),
    (';', TT.SEMI), (',', TT.COMMA),
    ('(', TT.LPAREN), (')', TT.RPAREN),
    ('{', TT.LBRACE), ('}', TT.RBRACE),
    ('[', TT.LBRACKET), (']', TT.RBRACKET),
]


def lex(source: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    line = 1
    while i < len(source):
        c = source[i]
        # Whitespace
        if c in ' \t\r':
            i += 1
            continue
        if c == '\n':
            line += 1
            i += 1
            continue
        # Line comment
        if source[i:i+2] == '//':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue
        # Block comment
        if source[i:i+2] == '/*':
            i += 2
            while i + 1 < len(source) and source[i:i+2] != '*/':
                if source[i] == '\n':
                    line += 1
                i += 1
            i += 2
            continue
        # Number
        if c.isdigit() or (c == '-' and i + 1 < len(source) and source[i+1].isdigit()
                           and (not tokens or tokens[-1].type in (
                               TT.LPAREN, TT.COMMA, TT.ASSIGN, TT.PLUS_ASSIGN,
                               TT.MINUS_ASSIGN, TT.SEMI, TT.LBRACKET,
                               TT.PLUS, TT.MINUS, TT.EQ, TT.NEQ, TT.LT, TT.GT,
                               TT.LE, TT.GE, TT.AND, TT.OR, TT.RETURN,
                               TT.AMP, TT.PIPE, TT.CARET, TT.BANG, TT.TILDE))):
            j = i
            if source[j] == '-':
                j += 1
            while j < len(source) and source[j].isdigit():
                j += 1
            tokens.append(Token(TT.NUMBER, source[i:j], line))
            i = j
            continue
        # Identifier / keyword
        if c.isalpha() or c == '_':
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == '_'):
                j += 1
            word = source[i:j]
            tt = KEYWORDS.get(word, TT.IDENT)
            tokens.append(Token(tt, word, line))
            i = j
            continue
        # Operators / punctuation
        matched = False
        for op, tt in OPERATORS:
            if source[i:i+len(op)] == op:
                tokens.append(Token(tt, op, line))
                i += len(op)
                matched = True
                break
        if matched:
            continue
        raise SyntaxError(f"Unexpected character '{c}' at line {line}")

    tokens.append(Token(TT.EOF, '', line))
    return tokens


# ============================================================
# AST Nodes
# ============================================================

@dataclass
class IntLiteral:
    value: int

@dataclass
class VarRef:
    name: str

@dataclass
class ArrayRef:
    name: str
    index: 'Expr'

@dataclass
class UnaryOp:
    op: str   # '-', '~', '!'
    expr: 'Expr'

@dataclass
class BinaryOp:
    op: str   # '+', '-', '&', '|', '^', '==', '!=', '<', '>', '<=', '>=', '&&', '||'
    left: 'Expr'
    right: 'Expr'

@dataclass
class FuncCall:
    name: str
    args: List['Expr']

Expr = Union[IntLiteral, VarRef, ArrayRef, UnaryOp, BinaryOp, FuncCall]

@dataclass
class VarDecl:
    name: str
    size: Optional[int]  # None for scalar, >0 for array
    init: Optional[Expr]
    line: int = 0

@dataclass
class Assignment:
    target: Union[VarRef, ArrayRef]
    op: str   # '=', '+=', '-='
    value: Expr
    line: int = 0

@dataclass
class IfStmt:
    cond: Expr
    then_body: 'Block'
    else_body: Optional['Block']
    line: int = 0

@dataclass
class WhileStmt:
    cond: Expr
    body: 'Block'
    line: int = 0

@dataclass
class ForStmt:
    init: Optional['Stmt']
    cond: Optional[Expr]
    update: Optional['Stmt']
    body: 'Block'
    line: int = 0

@dataclass
class ReturnStmt:
    value: Optional[Expr]
    line: int = 0

@dataclass
class ExprStmt:
    expr: Expr
    line: int = 0

Stmt = Union[VarDecl, Assignment, IfStmt, WhileStmt, ForStmt, ReturnStmt, ExprStmt]

@dataclass
class Block:
    stmts: List[Stmt]

@dataclass
class FuncDecl:
    name: str
    params: List[str]
    body: Block

@dataclass
class Program:
    functions: List[FuncDecl]
    globals: List[VarDecl]


# ============================================================
# Parser (recursive descent)
# ============================================================

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, tt: TT) -> Token:
        t = self.advance()
        if t.type != tt:
            raise SyntaxError(f"Expected {tt.name}, got '{t.value}' at line {t.line}")
        return t

    def match(self, tt: TT) -> Optional[Token]:
        if self.peek().type == tt:
            return self.advance()
        return None

    def parse_program(self) -> Program:
        functions = []
        globals_ = []
        while self.peek().type != TT.EOF:
            if self.peek().type in (TT.INT, TT.VOID):
                # Look ahead: type name '(' => function, else global var
                saved = self.pos
                self.advance()  # type
                name_tok = self.advance()  # name
                if self.peek().type == TT.LPAREN:
                    self.pos = saved
                    functions.append(self.parse_func_decl())
                else:
                    self.pos = saved
                    globals_.append(self.parse_var_decl())
            else:
                raise SyntaxError(f"Unexpected token '{self.peek().value}' at line {self.peek().line}")
        return Program(functions=functions, globals=globals_)

    def parse_func_decl(self) -> FuncDecl:
        self.advance()  # int/void
        name = self.expect(TT.IDENT).value
        self.expect(TT.LPAREN)
        params = []
        if self.peek().type != TT.RPAREN:
            self.expect(TT.INT)
            params.append(self.expect(TT.IDENT).value)
            while self.match(TT.COMMA):
                self.expect(TT.INT)
                params.append(self.expect(TT.IDENT).value)
        self.expect(TT.RPAREN)
        body = self.parse_block()
        return FuncDecl(name=name, params=params, body=body)

    def parse_block(self) -> Block:
        self.expect(TT.LBRACE)
        stmts = []
        while self.peek().type != TT.RBRACE:
            stmts.append(self.parse_stmt())
        self.expect(TT.RBRACE)
        return Block(stmts=stmts)

    def parse_stmt(self) -> Stmt:
        t = self.peek()
        if t.type == TT.INT:
            return self.parse_var_decl()
        if t.type == TT.IF:
            return self.parse_if()
        if t.type == TT.WHILE:
            return self.parse_while()
        if t.type == TT.FOR:
            return self.parse_for()
        if t.type == TT.RETURN:
            return self.parse_return()
        if t.type == TT.LBRACE:
            return self.parse_block()
        # Assignment or expression statement
        return self.parse_assign_or_expr()

    def parse_var_decl(self) -> VarDecl:
        ln = self.peek().line
        self.expect(TT.INT)
        name = self.expect(TT.IDENT).value
        size = None
        init = None
        if self.match(TT.LBRACKET):
            size = int(self.expect(TT.NUMBER).value)
            self.expect(TT.RBRACKET)
        if self.match(TT.ASSIGN):
            init = self.parse_expr()
        self.expect(TT.SEMI)
        return VarDecl(name=name, size=size, init=init, line=ln)

    def parse_if(self) -> IfStmt:
        ln = self.peek().line
        self.expect(TT.IF)
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        then_body = self._parse_body()
        else_body = None
        if self.match(TT.ELSE):
            else_body = self._parse_body()
        return IfStmt(cond=cond, then_body=then_body, else_body=else_body, line=ln)

    def parse_while(self) -> WhileStmt:
        ln = self.peek().line
        self.expect(TT.WHILE)
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        body = self._parse_body()
        return WhileStmt(cond=cond, body=body, line=ln)

    def parse_for(self) -> ForStmt:
        ln = self.peek().line
        self.expect(TT.FOR)
        self.expect(TT.LPAREN)
        init = None
        if self.peek().type != TT.SEMI:
            init = self.parse_stmt()  # includes semicolon
        else:
            self.expect(TT.SEMI)
        cond = None
        if self.peek().type != TT.SEMI:
            cond = self.parse_expr()
        self.expect(TT.SEMI)
        update = None
        if self.peek().type != TT.RPAREN:
            update = self.parse_assign_or_expr_no_semi()
        self.expect(TT.RPAREN)
        body = self._parse_body()
        return ForStmt(init=init, cond=cond, update=update, body=body, line=ln)

    def parse_return(self) -> ReturnStmt:
        ln = self.peek().line
        self.expect(TT.RETURN)
        value = None
        if self.peek().type != TT.SEMI:
            value = self.parse_expr()
        self.expect(TT.SEMI)
        return ReturnStmt(value=value, line=ln)

    def parse_assign_or_expr(self) -> Stmt:
        stmt = self.parse_assign_or_expr_no_semi()
        self.expect(TT.SEMI)
        return stmt

    def parse_assign_or_expr_no_semi(self) -> Stmt:
        ln = self.peek().line
        expr = self.parse_expr()
        if self.peek().type in (TT.ASSIGN, TT.PLUS_ASSIGN, TT.MINUS_ASSIGN):
            op = self.advance().value
            value = self.parse_expr()
            if isinstance(expr, (VarRef, ArrayRef)):
                return Assignment(target=expr, op=op, value=value, line=ln)
            raise SyntaxError(f"Invalid assignment target")
        return ExprStmt(expr=expr, line=ln)

    def _parse_body(self) -> Block:
        if self.peek().type == TT.LBRACE:
            return self.parse_block()
        # Single statement body
        return Block(stmts=[self.parse_stmt()])

    # ---- Expression parsing (precedence climbing) ----

    def parse_expr(self) -> Expr:
        return self.parse_or()

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.peek().type == TT.OR:
            self.advance()
            right = self.parse_and()
            left = BinaryOp('||', left, right)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_bitor()
        while self.peek().type == TT.AND:
            self.advance()
            right = self.parse_bitor()
            left = BinaryOp('&&', left, right)
        return left

    def parse_bitor(self) -> Expr:
        left = self.parse_bitxor()
        while self.peek().type == TT.PIPE:
            self.advance()
            right = self.parse_bitxor()
            left = BinaryOp('|', left, right)
        return left

    def parse_bitxor(self) -> Expr:
        left = self.parse_bitand()
        while self.peek().type == TT.CARET:
            self.advance()
            right = self.parse_bitand()
            left = BinaryOp('^', left, right)
        return left

    def parse_bitand(self) -> Expr:
        left = self.parse_equality()
        while self.peek().type == TT.AMP:
            self.advance()
            right = self.parse_equality()
            left = BinaryOp('&', left, right)
        return left

    def parse_equality(self) -> Expr:
        left = self.parse_comparison()
        while self.peek().type in (TT.EQ, TT.NEQ):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(op, left, right)
        return left

    def parse_comparison(self) -> Expr:
        left = self.parse_shift()
        while self.peek().type in (TT.LT, TT.GT, TT.LE, TT.GE):
            op = self.advance().value
            right = self.parse_shift()
            left = BinaryOp(op, left, right)
        return left

    def parse_shift(self) -> Expr:
        left = self.parse_additive()
        while self.peek().type in (TT.LSHIFT, TT.RSHIFT):
            op = self.advance().value
            right = self.parse_additive()
            left = BinaryOp(op, left, right)
        return left

    def parse_additive(self) -> Expr:
        left = self.parse_unary()
        while self.peek().type in (TT.PLUS, TT.MINUS):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def parse_unary(self) -> Expr:
        if self.peek().type == TT.MINUS:
            self.advance()
            expr = self.parse_unary()
            # Fold constant: -(literal)
            if isinstance(expr, IntLiteral):
                return IntLiteral(-expr.value)
            return UnaryOp('-', expr)
        if self.peek().type == TT.TILDE:
            self.advance()
            return UnaryOp('~', self.parse_unary())
        if self.peek().type == TT.BANG:
            self.advance()
            return UnaryOp('!', self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        t = self.peek()
        if t.type == TT.NUMBER:
            self.advance()
            return IntLiteral(int(t.value))
        if t.type == TT.IDENT:
            self.advance()
            name = t.value
            # Function call
            if self.peek().type == TT.LPAREN:
                self.advance()
                args = []
                if self.peek().type != TT.RPAREN:
                    args.append(self.parse_expr())
                    while self.match(TT.COMMA):
                        args.append(self.parse_expr())
                self.expect(TT.RPAREN)
                return FuncCall(name=name, args=args)
            # Array access
            if self.peek().type == TT.LBRACKET:
                self.advance()
                index = self.parse_expr()
                self.expect(TT.RBRACKET)
                return ArrayRef(name=name, index=index)
            return VarRef(name=name)
        if t.type == TT.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TT.RPAREN)
            return expr
        raise SyntaxError(f"Unexpected token '{t.value}' at line {t.line}")


# ============================================================
# Memory Allocator
# ============================================================

@dataclass
class MemSlot:
    addr: int       # memory index (0-based, actual address = s + addr)
    name: str
    size: int = 1   # >1 for arrays


class MemoryAllocator:
    """Allocates memory slots for constants, temporaries, and user variables."""

    def __init__(self, m: int = 64, N: int = 8):
        self.m = m
        self.N = N
        self.slots: Dict[str, MemSlot] = {}
        self.constants: Dict[int, int] = {}  # value -> mem_idx
        self.next_addr = 0

    def _alloc(self, name: str, size: int = 1) -> int:
        if self.next_addr + size > self.m:
            raise RuntimeError(f"Out of memory allocating '{name}' (need {size}, {self.next_addr}/{self.m} used)")
        addr = self.next_addr
        self.slots[name] = MemSlot(addr=addr, name=name, size=size)
        self.next_addr += size
        return addr

    def alloc_constant(self, value: int) -> int:
        """Get or allocate a memory slot for a constant value."""
        mask = (1 << self.N) - 1
        half = 1 << (self.N - 1)
        value = value & mask
        if value >= half:
            value -= (1 << self.N)
        if value in self.constants:
            return self.constants[value]
        name = f"__const_{value}"
        addr = self._alloc(name)
        self.constants[value] = addr
        return addr

    def alloc_temp(self, name: str) -> int:
        return self._alloc(name)

    def alloc_var(self, name: str, size: int = 1) -> int:
        if name in self.slots:
            return self.slots[name].addr
        return self._alloc(name, size)

    def get_var(self, name: str) -> int:
        if name not in self.slots:
            raise NameError(f"Undefined variable '{name}'")
        return self.slots[name].addr

    def has_var(self, name: str) -> bool:
        return name in self.slots

    def init_memory(self) -> List[int]:
        mem = [0] * self.m
        for val, addr in self.constants.items():
            mem[addr] = val
        return mem


# ============================================================
# Code Generator
# ============================================================

class CodeGen:
    """Compiles AST to Extended ISA instructions."""

    def __init__(self, cfg: ExtendedConfigV4):
        self.cfg = cfg
        self.mem = MemoryAllocator(cfg.m, cfg.N)
        self.cmds: List[List] = []  # [a, b, c] where c can be str (label)
        self.labels: Dict[str, int] = {}
        self.label_counter = 0
        self.functions: Dict[str, FuncDecl] = {}
        self.source_map: List[Tuple[int, int]] = []  # [(instr_index, source_line), ...]

        # Pre-allocate essential constants and temporaries
        self.CONST_ZERO = self.mem.alloc_constant(0)
        self.CONST_ONE = self.mem.alloc_constant(1)
        self.CONST_SIGN = self.mem.alloc_constant(-(1 << (cfg.N - 1)))

        # Temporaries for expression evaluation
        self.T0 = self.mem.alloc_temp('__t0')
        self.T1 = self.mem.alloc_temp('__t1')
        self.T2 = self.mem.alloc_temp('__t2')
        self.T3 = self.mem.alloc_temp('__t3')

        # Return value slot
        self.RETVAL = self.mem.alloc_temp('__retval')

        # Output slot (for printf)
        self.OUTPUT = self.mem.alloc_temp('__output')

    def new_label(self, prefix: str = "L") -> str:
        self.label_counter += 1
        return f"__{prefix}_{self.label_counter}"

    def addr(self, mem_idx: int) -> int:
        return self.cfg.s + mem_idx

    def emit(self, a: int, b: int, c) -> None:
        self.cmds.append([a, b, c])

    def emit_label(self, name: str) -> None:
        self.labels[name] = len(self.cmds)

    def emit_mov(self, dest: int, src: int) -> None:
        self.emit(OP_MOV, self.addr(dest), self.addr(src))

    def emit_add(self, dest: int, src: int) -> None:
        self.emit(OP_ADD, self.addr(dest), self.addr(src))

    def emit_sub(self, dest: int, src: int) -> None:
        self.emit(OP_SUB, self.addr(dest), self.addr(src))

    def emit_and(self, dest: int, src: int) -> None:
        self.emit(OP_AND, self.addr(dest), self.addr(src))

    def emit_or(self, dest: int, src: int) -> None:
        self.emit(OP_OR, self.addr(dest), self.addr(src))

    def emit_xor(self, dest: int, src: int) -> None:
        self.emit(OP_XOR, self.addr(dest), self.addr(src))

    def emit_shl(self, dest: int) -> None:
        self.emit(OP_SHL, self.addr(dest), 0)

    def emit_shr(self, dest: int) -> None:
        self.emit(OP_SHR, self.addr(dest), 0)

    def emit_inc(self, dest: int) -> None:
        self.emit(OP_INC, self.addr(dest), 0)

    def emit_dec(self, dest: int) -> None:
        self.emit(OP_DEC, self.addr(dest), 0)

    def emit_jz(self, mem_idx: int, label: str) -> None:
        self.emit(OP_JZ, self.addr(mem_idx), label)

    def emit_jnz(self, mem_idx: int, label: str) -> None:
        self.emit(OP_JNZ, self.addr(mem_idx), label)

    def emit_cmp(self, mem_idx: int, label: str) -> None:
        """CMP (BLTZ): if mem[mem_idx] < 0, jump to label."""
        self.emit(OP_CMP, self.addr(mem_idx), label)

    def emit_cmov(self, dest: int, src: int) -> None:
        """CMOV: if mem[dest] < 0, mem[dest] = mem[src]; else no-op."""
        self.emit(OP_CMOV, self.addr(dest), self.addr(src))

    def emit_mulacc(self, dest: int, src: int) -> None:
        """MULACC: mem[dest] = (mem[dest] << 1) + (mem[src] if MSB(dest) else 0)."""
        self.emit(OP_MULACC, self.addr(dest), self.addr(src))

    def emit_swap(self, a: int, b: int) -> None:
        """SWAP: swap mem[a] and mem[b] in a single instruction."""
        self.emit(OP_SWAP, self.addr(a), self.addr(b))

    def emit_load(self, dest: int, pointer: int) -> None:
        """LOAD: mem[dest] = mem[mem[pointer]]. Indirect read."""
        self.emit(OP_LOAD, self.addr(dest), self.addr(pointer))

    def emit_store(self, src: int, pointer: int) -> None:
        """STORE: mem[mem[pointer]] = mem[src]. Indirect write."""
        self.emit(OP_STORE, self.addr(src), self.addr(pointer))

    def emit_jmp(self, label: str) -> None:
        self.emit(OP_JMP, 0, label)

    def emit_halt(self) -> None:
        self.emit(OP_HALT, 0, 0)

    def resolve(self) -> List[Tuple[int, int, int]]:
        cmd_start = self.cfg.s + self.cfg.m
        for cmd in self.cmds:
            if isinstance(cmd[2], str):
                if cmd[2] not in self.labels:
                    raise NameError(f"Undefined label '{cmd[2]}'")
                cmd[2] = cmd_start + self.labels[cmd[2]]
        return [tuple(cmd) for cmd in self.cmds]

    # ---- Compilation ----

    def compile_program(self, prog: Program) -> None:
        # Register all functions
        for func in prog.functions:
            self.functions[func.name] = func

        # Compile global variable declarations
        for g in prog.globals:
            self.compile_var_decl(g)

        # Find and compile main()
        if 'main' not in self.functions:
            raise SyntaxError("No main() function found")
        main = self.functions['main']
        self.compile_block(main.body)
        self.emit_halt()

    def compile_block(self, block: Block) -> None:
        for stmt in block.stmts:
            self.compile_stmt(stmt)

    def compile_stmt(self, stmt: Stmt) -> None:
        # Record source map entry
        line = getattr(stmt, 'line', 0)
        if line > 0:
            self.source_map.append((len(self.cmds), line))

        if isinstance(stmt, VarDecl):
            self.compile_var_decl(stmt)
        elif isinstance(stmt, Assignment):
            self.compile_assignment(stmt)
        elif isinstance(stmt, IfStmt):
            self.compile_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self.compile_while(stmt)
        elif isinstance(stmt, ForStmt):
            self.compile_for(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.compile_return(stmt)
        elif isinstance(stmt, ExprStmt):
            self.compile_expr(stmt.expr, self.T0)
        elif isinstance(stmt, Block):
            self.compile_block(stmt)
        else:
            raise SyntaxError(f"Unknown statement type: {type(stmt)}")

    def compile_var_decl(self, decl: VarDecl) -> None:
        if decl.size is not None:
            addr = self.mem.alloc_var(decl.name, decl.size)
        else:
            addr = self.mem.alloc_var(decl.name)
            if decl.init is not None:
                src = self.compile_expr(decl.init, self.T0)
                if src != addr:
                    self.emit_mov(addr, src)

    def compile_assignment(self, assign: Assignment) -> None:
        if isinstance(assign.target, VarRef):
            dest = self.mem.get_var(assign.target.name)
            if assign.op == '=':
                src = self.compile_expr(assign.value, self.T0)
                if src != dest:
                    self.emit_mov(dest, src)
            elif assign.op == '+=':
                src = self.compile_expr(assign.value, self.T0)
                self.emit_add(dest, src)
            elif assign.op == '-=':
                src = self.compile_expr(assign.value, self.T0)
                self.emit_sub(dest, src)
        elif isinstance(assign.target, ArrayRef):
            base = self.mem.get_var(assign.target.name)
            slot = self.mem.slots[assign.target.name]
            if isinstance(assign.target.index, IntLiteral):
                dest = base + assign.target.index.value
                if assign.op == '=':
                    src = self.compile_expr(assign.value, self.T0)
                    if src != dest:
                        self.emit_mov(dest, src)
                elif assign.op == '+=':
                    src = self.compile_expr(assign.value, self.T0)
                    self.emit_add(dest, src)
                elif assign.op == '-=':
                    src = self.compile_expr(assign.value, self.T0)
                    self.emit_sub(dest, src)
            else:
                # Variable index -- use STORE for indirect write
                # STORE(src, ptr): mem[mem[ptr]] = mem[src]
                # ptr must contain base + index
                idx_addr = self.compile_expr(assign.target.index, self.T1)
                base_const = self.mem.alloc_constant(base)
                # T2 = base + index (the pointer value)
                self.emit_mov(self.T2, base_const)
                self.emit_add(self.T2, idx_addr)

                if assign.op == '=':
                    src = self.compile_expr(assign.value, self.T0)
                    self.emit_store(src, self.T2)
                elif assign.op == '+=':
                    # Need LOAD + ADD + STORE
                    src = self.compile_expr(assign.value, self.T0)
                    self.emit_load(self.T3, self.T2)  # T3 = arr[idx]
                    self.emit_add(self.T3, src)        # T3 += val
                    self.emit_store(self.T3, self.T2)  # arr[idx] = T3
                elif assign.op == '-=':
                    src = self.compile_expr(assign.value, self.T0)
                    self.emit_load(self.T3, self.T2)
                    self.emit_sub(self.T3, src)
                    self.emit_store(self.T3, self.T2)
        else:
            raise SyntaxError("Invalid assignment target")

    def compile_if(self, stmt: IfStmt) -> None:
        else_label = self.new_label("else")
        end_label = self.new_label("endif")

        self.compile_condition(stmt.cond, else_label, invert=True)
        self.compile_block(stmt.then_body)
        if stmt.else_body:
            self.emit_jmp(end_label)
        self.emit_label(else_label)
        if stmt.else_body:
            self.compile_block(stmt.else_body)
            self.emit_label(end_label)

    def compile_while(self, stmt: WhileStmt) -> None:
        loop_label = self.new_label("while")
        end_label = self.new_label("endwhile")

        self.emit_label(loop_label)
        self.compile_condition(stmt.cond, end_label, invert=True)
        self.compile_block(stmt.body)
        self.emit_jmp(loop_label)
        self.emit_label(end_label)

    def compile_for(self, stmt: ForStmt) -> None:
        if stmt.init:
            self.compile_stmt(stmt.init)

        loop_label = self.new_label("for")
        end_label = self.new_label("endfor")

        self.emit_label(loop_label)
        if stmt.cond:
            self.compile_condition(stmt.cond, end_label, invert=True)
        self.compile_block(stmt.body)
        if stmt.update:
            self.compile_stmt(stmt.update)
        self.emit_jmp(loop_label)
        self.emit_label(end_label)

    def compile_return(self, stmt: ReturnStmt) -> None:
        if stmt.value:
            src = self.compile_expr(stmt.value, self.T0)
            self.emit_mov(self.RETVAL, src)

    # ---- Condition compilation (for control flow) ----

    def compile_condition(self, expr: Expr, false_label: str, invert: bool = False) -> None:
        """Compile a condition and jump to false_label if the condition is false (or true if inverted)."""
        if isinstance(expr, BinaryOp):
            if expr.op in ('==', '!=', '<', '>', '<=', '>='):
                left = self.compile_expr(expr.left, self.T0)
                right = self.compile_expr(expr.right, self.T1)
                self._emit_comparison(expr.op, left, right, false_label, invert)
                return
            if expr.op == '&&':
                if invert:
                    # Jump to target when A&&B is FALSE (short-circuit)
                    # A false → jump; else B false → jump; else fall through
                    self.compile_condition(expr.left, false_label, invert=True)
                    self.compile_condition(expr.right, false_label, invert=True)
                else:
                    # Jump to target when A&&B is TRUE
                    # A false → skip; A true and B true → jump
                    skip = self.new_label("and_skip")
                    self.compile_condition(expr.left, skip, invert=True)
                    self.compile_condition(expr.right, false_label, invert=False)
                    self.emit_label(skip)
                return
            if expr.op == '||':
                if invert:
                    # Jump to target when A||B is FALSE (both false)
                    # A true → skip; A false and B false → jump
                    skip = self.new_label("or_skip")
                    self.compile_condition(expr.left, skip, invert=False)
                    self.compile_condition(expr.right, false_label, invert=True)
                    self.emit_label(skip)
                else:
                    # Jump to target when A||B is TRUE (short-circuit)
                    # A true → jump; else B true → jump; else fall through
                    self.compile_condition(expr.left, false_label, invert=False)
                    self.compile_condition(expr.right, false_label, invert=False)
                return

        if isinstance(expr, UnaryOp) and expr.op == '!':
            self.compile_condition(expr.expr, false_label, invert=not invert)
            return

        # General case: evaluate expression, test for zero/nonzero
        result = self.compile_expr(expr, self.T0)
        if invert:
            self.emit_jz(result, false_label)
        else:
            self.emit_jnz(result, false_label)

    def _emit_comparison(self, op: str, left: int, right: int, false_label: str, invert: bool) -> None:
        """Emit comparison jump. Jumps to false_label if condition is false (or true if inverted).

        Strength reduction: a > b ≡ b < a, a <= b ≡ b >= a.
        This eliminates the expensive 3-instruction patterns for > and <=
        by swapping operands and reusing the cheaper < and >= paths.
        """
        # Strength reduction: swap operands to convert > and <= to < and >=
        if op == '>':
            op = '<'
            left, right = right, left
        elif op == '<=':
            op = '>='
            left, right = right, left

        # Compute diff = left - right in T2
        self.emit_mov(self.T2, left)
        self.emit_sub(self.T2, right)

        if op == '==' and invert:
            self.emit_jnz(self.T2, false_label)
        elif op == '==' and not invert:
            self.emit_jz(self.T2, false_label)
        elif op == '!=' and invert:
            self.emit_jz(self.T2, false_label)
        elif op == '!=' and not invert:
            self.emit_jnz(self.T2, false_label)
        elif op == '<' and invert:
            # Jump if diff >= 0. CMP jumps if < 0, so use CMP to skip past JMP.
            skip = self.new_label("lt_skip")
            self.emit_cmp(self.T2, skip)
            self.emit_jmp(false_label)
            self.emit_label(skip)
        elif op == '<' and not invert:
            # Jump if diff < 0 => direct CMP
            self.emit_cmp(self.T2, false_label)
        elif op == '>=' and invert:
            # Jump if diff < 0 => direct CMP
            self.emit_cmp(self.T2, false_label)
        elif op == '>=' and not invert:
            # Jump if diff >= 0. CMP jumps if < 0, so skip past JMP.
            skip = self.new_label("ge_skip")
            self.emit_cmp(self.T2, skip)
            self.emit_jmp(false_label)
            self.emit_label(skip)

    # ---- Expression compilation ----

    def compile_expr(self, expr: Expr, dest: int) -> int:
        """Compile expression, result goes to dest slot. Returns the slot containing the result."""
        if isinstance(expr, IntLiteral):
            addr = self.mem.alloc_constant(expr.value)
            return addr  # Caller can use directly, no need to copy

        if isinstance(expr, VarRef):
            return self.mem.get_var(expr.name)

        if isinstance(expr, ArrayRef):
            base = self.mem.get_var(expr.name)
            if isinstance(expr.index, IntLiteral):
                return base + expr.index.value
            # Variable index: O(1) indirect read via LOAD
            idx_addr = self.compile_expr(expr.index, self.T1)
            base_const = self.mem.alloc_constant(base)
            self.emit_mov(self.T2, base_const)
            self.emit_add(self.T2, idx_addr)
            self.emit_load(dest, self.T2)
            return dest

        if isinstance(expr, UnaryOp):
            if expr.op == '-':
                src = self.compile_expr(expr.expr, dest)
                # Negate: dest = 0 - src
                self.emit_mov(dest, self.CONST_ZERO)
                self.emit_sub(dest, src)
                return dest
            if expr.op == '~':
                src = self.compile_expr(expr.expr, dest)
                if src != dest:
                    self.emit_mov(dest, src)
                # XOR with all-ones (-1 in two's complement)
                all_ones = self.mem.alloc_constant(-1)
                self.emit_xor(dest, all_ones)
                return dest
            if expr.op == '!':
                # Logical NOT: result is 1 if expr==0, else 0
                src = self.compile_expr(expr.expr, dest)
                nonzero_label = self.new_label("not_nz")
                end_label = self.new_label("not_end")
                self.emit_jnz(src, nonzero_label)
                self.emit_mov(dest, self.CONST_ONE)
                self.emit_jmp(end_label)
                self.emit_label(nonzero_label)
                self.emit_mov(dest, self.CONST_ZERO)
                self.emit_label(end_label)
                return dest

        if isinstance(expr, BinaryOp):
            # Constant folding: evaluate at compile time when both operands are literals
            if isinstance(expr.left, IntLiteral) and isinstance(expr.right, IntLiteral):
                a, b = expr.left.value, expr.right.value
                fold_ops = {
                    '+': lambda: a + b, '-': lambda: a - b,
                    '&': lambda: a & b, '|': lambda: a | b, '^': lambda: a ^ b,
                    '<<': lambda: a << b, '>>': lambda: a >> b,
                    '==': lambda: int(a == b), '!=': lambda: int(a != b),
                    '<': lambda: int(a < b), '>': lambda: int(a > b),
                    '<=': lambda: int(a <= b), '>=': lambda: int(a >= b),
                }
                if expr.op in fold_ops:
                    return self.compile_expr(IntLiteral(fold_ops[expr.op]()), dest)

            # For comparisons used as values (not in conditions), produce 0/1
            if expr.op in ('==', '!=', '<', '>', '<=', '>='):
                true_label = self.new_label("cmp_true")
                end_label = self.new_label("cmp_end")
                left = self.compile_expr(expr.left, self.T0)
                right = self.compile_expr(expr.right, self.T1)
                self._emit_comparison(expr.op, left, right, true_label, invert=False)
                # Condition was false (didn't jump)
                self.emit_mov(dest, self.CONST_ZERO)
                self.emit_jmp(end_label)
                self.emit_label(true_label)
                self.emit_mov(dest, self.CONST_ONE)
                self.emit_label(end_label)
                return dest

            if expr.op in ('&&', '||'):
                # Produce 0/1 value
                true_label = self.new_label("log_true")
                end_label = self.new_label("log_end")
                self.compile_condition(expr, true_label, invert=False)
                # false path
                self.emit_mov(dest, self.CONST_ZERO)
                self.emit_jmp(end_label)
                self.emit_label(true_label)
                self.emit_mov(dest, self.CONST_ONE)
                self.emit_label(end_label)
                return dest

            # Arithmetic / bitwise
            left = self.compile_expr(expr.left, dest)
            # Use a different temp for right to avoid clobbering
            right_temp = self.T1 if dest != self.T1 else self.T3
            right = self.compile_expr(expr.right, right_temp)

            if left != dest:
                self.emit_mov(dest, left)

            op_map = {
                '+': self.emit_add,
                '-': self.emit_sub,
                '&': self.emit_and,
                '|': self.emit_or,
                '^': self.emit_xor,
            }
            if expr.op in op_map:
                op_map[expr.op](dest, right)
                return dest

            # Shift operators: ISA shifts by 1, so emit N shift instructions
            if expr.op in ('<<', '>>'):
                shift_fn = self.emit_shl if expr.op == '<<' else self.emit_shr
                if isinstance(expr.right, IntLiteral):
                    for _ in range(expr.right.value):
                        shift_fn(dest)
                else:
                    # Variable shift: loop
                    counter = self.T2 if dest != self.T2 else self.T3
                    self.emit_mov(counter, right)
                    loop = self.new_label('shl_loop' if expr.op == '<<' else 'shr_loop')
                    end = self.new_label('shift_end')
                    self.emit_label(loop)
                    self.emit_jz(counter, end)
                    shift_fn(dest)
                    self.emit_dec(counter)
                    self.emit_jmp(loop)
                    self.emit_label(end)
                return dest

            raise SyntaxError(f"Unsupported binary operator: {expr.op}")

        if isinstance(expr, FuncCall):
            return self.compile_func_call(expr, dest)

        raise SyntaxError(f"Unknown expression type: {type(expr)}")

    def compile_func_call(self, call: FuncCall, dest: int) -> int:
        if call.name == 'printf' or call.name == 'output':
            # Special: write first arg to output slot
            if call.args:
                src = self.compile_expr(call.args[0], self.T0)
                self.emit_mov(self.OUTPUT, src)
            return self.OUTPUT

        # Built-in: abs(x) — 4 instructions using CMOV
        if call.name == 'abs':
            if len(call.args) != 1:
                raise SyntaxError("abs() takes exactly 1 argument")
            src = self.compile_expr(call.args[0], dest)
            if src != dest:
                self.emit_mov(dest, src)
            # neg = -dest; CMOV dest, neg (if dest < 0, dest = -dest)
            neg = self.mem.alloc_var('__abs_neg')
            self.emit_mov(neg, self.CONST_ZERO)
            self.emit_sub(neg, dest)
            self.emit_cmov(dest, neg)
            return dest

        # Built-in: min(a, b) — 4 instructions using CMP
        if call.name == 'min':
            if len(call.args) != 2:
                raise SyntaxError("min() takes exactly 2 arguments")
            a = self.compile_expr(call.args[0], dest)
            b = self.compile_expr(call.args[1], self.T1)
            if a != dest:
                self.emit_mov(dest, a)
            # T2 = a - b; CMP T2, skip (if a < b, skip MOV); MOV dest, b; skip:
            self.emit_mov(self.T2, dest)
            self.emit_sub(self.T2, b)
            skip = self.new_label("min_skip")
            self.emit_cmp(self.T2, skip)  # if a-b < 0, a < b, keep a
            self.emit_mov(dest, b)        # a >= b, use b
            self.emit_label(skip)
            return dest

        # Built-in: max(a, b) — 4 instructions using CMP
        if call.name == 'max':
            if len(call.args) != 2:
                raise SyntaxError("max() takes exactly 2 arguments")
            a = self.compile_expr(call.args[0], dest)
            b = self.compile_expr(call.args[1], self.T1)
            if a != dest:
                self.emit_mov(dest, a)
            # T2 = b - a; CMP T2, skip (if b < a, a > b, keep a); MOV dest, b; skip:
            self.emit_mov(self.T2, b)
            self.emit_sub(self.T2, dest)
            skip = self.new_label("max_skip")
            self.emit_cmp(self.T2, skip)  # if b-a < 0, a > b, keep a
            self.emit_mov(dest, b)        # a <= b, use b
            self.emit_label(skip)
            return dest

        # Built-in: mul(a, b) — signed N-bit multiply using MULACC
        # MULACC does MSB-first shift-and-add in a single accumulator,
        # which only works for non-negative multipliers. So we:
        # 1. Save sign info (XOR of originals → MSB=1 if signs differ)
        # 2. Take abs of both operands
        # 3. Run N MULACC steps (unsigned multiply)
        # 4. Negate result if signs differed
        if call.name == 'mul':
            if len(call.args) != 2:
                raise SyntaxError("mul() takes exactly 2 arguments")
            a = self.compile_expr(call.args[0], dest)
            b = self.compile_expr(call.args[1], self.T1)
            if a != dest:
                self.emit_mov(dest, a)
            b_slot = self.T1 if b == self.T1 else b
            if b_slot == dest:
                raise SyntaxError("mul() internal error: operand collision")

            neg = self.mem.alloc_var('__mul_neg')

            # T2 = sign flag: XOR of originals (MSB=1 iff signs differ)
            self.emit_mov(self.T2, dest)
            self.emit_xor(self.T2, b_slot)

            # abs(a) -> dest
            self.emit_mov(neg, self.CONST_ZERO)
            self.emit_sub(neg, dest)
            self.emit_cmov(dest, neg)

            # abs(b) -> T3
            self.emit_mov(self.T3, b_slot)
            self.emit_mov(neg, self.CONST_ZERO)
            self.emit_sub(neg, self.T3)
            self.emit_cmov(self.T3, neg)

            # N MULACC steps: dest = abs(a) * abs(b)
            for _ in range(self.cfg.N):
                self.emit_mulacc(dest, self.T3)

            # Conditionally negate result if signs differed
            self.emit_mov(neg, self.CONST_ZERO)
            self.emit_sub(neg, dest)       # neg = -result
            neg_label = self.new_label("mul_neg")
            end_label = self.new_label("mul_end")
            self.emit_cmp(self.T2, neg_label)  # if T2 < 0, signs differ
            self.emit_jmp(end_label)            # signs same, skip
            self.emit_label(neg_label)
            self.emit_mov(dest, neg)            # dest = -result
            self.emit_label(end_label)
            return dest

        # Built-in: swap(a, b) — swap two variables in a single instruction
        if call.name == 'swap':
            if len(call.args) != 2:
                raise SyntaxError("swap() takes exactly 2 arguments")
            a = self.compile_expr(call.args[0], dest)
            b = self.compile_expr(call.args[1], self.T1)
            self.emit_swap(a, b)
            return dest

        # Inline the function
        if call.name not in self.functions:
            raise NameError(f"Undefined function '{call.name}'")
        func = self.functions[call.name]
        if len(call.args) != len(func.params):
            raise SyntaxError(f"Function '{call.name}' expects {len(func.params)} args, got {len(call.args)}")

        # Evaluate args and assign to parameter slots
        for i, (param, arg) in enumerate(zip(func.params, call.args)):
            src = self.compile_expr(arg, self.T0)
            param_addr = self.mem.alloc_var(f"__{call.name}_{param}")
            if src != param_addr:
                self.emit_mov(param_addr, src)

        # Save and remap parameters
        saved_slots = {}
        for param in func.params:
            internal_name = f"__{call.name}_{param}"
            saved_slots[param] = self.mem.slots.get(param)
            self.mem.slots[param] = self.mem.slots[internal_name]

        # Compile function body inline
        self.compile_block(func.body)

        # Restore parameter mappings
        for param in func.params:
            if saved_slots[param] is not None:
                self.mem.slots[param] = saved_slots[param]
            else:
                del self.mem.slots[param]

        # Return value is in RETVAL
        return self.RETVAL


# ============================================================
# Public API
# ============================================================

def compile_c(
    source: str,
    s: int = 32,
    m: int = 64,
    n: int = 1024,
    N: int = 8,
) -> Tuple[ExtendedConfigV4, List[int], List[Tuple[int, int, int]], Dict]:
    """
    Compile C source to Extended ISA V4 instructions.

    Returns:
        cfg: ExtendedConfigV4 configuration
        memory: initial memory contents (list of m ints)
        commands: list of (a, b, c) instruction triples
        meta: dict with variable addresses and debug info
    """
    cfg = ExtendedConfigV4(s=s, m=m, n=n, N=N)

    # Parse
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse_program()

    # Generate code
    codegen = CodeGen(cfg)
    codegen.compile_program(program)

    # Resolve labels and get outputs
    memory = codegen.mem.init_memory()
    commands = codegen.resolve()

    # Build metadata
    meta = {
        'variables': {name: slot.addr for name, slot in codegen.mem.slots.items()
                      if not name.startswith('__')},
        'constants': codegen.mem.constants.copy(),
        'num_instructions': len(commands),
        'retval_addr': codegen.RETVAL,
        'output_addr': codegen.OUTPUT,
        'labels': codegen.labels.copy(),
        'source_map': codegen.source_map,  # [(instr_index, source_line), ...]
    }

    return cfg, memory, commands, meta


# ============================================================
# Convenience: compile and run
# ============================================================

def compile_and_run(
    source: str,
    max_steps: int = 50000,
    max_seconds: float = 300.0,
    trace: bool = False,
    device: str = 'cpu',
    **kwargs,
) -> Dict[str, int]:
    """Compile C source and run it on the neural computer. Returns variable values."""
    import time
    import torch

    cfg, memory, commands, meta = compile_c(source, **kwargs)

    comp = ExtendedNeuralComputerV4(cfg)
    X = init_state_v4(cfg, memory, commands)

    if device != 'cpu':
        dev = torch.device(device)
        X = X.to(dev)
        from snake_game.snake_extended import move_computer_to_device
        move_computer_to_device(comp, dev)

    steps = 0
    start = time.monotonic()
    with torch.no_grad():
        while True:
            if steps >= max_steps:
                print(f"WARNING: exceeded max_steps={max_steps}")
                break
            if time.monotonic() - start >= max_seconds:
                print(f"WARNING: exceeded max_seconds={max_seconds}")
                break
            pc = get_pc_v4(X if device == 'cpu' else X.cpu(), cfg)
            if pc == 0:
                break
            if trace and steps < 200:
                print(f"  step {steps}: PC={pc}")
            X = comp.step(X)
            steps += 1

    elapsed = time.monotonic() - start
    result_mem = read_memory_v4(X if device == 'cpu' else X.cpu(), cfg)

    results = {}
    for name, addr in meta['variables'].items():
        results[name] = result_mem[addr]
    results['__retval'] = result_mem[meta['retval_addr']]
    results['__output'] = result_mem[meta['output_addr']]
    results['__steps'] = steps
    results['__elapsed'] = round(elapsed, 3)
    results['__num_instructions'] = meta['num_instructions']

    return results


# ============================================================
# Demo / self-test
# ============================================================

def test_compiler():
    """Run a suite of test programs to verify the compiler."""
    print("=" * 60)
    print("C Compiler for Neural SUBLEQ Computer - Test Suite")
    print("=" * 60)

    tests = [
        (
            "arithmetic",
            """
            int main() {
                int a = 3;
                int b = 5;
                int c = a + b;
                int d = c - a;
                return c;
            }
            """,
            {'a': 3, 'b': 5, 'c': 8, 'd': 5, '__retval': 8},
        ),
        (
            "if_else",
            """
            int main() {
                int x = 10;
                int y = 0;
                if (x == 10) {
                    y = 1;
                } else {
                    y = 2;
                }
                return y;
            }
            """,
            {'x': 10, 'y': 1, '__retval': 1},
        ),
        (
            "while_loop",
            """
            int main() {
                int i = 0;
                int sum = 0;
                while (i < 5) {
                    sum = sum + i;
                    i = i + 1;
                }
                return sum;
            }
            """,
            {'sum': 10, 'i': 5, '__retval': 10},
        ),
        (
            "for_loop",
            """
            int main() {
                int result = 1;
                int i;
                for (i = 0; i < 4; i += 1) {
                    result = result + result;
                }
                return result;
            }
            """,
            {'result': 16, '__retval': 16},
        ),
        (
            "nested_if",
            """
            int main() {
                int a = 7;
                int b = 3;
                int max = 0;
                if (a > b) {
                    max = a;
                } else {
                    max = b;
                }
                return max;
            }
            """,
            {'max': 7, '__retval': 7},
        ),
        (
            "fibonacci",
            """
            int main() {
                int n = 7;
                int a = 0;
                int b = 1;
                int i = 0;
                int t;
                while (i < n) {
                    t = a + b;
                    a = b;
                    b = t;
                    i = i + 1;
                }
                return a;
            }
            """,
            {'a': 13, '__retval': 13},
        ),
        (
            "gcd",
            """
            int main() {
                int a = 48;
                int b = 18;
                int temp;
                while (b != 0) {
                    // a = a mod b via repeated subtraction
                    while (a >= b) {
                        a = a - b;
                    }
                    // swap a and b
                    temp = a;
                    a = b;
                    b = temp;
                }
                return a;
            }
            """,
            {'a': 6, '__retval': 6},
        ),
        (
            "comparison_operators",
            """
            int main() {
                int a = 5;
                int b = 10;
                int r1 = 0;
                int r2 = 0;
                int r3 = 0;
                int r4 = 0;
                if (a < b) { r1 = 1; }
                if (a > b) { r2 = 1; }
                if (a <= 5) { r3 = 1; }
                if (b >= 10) { r4 = 1; }
                return r1;
            }
            """,
            {'r1': 1, 'r2': 0, 'r3': 1, 'r4': 1},
        ),
        (
            "bitwise",
            """
            int main() {
                int a = 15;
                int b = 6;
                int r_and = a & b;
                int r_or  = a | b;
                int r_xor = a ^ b;
                return r_and;
            }
            """,
            {'r_and': 6, 'r_or': 15, 'r_xor': 9},
        ),
    ]

    passed = 0
    failed = 0
    for name, source, expected in tests:
        try:
            results = compile_and_run(source, max_steps=50000, max_seconds=300.0)
            ok = True
            for var, val in expected.items():
                if var in results and results[var] != val:
                    print(f"  FAIL {name}: {var} = {results[var]}, expected {val}")
                    ok = False
            if ok:
                steps = results.get('__steps', '?')
                n_inst = results.get('__num_instructions', '?')
                print(f"  PASS {name} ({n_inst} instructions, {steps} steps)")
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    test_compiler()
