/**
 * C-to-Extended-ISA Compiler (JavaScript port)
 *
 * Compiles a subset of C to Extended ISA V4 instructions that run on the
 * neural SUBLEQ transformer computer or the ISA interpreter.
 *
 * Ported from c_compiler.py — identical semantics.
 */

// ============================================================
// Opcodes
// ============================================================

const OP_HALT = 0, OP_MOV = 1, OP_ADD = 2, OP_JMP = 3, OP_JZ = 4,
      OP_JNZ = 5, OP_INC = 6, OP_DEC = 7, OP_SHL = 8, OP_SHR = 9,
      OP_CMP = 10, OP_LOAD = 11, OP_AND = 12, OP_OR = 13, OP_XOR = 14,
      OP_SUB = 15, OP_FIND = 16, OP_SWAP = 17, OP_CMOV = 18, OP_MULACC = 19, OP_STORE = 20;

// ============================================================
// Token Types
// ============================================================

const TT = {
    INT: 1, IF: 2, ELSE: 3, WHILE: 4, FOR: 5, RETURN: 6, VOID: 7,
    IDENT: 8, NUMBER: 9,
    PLUS: 10, MINUS: 11, AMP: 12, PIPE: 13, CARET: 14, TILDE: 15, BANG: 16,
    LSHIFT: 17, RSHIFT: 18,
    EQ: 19, NEQ: 20, LT: 21, GT: 22, LE: 23, GE: 24, AND: 25, OR: 26,
    ASSIGN: 27, PLUS_ASSIGN: 28, MINUS_ASSIGN: 29,
    SEMI: 30, COMMA: 31, LPAREN: 32, RPAREN: 33,
    LBRACE: 34, RBRACE: 35, LBRACKET: 36, RBRACKET: 37,
    EOF: 38,
};

const KEYWORDS = {
    'int': TT.INT, 'if': TT.IF, 'else': TT.ELSE,
    'while': TT.WHILE, 'for': TT.FOR, 'return': TT.RETURN,
    'void': TT.VOID,
};

const OPERATORS = [
    ['==', TT.EQ], ['!=', TT.NEQ], ['<=', TT.LE], ['>=', TT.GE],
    ['<<', TT.LSHIFT], ['>>', TT.RSHIFT],
    ['&&', TT.AND], ['||', TT.OR],
    ['+=', TT.PLUS_ASSIGN], ['-=', TT.MINUS_ASSIGN],
    ['+', TT.PLUS], ['-', TT.MINUS],
    ['&', TT.AMP], ['|', TT.PIPE], ['^', TT.CARET],
    ['~', TT.TILDE], ['!', TT.BANG],
    ['<', TT.LT], ['>', TT.GT], ['=', TT.ASSIGN],
    [';', TT.SEMI], [',', TT.COMMA],
    ['(', TT.LPAREN], [')', TT.RPAREN],
    ['{', TT.LBRACE], ['}', TT.RBRACE],
    ['[', TT.LBRACKET], [']', TT.RBRACKET],
];

// ============================================================
// Lexer
// ============================================================

function lex(source) {
    const tokens = [];
    let i = 0, line = 1;
    const len = source.length;

    const ASSIGN_OPS = new Set([
        TT.LPAREN, TT.COMMA, TT.ASSIGN, TT.PLUS_ASSIGN,
        TT.MINUS_ASSIGN, TT.SEMI, TT.LBRACKET,
        TT.PLUS, TT.MINUS, TT.EQ, TT.NEQ, TT.LT, TT.GT,
        TT.LE, TT.GE, TT.AND, TT.OR, TT.RETURN,
        TT.AMP, TT.PIPE, TT.CARET, TT.BANG, TT.TILDE,
    ]);

    while (i < len) {
        const c = source[i];
        if (c === ' ' || c === '\t' || c === '\r') { i++; continue; }
        if (c === '\n') { line++; i++; continue; }
        // Line comment
        if (i + 1 < len && source[i] === '/' && source[i + 1] === '/') {
            while (i < len && source[i] !== '\n') i++;
            continue;
        }
        // Block comment
        if (i + 1 < len && source[i] === '/' && source[i + 1] === '*') {
            i += 2;
            while (i + 1 < len && !(source[i] === '*' && source[i + 1] === '/')) {
                if (source[i] === '\n') line++;
                i++;
            }
            i += 2;
            continue;
        }
        // Number (including negative literals in certain contexts)
        if (c >= '0' && c <= '9' ||
            (c === '-' && i + 1 < len && source[i + 1] >= '0' && source[i + 1] <= '9' &&
             (tokens.length === 0 || ASSIGN_OPS.has(tokens[tokens.length - 1].type)))) {
            let j = i;
            if (source[j] === '-') j++;
            while (j < len && source[j] >= '0' && source[j] <= '9') j++;
            tokens.push({ type: TT.NUMBER, value: source.slice(i, j), line });
            i = j;
            continue;
        }
        // Identifier / keyword
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_') {
            let j = i;
            while (j < len && ((source[j] >= 'a' && source[j] <= 'z') ||
                               (source[j] >= 'A' && source[j] <= 'Z') ||
                               (source[j] >= '0' && source[j] <= '9') ||
                               source[j] === '_')) j++;
            const word = source.slice(i, j);
            tokens.push({ type: KEYWORDS[word] || TT.IDENT, value: word, line });
            i = j;
            continue;
        }
        // Operators / punctuation
        let matched = false;
        for (const [op, tt] of OPERATORS) {
            if (source.startsWith(op, i)) {
                tokens.push({ type: tt, value: op, line });
                i += op.length;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        throw new SyntaxError(`Unexpected character '${c}' at line ${line}`);
    }
    tokens.push({ type: TT.EOF, value: '', line });
    return tokens;
}

// ============================================================
// Parser (recursive descent)
// ============================================================

class Parser {
    constructor(tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }
    peek() { return this.tokens[this.pos]; }
    advance() { return this.tokens[this.pos++]; }
    expect(tt) {
        const t = this.advance();
        if (t.type !== tt) throw new SyntaxError(`Expected ${tt}, got '${t.value}' at line ${t.line}`);
        return t;
    }
    match(tt) {
        if (this.peek().type === tt) return this.advance();
        return null;
    }

    parseProgram() {
        const functions = [], globals = [];
        while (this.peek().type !== TT.EOF) {
            if (this.peek().type === TT.INT || this.peek().type === TT.VOID) {
                const saved = this.pos;
                this.advance();
                this.advance();
                if (this.peek().type === TT.LPAREN) {
                    this.pos = saved;
                    functions.push(this.parseFuncDecl());
                } else {
                    this.pos = saved;
                    globals.push(this.parseVarDecl());
                }
            } else {
                throw new SyntaxError(`Unexpected token '${this.peek().value}' at line ${this.peek().line}`);
            }
        }
        return { kind: 'Program', functions, globals };
    }

    parseFuncDecl() {
        this.advance(); // int/void
        const name = this.expect(TT.IDENT).value;
        this.expect(TT.LPAREN);
        const params = [];
        if (this.peek().type !== TT.RPAREN) {
            this.expect(TT.INT);
            params.push(this.expect(TT.IDENT).value);
            while (this.match(TT.COMMA)) {
                this.expect(TT.INT);
                params.push(this.expect(TT.IDENT).value);
            }
        }
        this.expect(TT.RPAREN);
        const body = this.parseBlock();
        return { kind: 'FuncDecl', name, params, body };
    }

    parseBlock() {
        this.expect(TT.LBRACE);
        const stmts = [];
        while (this.peek().type !== TT.RBRACE) stmts.push(this.parseStmt());
        this.expect(TT.RBRACE);
        return { kind: 'Block', stmts };
    }

    parseStmt() {
        const t = this.peek();
        if (t.type === TT.INT) return this.parseVarDecl();
        if (t.type === TT.IF) return this.parseIf();
        if (t.type === TT.WHILE) return this.parseWhile();
        if (t.type === TT.FOR) return this.parseFor();
        if (t.type === TT.RETURN) return this.parseReturn();
        if (t.type === TT.LBRACE) return this.parseBlock();
        return this.parseAssignOrExpr();
    }

    parseVarDecl() {
        const ln = this.peek().line;
        this.expect(TT.INT);
        const name = this.expect(TT.IDENT).value;
        let size = null, init = null;
        if (this.match(TT.LBRACKET)) {
            size = parseInt(this.expect(TT.NUMBER).value);
            this.expect(TT.RBRACKET);
        }
        if (this.match(TT.ASSIGN)) init = this.parseExpr();
        this.expect(TT.SEMI);
        return { kind: 'VarDecl', name, size, init, line: ln };
    }

    parseIf() {
        const ln = this.peek().line;
        this.expect(TT.IF);
        this.expect(TT.LPAREN);
        const cond = this.parseExpr();
        this.expect(TT.RPAREN);
        const thenBody = this._parseBody();
        let elseBody = null;
        if (this.match(TT.ELSE)) elseBody = this._parseBody();
        return { kind: 'IfStmt', cond, thenBody, elseBody, line: ln };
    }

    parseWhile() {
        const ln = this.peek().line;
        this.expect(TT.WHILE);
        this.expect(TT.LPAREN);
        const cond = this.parseExpr();
        this.expect(TT.RPAREN);
        const body = this._parseBody();
        return { kind: 'WhileStmt', cond, body, line: ln };
    }

    parseFor() {
        const ln = this.peek().line;
        this.expect(TT.FOR);
        this.expect(TT.LPAREN);
        let init = null;
        if (this.peek().type !== TT.SEMI) init = this.parseStmt();
        else this.expect(TT.SEMI);
        let cond = null;
        if (this.peek().type !== TT.SEMI) cond = this.parseExpr();
        this.expect(TT.SEMI);
        let update = null;
        if (this.peek().type !== TT.RPAREN) update = this._parseAssignOrExprNoSemi();
        this.expect(TT.RPAREN);
        const body = this._parseBody();
        return { kind: 'ForStmt', init, cond, update, body, line: ln };
    }

    parseReturn() {
        const ln = this.peek().line;
        this.expect(TT.RETURN);
        let value = null;
        if (this.peek().type !== TT.SEMI) value = this.parseExpr();
        this.expect(TT.SEMI);
        return { kind: 'ReturnStmt', value, line: ln };
    }

    parseAssignOrExpr() {
        const stmt = this._parseAssignOrExprNoSemi();
        this.expect(TT.SEMI);
        return stmt;
    }

    _parseAssignOrExprNoSemi() {
        const ln = this.peek().line;
        const expr = this.parseExpr();
        if (this.peek().type === TT.ASSIGN || this.peek().type === TT.PLUS_ASSIGN || this.peek().type === TT.MINUS_ASSIGN) {
            const op = this.advance().value;
            const value = this.parseExpr();
            if (expr.kind === 'VarRef' || expr.kind === 'ArrayRef')
                return { kind: 'Assignment', target: expr, op, value, line: ln };
            throw new SyntaxError('Invalid assignment target');
        }
        return { kind: 'ExprStmt', expr, line: ln };
    }

    _parseBody() {
        if (this.peek().type === TT.LBRACE) return this.parseBlock();
        return { kind: 'Block', stmts: [this.parseStmt()] };
    }

    // ---- Expression parsing (precedence climbing) ----
    parseExpr() { return this.parseOr(); }

    parseOr() {
        let left = this.parseAnd();
        while (this.peek().type === TT.OR) { this.advance(); left = { kind: 'BinaryOp', op: '||', left, right: this.parseAnd() }; }
        return left;
    }
    parseAnd() {
        let left = this.parseBitOr();
        while (this.peek().type === TT.AND) { this.advance(); left = { kind: 'BinaryOp', op: '&&', left, right: this.parseBitOr() }; }
        return left;
    }
    parseBitOr() {
        let left = this.parseBitXor();
        while (this.peek().type === TT.PIPE) { this.advance(); left = { kind: 'BinaryOp', op: '|', left, right: this.parseBitXor() }; }
        return left;
    }
    parseBitXor() {
        let left = this.parseBitAnd();
        while (this.peek().type === TT.CARET) { this.advance(); left = { kind: 'BinaryOp', op: '^', left, right: this.parseBitAnd() }; }
        return left;
    }
    parseBitAnd() {
        let left = this.parseEquality();
        while (this.peek().type === TT.AMP) { this.advance(); left = { kind: 'BinaryOp', op: '&', left, right: this.parseEquality() }; }
        return left;
    }
    parseEquality() {
        let left = this.parseComparison();
        while (this.peek().type === TT.EQ || this.peek().type === TT.NEQ) {
            const op = this.advance().value;
            left = { kind: 'BinaryOp', op, left, right: this.parseComparison() };
        }
        return left;
    }
    parseComparison() {
        let left = this.parseShift();
        while (this.peek().type === TT.LT || this.peek().type === TT.GT ||
               this.peek().type === TT.LE || this.peek().type === TT.GE) {
            const op = this.advance().value;
            left = { kind: 'BinaryOp', op, left, right: this.parseShift() };
        }
        return left;
    }
    parseShift() {
        let left = this.parseAdditive();
        while (this.peek().type === TT.LSHIFT || this.peek().type === TT.RSHIFT) {
            const op = this.advance().value;
            left = { kind: 'BinaryOp', op, left, right: this.parseAdditive() };
        }
        return left;
    }
    parseAdditive() {
        let left = this.parseUnary();
        while (this.peek().type === TT.PLUS || this.peek().type === TT.MINUS) {
            const op = this.advance().value;
            left = { kind: 'BinaryOp', op, left, right: this.parseUnary() };
        }
        return left;
    }
    parseUnary() {
        if (this.peek().type === TT.MINUS) {
            this.advance();
            const expr = this.parseUnary();
            if (expr.kind === 'IntLiteral') return { kind: 'IntLiteral', value: -expr.value };
            return { kind: 'UnaryOp', op: '-', expr };
        }
        if (this.peek().type === TT.TILDE) { this.advance(); return { kind: 'UnaryOp', op: '~', expr: this.parseUnary() }; }
        if (this.peek().type === TT.BANG) { this.advance(); return { kind: 'UnaryOp', op: '!', expr: this.parseUnary() }; }
        return this.parsePrimary();
    }
    parsePrimary() {
        const t = this.peek();
        if (t.type === TT.NUMBER) { this.advance(); return { kind: 'IntLiteral', value: parseInt(t.value) }; }
        if (t.type === TT.IDENT) {
            this.advance();
            const name = t.value;
            if (this.peek().type === TT.LPAREN) {
                this.advance();
                const args = [];
                if (this.peek().type !== TT.RPAREN) {
                    args.push(this.parseExpr());
                    while (this.match(TT.COMMA)) args.push(this.parseExpr());
                }
                this.expect(TT.RPAREN);
                return { kind: 'FuncCall', name, args };
            }
            if (this.peek().type === TT.LBRACKET) {
                this.advance();
                const index = this.parseExpr();
                this.expect(TT.RBRACKET);
                return { kind: 'ArrayRef', name, index };
            }
            return { kind: 'VarRef', name };
        }
        if (t.type === TT.LPAREN) {
            this.advance();
            const expr = this.parseExpr();
            this.expect(TT.RPAREN);
            return expr;
        }
        throw new SyntaxError(`Unexpected token '${t.value}' at line ${t.line}`);
    }
}

// ============================================================
// Memory Allocator
// ============================================================

class MemoryAllocator {
    constructor(m, N) {
        this.m = m;
        this.N = N;
        this.slots = {};      // name -> { addr, name, size }
        this.constants = {};   // value -> addr
        this.nextAddr = 0;
    }

    _alloc(name, size = 1) {
        if (this.nextAddr + size > this.m)
            throw new Error(`Out of memory allocating '${name}' (need ${size}, ${this.nextAddr}/${this.m} used)`);
        const addr = this.nextAddr;
        this.slots[name] = { addr, name, size };
        this.nextAddr += size;
        return addr;
    }

    allocConstant(value) {
        const mask = (1 << this.N) - 1;
        const half = 1 << (this.N - 1);
        value = value & mask;
        if (value >= half) value -= (1 << this.N);
        if (value in this.constants) return this.constants[value];
        const addr = this._alloc(`__const_${value}`);
        this.constants[value] = addr;
        return addr;
    }

    allocTemp(name) { return this._alloc(name); }

    allocVar(name, size = 1) {
        if (name in this.slots) return this.slots[name].addr;
        return this._alloc(name, size);
    }

    getVar(name) {
        if (!(name in this.slots)) throw new Error(`Undefined variable '${name}'`);
        return this.slots[name].addr;
    }

    hasVar(name) { return name in this.slots; }

    initMemory() {
        const mem = new Array(this.m).fill(0);
        for (const [val, addr] of Object.entries(this.constants)) {
            mem[addr] = parseInt(val);
        }
        return mem;
    }
}

// ============================================================
// Config Builder
// ============================================================

function makeConfig(s, m, n, N) {
    const logn = Math.ceil(Math.log2(n));
    const nrows_cmds = 3 * logn;
    const nrows_memory = N;
    const nrows_scratchpad = 3 * logn + 2 * N;
    const nrows_pc = logn;
    const nrows_pos_enc = logn;
    const nrows_buffer = 3 * N + N + logn;
    const nrows_tag = N;
    const nrows_indicator = 1;

    const idx_memory = nrows_cmds;
    const idx_scratchpad = idx_memory + nrows_memory;
    const idx_pc = idx_scratchpad + nrows_scratchpad;
    const idx_pos_enc = idx_pc + nrows_pc;
    const idx_buffer = idx_pos_enc + nrows_pos_enc;
    const idx_tag = idx_buffer + nrows_buffer;

    const d_model = nrows_cmds + nrows_memory + nrows_scratchpad +
                    nrows_pc + nrows_pos_enc + nrows_buffer +
                    nrows_tag + nrows_indicator;

    return {
        s, m, n, N, logn, d_model,
        idx_memory, idx_scratchpad, idx_pc, idx_pos_enc, idx_buffer, idx_tag,
        nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc,
        nrows_pos_enc, nrows_buffer, nrows_tag, nrows_indicator,
    };
}

// ============================================================
// Code Generator
// ============================================================

class CodeGen {
    constructor(cfg) {
        this.cfg = cfg;
        this.mem = new MemoryAllocator(cfg.m, cfg.N);
        this.cmds = [];
        this.labels = {};
        this.labelCounter = 0;
        this.functions = {};
        this.sourceMap = [];

        this.CONST_ZERO = this.mem.allocConstant(0);
        this.CONST_ONE = this.mem.allocConstant(1);
        this.CONST_SIGN = this.mem.allocConstant(-(1 << (cfg.N - 1)));
        this.T0 = this.mem.allocTemp('__t0');
        this.T1 = this.mem.allocTemp('__t1');
        this.T2 = this.mem.allocTemp('__t2');
        this.T3 = this.mem.allocTemp('__t3');
        this.RETVAL = this.mem.allocTemp('__retval');
        this.OUTPUT = this.mem.allocTemp('__output');
    }

    newLabel(prefix = 'L') { return `__${prefix}_${++this.labelCounter}`; }
    addr(memIdx) { return this.cfg.s + memIdx; }

    emit(a, b, c) { this.cmds.push([a, b, c]); }
    emitLabel(name) { this.labels[name] = this.cmds.length; }

    emitMov(dest, src) { this.emit(OP_MOV, this.addr(dest), this.addr(src)); }
    emitAdd(dest, src) { this.emit(OP_ADD, this.addr(dest), this.addr(src)); }
    emitSub(dest, src) { this.emit(OP_SUB, this.addr(dest), this.addr(src)); }
    emitAnd(dest, src) { this.emit(OP_AND, this.addr(dest), this.addr(src)); }
    emitOr(dest, src) { this.emit(OP_OR, this.addr(dest), this.addr(src)); }
    emitXor(dest, src) { this.emit(OP_XOR, this.addr(dest), this.addr(src)); }
    emitShl(dest) { this.emit(OP_SHL, this.addr(dest), 0); }
    emitShr(dest) { this.emit(OP_SHR, this.addr(dest), 0); }
    emitInc(dest) { this.emit(OP_INC, this.addr(dest), 0); }
    emitDec(dest) { this.emit(OP_DEC, this.addr(dest), 0); }
    emitJz(memIdx, label) { this.emit(OP_JZ, this.addr(memIdx), label); }
    emitJnz(memIdx, label) { this.emit(OP_JNZ, this.addr(memIdx), label); }
    emitCmp(memIdx, label) { this.emit(OP_CMP, this.addr(memIdx), label); }
    emitCmov(dest, src) { this.emit(OP_CMOV, this.addr(dest), this.addr(src)); }
    emitMulacc(dest, src) { this.emit(OP_MULACC, this.addr(dest), this.addr(src)); }
    emitStore(src, pointer) { this.emit(OP_STORE, this.addr(src), this.addr(pointer)); }
    emitSwap(a, b) { this.emit(OP_SWAP, this.addr(a), this.addr(b)); }
    emitLoad(dest, ptr) { this.emit(OP_LOAD, this.addr(dest), this.addr(ptr)); }
    emitJmp(label) { this.emit(OP_JMP, 0, label); }
    emitHalt() { this.emit(OP_HALT, 0, 0); }

    resolve() {
        const cmdStart = this.cfg.s + this.cfg.m;
        for (const cmd of this.cmds) {
            if (typeof cmd[2] === 'string') {
                if (!(cmd[2] in this.labels)) throw new Error(`Undefined label '${cmd[2]}'`);
                cmd[2] = cmdStart + this.labels[cmd[2]];
            }
        }
        return this.cmds.map(c => [c[0], c[1], c[2]]);
    }

    // ---- Compilation ----

    compileProgram(prog) {
        for (const func of prog.functions) this.functions[func.name] = func;
        for (const g of prog.globals) this.compileVarDecl(g);
        if (!('main' in this.functions)) throw new SyntaxError("No main() function found");
        this.compileBlock(this.functions['main'].body);
        this.emitHalt();
    }

    compileBlock(block) { for (const stmt of block.stmts) this.compileStmt(stmt); }

    compileStmt(stmt) {
        const line = stmt.line || 0;
        if (line > 0) this.sourceMap.push([this.cmds.length, line]);

        switch (stmt.kind) {
            case 'VarDecl': this.compileVarDecl(stmt); break;
            case 'Assignment': this.compileAssignment(stmt); break;
            case 'IfStmt': this.compileIf(stmt); break;
            case 'WhileStmt': this.compileWhile(stmt); break;
            case 'ForStmt': this.compileFor(stmt); break;
            case 'ReturnStmt': this.compileReturn(stmt); break;
            case 'ExprStmt': this.compileExpr(stmt.expr, this.T0); break;
            case 'Block': this.compileBlock(stmt); break;
            default: throw new SyntaxError(`Unknown statement type: ${stmt.kind}`);
        }
    }

    compileVarDecl(decl) {
        if (decl.size != null) {
            this.mem.allocVar(decl.name, decl.size);
        } else {
            const addr = this.mem.allocVar(decl.name);
            if (decl.init != null) {
                const src = this.compileExpr(decl.init, this.T0);
                if (src !== addr) this.emitMov(addr, src);
            }
        }
    }

    compileAssignment(assign) {
        if (assign.target.kind === 'VarRef') {
            const dest = this.mem.getVar(assign.target.name);
            if (assign.op === '=') {
                const src = this.compileExpr(assign.value, this.T0);
                if (src !== dest) this.emitMov(dest, src);
            } else if (assign.op === '+=') {
                const src = this.compileExpr(assign.value, this.T0);
                this.emitAdd(dest, src);
            } else if (assign.op === '-=') {
                const src = this.compileExpr(assign.value, this.T0);
                this.emitSub(dest, src);
            }
        } else if (assign.target.kind === 'ArrayRef') {
            const base = this.mem.getVar(assign.target.name);
            const slot = this.mem.slots[assign.target.name];
            if (assign.target.index.kind === 'IntLiteral') {
                const dest = base + assign.target.index.value;
                if (assign.op === '=') {
                    const src = this.compileExpr(assign.value, this.T0);
                    if (src !== dest) this.emitMov(dest, src);
                } else if (assign.op === '+=') {
                    const src = this.compileExpr(assign.value, this.T0);
                    this.emitAdd(dest, src);
                } else if (assign.op === '-=') {
                    const src = this.compileExpr(assign.value, this.T0);
                    this.emitSub(dest, src);
                }
            } else {
                // Variable index: use STORE for indirect write
                const idxAddr = this.compileExpr(assign.target.index, this.T1);
                const baseConst = this.mem.allocConstant(base);
                this.emitMov(this.T2, baseConst);
                this.emitAdd(this.T2, idxAddr);
                if (assign.op === '=') {
                    const src = this.compileExpr(assign.value, this.T0);
                    this.emitStore(src, this.T2);
                } else if (assign.op === '+=') {
                    const src = this.compileExpr(assign.value, this.T0);
                    this.emitLoad(this.T3, this.T2);
                    this.emitAdd(this.T3, src);
                    this.emitStore(this.T3, this.T2);
                } else if (assign.op === '-=') {
                    const src = this.compileExpr(assign.value, this.T0);
                    this.emitLoad(this.T3, this.T2);
                    this.emitSub(this.T3, src);
                    this.emitStore(this.T3, this.T2);
                }
            }
        } else {
            throw new SyntaxError('Invalid assignment target');
        }
    }

    compileIf(stmt) {
        const elseLabel = this.newLabel('else');
        const endLabel = this.newLabel('endif');
        this.compileCondition(stmt.cond, elseLabel, true);
        this.compileBlock(stmt.thenBody);
        if (stmt.elseBody) this.emitJmp(endLabel);
        this.emitLabel(elseLabel);
        if (stmt.elseBody) {
            this.compileBlock(stmt.elseBody);
            this.emitLabel(endLabel);
        }
    }

    compileWhile(stmt) {
        const loopLabel = this.newLabel('while');
        const endLabel = this.newLabel('endwhile');
        this.emitLabel(loopLabel);
        this.compileCondition(stmt.cond, endLabel, true);
        this.compileBlock(stmt.body);
        this.emitJmp(loopLabel);
        this.emitLabel(endLabel);
    }

    compileFor(stmt) {
        if (stmt.init) this.compileStmt(stmt.init);
        const loopLabel = this.newLabel('for');
        const endLabel = this.newLabel('endfor');
        this.emitLabel(loopLabel);
        if (stmt.cond) this.compileCondition(stmt.cond, endLabel, true);
        this.compileBlock(stmt.body);
        if (stmt.update) this.compileStmt(stmt.update);
        this.emitJmp(loopLabel);
        this.emitLabel(endLabel);
    }

    compileReturn(stmt) {
        if (stmt.value) {
            const src = this.compileExpr(stmt.value, this.T0);
            this.emitMov(this.RETVAL, src);
        }
    }

    // ---- Condition compilation ----

    compileCondition(expr, falseLabel, invert = false) {
        if (expr.kind === 'BinaryOp') {
            if (['==', '!=', '<', '>', '<=', '>='].includes(expr.op)) {
                const left = this.compileExpr(expr.left, this.T0);
                const right = this.compileExpr(expr.right, this.T1);
                this._emitComparison(expr.op, left, right, falseLabel, invert);
                return;
            }
            if (expr.op === '&&') {
                if (invert) {
                    this.compileCondition(expr.left, falseLabel, true);
                    this.compileCondition(expr.right, falseLabel, true);
                } else {
                    const skip = this.newLabel('and_skip');
                    this.compileCondition(expr.left, skip, true);
                    this.compileCondition(expr.right, falseLabel, false);
                    this.emitLabel(skip);
                }
                return;
            }
            if (expr.op === '||') {
                if (invert) {
                    const skip = this.newLabel('or_skip');
                    this.compileCondition(expr.left, skip, false);
                    this.compileCondition(expr.right, falseLabel, true);
                    this.emitLabel(skip);
                } else {
                    this.compileCondition(expr.left, falseLabel, false);
                    this.compileCondition(expr.right, falseLabel, false);
                }
                return;
            }
        }

        if (expr.kind === 'UnaryOp' && expr.op === '!') {
            this.compileCondition(expr.expr, falseLabel, !invert);
            return;
        }

        const result = this.compileExpr(expr, this.T0);
        if (invert) this.emitJz(result, falseLabel);
        else this.emitJnz(result, falseLabel);
    }

    _emitComparison(op, left, right, falseLabel, invert) {
        // Strength reduction
        if (op === '>') { op = '<'; [left, right] = [right, left]; }
        else if (op === '<=') { op = '>='; [left, right] = [right, left]; }

        this.emitMov(this.T2, left);
        this.emitSub(this.T2, right);

        if (op === '==' && invert) this.emitJnz(this.T2, falseLabel);
        else if (op === '==' && !invert) this.emitJz(this.T2, falseLabel);
        else if (op === '!=' && invert) this.emitJz(this.T2, falseLabel);
        else if (op === '!=' && !invert) this.emitJnz(this.T2, falseLabel);
        else if (op === '<' && invert) {
            const skip = this.newLabel('lt_skip');
            this.emitCmp(this.T2, skip);
            this.emitJmp(falseLabel);
            this.emitLabel(skip);
        } else if (op === '<' && !invert) {
            this.emitCmp(this.T2, falseLabel);
        } else if (op === '>=' && invert) {
            this.emitCmp(this.T2, falseLabel);
        } else if (op === '>=' && !invert) {
            const skip = this.newLabel('ge_skip');
            this.emitCmp(this.T2, skip);
            this.emitJmp(falseLabel);
            this.emitLabel(skip);
        }
    }

    // ---- Expression compilation ----

    compileExpr(expr, dest) {
        if (expr.kind === 'IntLiteral') return this.mem.allocConstant(expr.value);
        if (expr.kind === 'VarRef') return this.mem.getVar(expr.name);

        if (expr.kind === 'ArrayRef') {
            const base = this.mem.getVar(expr.name);
            if (expr.index.kind === 'IntLiteral') return base + expr.index.value;
            const idxAddr = this.compileExpr(expr.index, this.T1);
            const baseConst = this.mem.allocConstant(base);
            this.emitMov(this.T2, baseConst);
            this.emitAdd(this.T2, idxAddr);
            this.emitLoad(dest, this.T2);
            return dest;
        }

        if (expr.kind === 'UnaryOp') {
            if (expr.op === '-') {
                const src = this.compileExpr(expr.expr, dest);
                this.emitMov(dest, this.CONST_ZERO);
                this.emitSub(dest, src);
                return dest;
            }
            if (expr.op === '~') {
                const src = this.compileExpr(expr.expr, dest);
                if (src !== dest) this.emitMov(dest, src);
                const allOnes = this.mem.allocConstant(-1);
                this.emitXor(dest, allOnes);
                return dest;
            }
            if (expr.op === '!') {
                const src = this.compileExpr(expr.expr, dest);
                const nzLabel = this.newLabel('not_nz');
                const endLabel = this.newLabel('not_end');
                this.emitJnz(src, nzLabel);
                this.emitMov(dest, this.CONST_ONE);
                this.emitJmp(endLabel);
                this.emitLabel(nzLabel);
                this.emitMov(dest, this.CONST_ZERO);
                this.emitLabel(endLabel);
                return dest;
            }
        }

        if (expr.kind === 'BinaryOp') {
            // Constant folding
            if (expr.left.kind === 'IntLiteral' && expr.right.kind === 'IntLiteral') {
                const a = expr.left.value, b = expr.right.value;
                const foldOps = {
                    '+': () => a + b, '-': () => a - b,
                    '&': () => a & b, '|': () => a | b, '^': () => a ^ b,
                    '<<': () => a << b, '>>': () => a >> b,
                    '==': () => +(a === b), '!=': () => +(a !== b),
                    '<': () => +(a < b), '>': () => +(a > b),
                    '<=': () => +(a <= b), '>=': () => +(a >= b),
                };
                if (expr.op in foldOps)
                    return this.compileExpr({ kind: 'IntLiteral', value: foldOps[expr.op]() }, dest);
            }

            // Comparisons as values
            if (['==', '!=', '<', '>', '<=', '>='].includes(expr.op)) {
                const trueLabel = this.newLabel('cmp_true');
                const endLabel = this.newLabel('cmp_end');
                const left = this.compileExpr(expr.left, this.T0);
                const right = this.compileExpr(expr.right, this.T1);
                this._emitComparison(expr.op, left, right, trueLabel, false);
                this.emitMov(dest, this.CONST_ZERO);
                this.emitJmp(endLabel);
                this.emitLabel(trueLabel);
                this.emitMov(dest, this.CONST_ONE);
                this.emitLabel(endLabel);
                return dest;
            }

            // Logical as values
            if (expr.op === '&&' || expr.op === '||') {
                const trueLabel = this.newLabel('log_true');
                const endLabel = this.newLabel('log_end');
                this.compileCondition(expr, trueLabel, false);
                this.emitMov(dest, this.CONST_ZERO);
                this.emitJmp(endLabel);
                this.emitLabel(trueLabel);
                this.emitMov(dest, this.CONST_ONE);
                this.emitLabel(endLabel);
                return dest;
            }

            // Arithmetic / bitwise
            const left = this.compileExpr(expr.left, dest);
            const rightTemp = dest !== this.T1 ? this.T1 : this.T3;
            const right = this.compileExpr(expr.right, rightTemp);
            if (left !== dest) this.emitMov(dest, left);

            const opMap = {
                '+': 'emitAdd', '-': 'emitSub',
                '&': 'emitAnd', '|': 'emitOr', '^': 'emitXor',
            };
            if (expr.op in opMap) {
                this[opMap[expr.op]](dest, right);
                return dest;
            }

            // Shift
            if (expr.op === '<<' || expr.op === '>>') {
                const shiftFn = expr.op === '<<' ? 'emitShl' : 'emitShr';
                if (expr.right.kind === 'IntLiteral') {
                    for (let i = 0; i < expr.right.value; i++) this[shiftFn](dest);
                } else {
                    const counter = dest !== this.T2 ? this.T2 : this.T3;
                    this.emitMov(counter, right);
                    const loop = this.newLabel(expr.op === '<<' ? 'shl_loop' : 'shr_loop');
                    const end = this.newLabel('shift_end');
                    this.emitLabel(loop);
                    this.emitJz(counter, end);
                    this[shiftFn](dest);
                    this.emitDec(counter);
                    this.emitJmp(loop);
                    this.emitLabel(end);
                }
                return dest;
            }

            throw new SyntaxError(`Unsupported binary operator: ${expr.op}`);
        }

        if (expr.kind === 'FuncCall') return this.compileFuncCall(expr, dest);

        throw new SyntaxError(`Unknown expression type: ${expr.kind}`);
    }

    compileFuncCall(call, dest) {
        if (call.name === 'printf' || call.name === 'output') {
            if (call.args.length > 0) {
                const src = this.compileExpr(call.args[0], this.T0);
                this.emitMov(this.OUTPUT, src);
            }
            return this.OUTPUT;
        }

        // Built-in: abs(x)
        if (call.name === 'abs') {
            if (call.args.length !== 1) throw new SyntaxError('abs() takes exactly 1 argument');
            const src = this.compileExpr(call.args[0], dest);
            if (src !== dest) this.emitMov(dest, src);
            const neg = this.mem.allocVar('__abs_neg');
            this.emitMov(neg, this.CONST_ZERO);
            this.emitSub(neg, dest);
            this.emitCmov(dest, neg);
            return dest;
        }

        // Built-in: min(a, b)
        if (call.name === 'min') {
            if (call.args.length !== 2) throw new SyntaxError('min() takes exactly 2 arguments');
            const a = this.compileExpr(call.args[0], dest);
            const b = this.compileExpr(call.args[1], this.T1);
            if (a !== dest) this.emitMov(dest, a);
            this.emitMov(this.T2, dest);
            this.emitSub(this.T2, b);
            const skip = this.newLabel('min_skip');
            this.emitCmp(this.T2, skip);
            this.emitMov(dest, b);
            this.emitLabel(skip);
            return dest;
        }

        // Built-in: max(a, b)
        if (call.name === 'max') {
            if (call.args.length !== 2) throw new SyntaxError('max() takes exactly 2 arguments');
            const a = this.compileExpr(call.args[0], dest);
            const b = this.compileExpr(call.args[1], this.T1);
            if (a !== dest) this.emitMov(dest, a);
            this.emitMov(this.T2, b);
            this.emitSub(this.T2, dest);
            const skip = this.newLabel('max_skip');
            this.emitCmp(this.T2, skip);
            this.emitMov(dest, b);
            this.emitLabel(skip);
            return dest;
        }

        // Built-in: mul(a, b)
        if (call.name === 'mul') {
            if (call.args.length !== 2) throw new SyntaxError('mul() takes exactly 2 arguments');
            const a = this.compileExpr(call.args[0], dest);
            const b = this.compileExpr(call.args[1], this.T1);
            if (a !== dest) this.emitMov(dest, a);
            const bSlot = b === this.T1 ? this.T1 : b;

            const neg = this.mem.allocVar('__mul_neg');
            this.emitMov(this.T2, dest);
            this.emitXor(this.T2, bSlot);
            // abs(a) -> dest
            this.emitMov(neg, this.CONST_ZERO);
            this.emitSub(neg, dest);
            this.emitCmov(dest, neg);
            // abs(b) -> T3
            this.emitMov(this.T3, bSlot);
            this.emitMov(neg, this.CONST_ZERO);
            this.emitSub(neg, this.T3);
            this.emitCmov(this.T3, neg);
            // N MULACC steps
            for (let i = 0; i < this.cfg.N; i++) this.emitMulacc(dest, this.T3);
            // Conditionally negate
            this.emitMov(neg, this.CONST_ZERO);
            this.emitSub(neg, dest);
            const negLabel = this.newLabel('mul_neg');
            const endLabel = this.newLabel('mul_end');
            this.emitCmp(this.T2, negLabel);
            this.emitJmp(endLabel);
            this.emitLabel(negLabel);
            this.emitMov(dest, neg);
            this.emitLabel(endLabel);
            return dest;
        }

        // Built-in: swap(a, b)
        if (call.name === 'swap') {
            if (call.args.length !== 2) throw new SyntaxError('swap() takes exactly 2 arguments');
            const a = this.compileExpr(call.args[0], dest);
            const b = this.compileExpr(call.args[1], this.T1);
            this.emitSwap(a, b);
            return dest;
        }

        // Inline user function
        if (!(call.name in this.functions)) throw new Error(`Undefined function '${call.name}'`);
        const func = this.functions[call.name];
        if (call.args.length !== func.params.length)
            throw new SyntaxError(`Function '${call.name}' expects ${func.params.length} args, got ${call.args.length}`);

        for (let i = 0; i < func.params.length; i++) {
            const src = this.compileExpr(call.args[i], this.T0);
            const paramAddr = this.mem.allocVar(`__${call.name}_${func.params[i]}`);
            if (src !== paramAddr) this.emitMov(paramAddr, src);
        }

        const savedSlots = {};
        for (const param of func.params) {
            const internalName = `__${call.name}_${param}`;
            savedSlots[param] = this.mem.slots[param] || null;
            this.mem.slots[param] = this.mem.slots[internalName];
        }

        this.compileBlock(func.body);

        for (const param of func.params) {
            if (savedSlots[param]) this.mem.slots[param] = savedSlots[param];
            else delete this.mem.slots[param];
        }

        return this.RETVAL;
    }
}

// ============================================================
// Public API
// ============================================================

function compileC(source, s = 32, m = 64, n = 1024, N = 8) {
    const cfg = makeConfig(s, m, n, N);
    const tokens = lex(source);
    const parser = new Parser(tokens);
    const program = parser.parseProgram();

    const codegen = new CodeGen(cfg);
    codegen.compileProgram(program);

    const memory = codegen.mem.initMemory();
    const commands = codegen.resolve();

    const variables = {};
    for (const [name, slot] of Object.entries(codegen.mem.slots)) {
        if (!name.startsWith('__')) variables[name] = slot.addr;
    }

    const meta = {
        variables,
        constants: { ...codegen.mem.constants },
        num_instructions: commands.length,
        retval_addr: codegen.RETVAL,
        output_addr: codegen.OUTPUT,
        labels: { ...codegen.labels },
        source_map: codegen.sourceMap,
    };

    return { config: cfg, memory, commands, meta };
}
