/**
 * Extended ISA V4 Interpreter
 *
 * Executes compiled instructions directly on a flat memory array,
 * bypassing the neural transformer entirely. Produces identical results
 * to the ONNX model when running the same compiled program.
 */

class ISAInterpreter {
    /**
     * @param {number} s - scratchpad columns (address offset)
     * @param {number} m - memory slots
     * @param {number} N - bits per value (default 8)
     * @param {number[]} initialMemory - initial memory values (length m)
     * @param {number[][]} commands - array of [op, b, c] instruction triples
     */
    constructor(s, m, N, initialMemory, commands) {
        this.s = s;
        this.m = m;
        this.N = N;
        this.mask = (1 << N) - 1;
        this.half = 1 << (N - 1);
        this.commands = commands;
        this.cmdStart = s + m;
        this.memory = new Array(m);
        for (let i = 0; i < m; i++) this.memory[i] = initialMemory[i] || 0;
        this.pc = this.cmdStart; // start at first instruction
    }

    clamp(val) {
        val = val & this.mask;
        if (val >= this.half) val -= (1 << this.N);
        return val;
    }

    getMem(addr) {
        const idx = addr - this.s;
        if (idx < 0 || idx >= this.m) return 0;
        return this.memory[idx];
    }

    setMem(addr, val) {
        const idx = addr - this.s;
        if (idx >= 0 && idx < this.m) this.memory[idx] = this.clamp(val);
    }

    /** Reset PC to start of program. Call between ticks. */
    resetPC() {
        this.pc = this.cmdStart;
    }

    /** Execute one instruction. Returns true if halted. */
    step() {
        if (this.pc === 0) return true;

        const instrIdx = this.pc - this.cmdStart;
        if (instrIdx < 0 || instrIdx >= this.commands.length) {
            this.pc = 0;
            return true;
        }

        const cmd = this.commands[instrIdx];
        const op = cmd[0], b = cmd[1], c = cmd[2];
        this.pc++; // default: advance

        switch (op) {
            case 0: // HALT
                this.pc = 0;
                return true;

            case 1: // MOV
                this.setMem(b, this.getMem(c));
                break;

            case 2: // ADD
                this.setMem(b, this.getMem(b) + this.getMem(c));
                break;

            case 3: // JMP
                this.pc = c;
                break;

            case 4: // JZ
                if (this.getMem(b) === 0) this.pc = c;
                break;

            case 5: // JNZ
                if (this.getMem(b) !== 0) this.pc = c;
                break;

            case 6: // INC
                this.setMem(b, this.getMem(b) + 1);
                break;

            case 7: // DEC
                this.setMem(b, this.getMem(b) - 1);
                break;

            case 8: // SHL
                this.setMem(b, this.getMem(b) << 1);
                break;

            case 9: { // SHR (arithmetic)
                let val = this.getMem(b);
                // Arithmetic right shift: preserve sign
                val = val >> 1;
                this.setMem(b, val);
                break;
            }

            case 10: // CMP (BLTZ): if mem[b] < 0, jump to c
                if (this.getMem(b) < 0) this.pc = c;
                break;

            case 11: { // LOAD: mem[b] = mem[mem[c]] (indirect read)
                const ptr = this.getMem(c); // pointer value = memory index
                this.setMem(b, this.getMem(this.s + ptr));
                break;
            }

            case 12: // AND
                this.setMem(b, this.getMem(b) & this.getMem(c));
                break;

            case 13: // OR
                this.setMem(b, this.getMem(b) | this.getMem(c));
                break;

            case 14: // XOR
                this.setMem(b, this.getMem(b) ^ this.getMem(c));
                break;

            case 15: // SUB
                this.setMem(b, this.getMem(b) - this.getMem(c));
                break;

            case 16: { // FIND: search memory for value, write index to mem[b]
                const target = this.getMem(c);
                let found = 0;
                for (let i = 0; i < this.m; i++) {
                    if (this.memory[i] === target) { found = i; break; }
                }
                this.setMem(b, found);
                break;
            }

            case 17: { // SWAP
                const tmp = this.getMem(b);
                this.setMem(b, this.getMem(c));
                this.setMem(c, tmp);
                break;
            }

            case 18: // CMOV: if mem[b] < 0, mem[b] = mem[c]
                if (this.getMem(b) < 0) this.setMem(b, this.getMem(c));
                break;

            case 19: { // MULACC: shift-and-add multiply step
                const val = this.getMem(b);
                const msb = (val < 0) ? 1 : 0; // MSB = sign bit for N-bit signed
                const shifted = this.clamp(val << 1);
                const addend = msb ? this.getMem(c) : 0;
                this.setMem(b, shifted + addend);
                break;
            }
            case 20: { // STORE: mem[mem[c]] = mem[b] (indirect write)
                const ptr = this.getMem(c); // pointer value = memory index
                this.setMem(this.s + ptr, this.getMem(b)); // write to column s+ptr
                break;
            }
        }

        return false;
    }

    /**
     * Run until halt or maxSteps.
     * @returns {{ steps: number, halted: boolean }}
     */
    run(maxSteps = 50000) {
        let steps = 0;
        while (steps < maxSteps) {
            if (this.step()) return { steps, halted: true };
            steps++;
        }
        return { steps, halted: false };
    }

    /**
     * Read a variable by its memory index (not address).
     */
    readVar(memIdx) {
        return this.memory[memIdx];
    }

    /**
     * Write a variable by its memory index (not address).
     */
    writeVar(memIdx, value) {
        this.memory[memIdx] = this.clamp(value);
    }

    /**
     * Get all memory as array (for state comparison).
     */
    getMemorySnapshot() {
        return this.memory.slice();
    }

    /**
     * Sync memory from a state tensor (for switching from ONNX to ISA mode).
     * Reads all memory slots from the bipolar-encoded state tensor.
     */
    syncFromState(flat, config) {
        const n = config.n;
        const N = config.N;
        const s = config.s;
        const idxMem = config.idx_memory;
        for (let i = 0; i < this.m; i++) {
            const col = s + i;
            let value = 0;
            for (let bit = 0; bit < N; bit++) {
                if (flat[(idxMem + bit) * n + col] > 0) {
                    value |= (1 << (N - 1 - bit));
                }
            }
            if (value >= (1 << (N - 1))) value -= (1 << N);
            this.memory[i] = value;
        }
    }

    /**
     * Write ISA memory into a state tensor (for switching from ISA to ONNX mode).
     */
    syncToState(flat, config) {
        const n = config.n;
        const N = config.N;
        const s = config.s;
        const idxMem = config.idx_memory;
        for (let i = 0; i < this.m; i++) {
            const col = s + i;
            let val = this.memory[i];
            if (val < 0) val += (1 << N);
            for (let bit = 0; bit < N; bit++) {
                flat[(idxMem + bit) * n + col] = ((val >> (N - 1 - bit)) & 1) ? 1.0 : -1.0;
            }
        }
    }
}

/**
 * Create an ISA interpreter from a compile result.
 * @param {{ config, memory, commands }} compileResult - output of compileC()
 * @returns {ISAInterpreter}
 */
function createInterpreter(compileResult) {
    const cfg = compileResult.config;
    return new ISAInterpreter(cfg.s, cfg.m, cfg.N, compileResult.memory, compileResult.commands);
}
