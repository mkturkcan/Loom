/**
 * Neural SUBLEQ Computer - Shared Demo Utilities
 *
 * Provides bipolar encoding, state tensor access, ONNX inference,
 * heatmap rendering, and layer animation shared across all demos.
 */

// ============================================================
// Bipolar Encoding
// ============================================================

function fromBipolar(bits) {
    let value = 0;
    const n = bits.length;
    for (let i = 0; i < n; i++) {
        if (bits[i] > 0) value |= (1 << (n - 1 - i));
    }
    if (value >= (1 << (n - 1))) value -= (1 << n);
    return value;
}

function toBipolar(value, bits) {
    if (value < 0) value = value + (1 << bits);
    const result = new Array(bits);
    for (let i = 0; i < bits; i++) {
        result[i] = (value >> (bits - 1 - i)) & 1 ? 1.0 : -1.0;
    }
    return result;
}

// ============================================================
// State Tensor Access (flat Float32Array, row-major [d_model x n])
// ============================================================

function readMemory(flat, config, addr) {
    const col = config.s + addr;
    const n = config.n;
    const N = config.N;
    let value = 0;
    for (let i = 0; i < N; i++) {
        if (flat[(config.idx_memory + i) * n + col] > 0) {
            value |= (1 << (N - 1 - i));
        }
    }
    if (value >= (1 << (N - 1))) value -= (1 << N);
    return value;
}

function writeMemory(flat, config, addr, value) {
    const col = config.s + addr;
    const n = config.n;
    const N = config.N;
    if (value < 0) value += (1 << N);
    for (let i = 0; i < N; i++) {
        flat[(config.idx_memory + i) * n + col] = ((value >> (N - 1 - i)) & 1) ? 1.0 : -1.0;
    }
}

function getPC(flat, config) {
    const n = config.n;
    const logn = config.logn;
    let pc = 0;
    for (let i = 0; i < logn; i++) {
        if (flat[(config.idx_pc + i) * n] > 0) {
            pc |= (1 << (logn - 1 - i));
        }
    }
    return pc;
}

function readAllMemory(flat, config) {
    const result = [];
    for (let i = 0; i < config.m; i++) {
        result.push(readMemory(flat, config, i));
    }
    return result;
}

function copyState(flat) {
    return new Float32Array(flat);
}

// ============================================================
// ONNX Inference
// ============================================================

let _onnxSession = null;

async function loadONNXModel(modelPath) {
    ort.env.wasm.numThreads = 1;
    // Fetch model as ArrayBuffer to avoid external data file resolution issues
    const resp = await fetch(modelPath);
    const modelBuffer = await resp.arrayBuffer();
    let backend = 'unknown';
    try {
        _onnxSession = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['webgpu']
        });
        backend = 'WebGPU';
    } catch (e) {
        console.warn('WebGPU not available, falling back to WASM:', e);
        _onnxSession = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm']
        });
        backend = 'WASM';
    }
    return backend;
}

async function runStep(flat, config) {
    const inputTensor = new ort.Tensor('float32', flat, [config.d_model, config.n]);
    const results = await _onnxSession.run({ state: inputTensor });
    return new Float32Array(results.new_state.data);
}

/**
 * Run until PC == 0 (halt) or PC == stopPC, with step/time limits.
 * Calls onStep(state, pc, stepNum) after each step if provided.
 * Returns { state, steps, halted, ms }.
 */
async function runUntilStop(state, config, opts = {}) {
    const maxSteps = opts.maxSteps || 50000;
    const stopPC = opts.stopPC || 0;
    const onStep = opts.onStep || null;
    const yieldEvery = opts.yieldEvery || 50;

    let steps = 0;
    const t0 = performance.now();

    while (steps < maxSteps) {
        const pc = getPC(state, config);
        if (pc === 0) return { state, steps, halted: true, ms: performance.now() - t0 };
        if (stopPC && pc === stopPC && steps > 0) return { state, steps, halted: false, ms: performance.now() - t0 };

        state = await runStep(state, config);
        steps++;

        if (onStep) onStep(state, getPC(state, config), steps);

        // Yield to UI periodically
        if (steps % yieldEvery === 0) {
            await new Promise(r => setTimeout(r, 0));
        }
    }

    return { state, steps, halted: false, ms: performance.now() - t0 };
}

/**
 * Fast execution path for real-time demos.
 * Reuses a single buffer + tensor to eliminate per-step allocations
 * (~320MB GC pressure removed). No callbacks, no yields.
 */
async function runUntilStopFast(inputState, config, maxSteps = 5000) {
    const n = config.n;
    const d = config.d_model;
    const logn = config.logn;
    const t0 = performance.now();
    let steps = 0;

    // Pre-compute PC row offsets for inline halt check
    const pcOffsets = new Array(logn);
    for (let i = 0; i < logn; i++) pcOffsets[i] = (config.idx_pc + i) * n;

    // Single reusable buffer — tensor wraps it by reference
    const buf = new Float32Array(inputState);
    const tensor = new ort.Tensor('float32', buf, [d, n]);

    while (steps < maxSteps) {
        // Inline PC check (avoid function call + object creation)
        let pc = 0;
        for (let i = 0; i < logn; i++) {
            if (buf[pcOffsets[i]] > 0) pc |= (1 << (logn - 1 - i));
        }
        if (pc === 0) return { state: buf, steps, halted: true, ms: performance.now() - t0 };

        // tensor.data IS buf, so session reads updated data each call
        const results = await _onnxSession.run({ state: tensor });
        buf.set(results.new_state.data);
        steps++;
    }

    return { state: buf, steps, halted: false, ms: performance.now() - t0 };
}

// ============================================================
// Layer Animation
// ============================================================

let _layerAnimFrame = null;

function animateLayerForward(stepDurationMs) {
    const layers = [1,2,3,4,5,6,7,8];
    const interval = stepDurationMs / layers.length;
    let i = 0;

    function tick() {
        document.querySelectorAll('.arch-layer[data-layer]').forEach(el => el.classList.remove('active'));
        if (i < layers.length) {
            const el = document.querySelector(`.arch-layer[data-layer="${layers[i]}"]`);
            if (el) el.classList.add('active');
            i++;
            _layerAnimFrame = setTimeout(tick, Math.max(interval, 8));
        }
    }
    tick();
}

function clearLayerAnimation() {
    if (_layerAnimFrame) clearTimeout(_layerAnimFrame);
    document.querySelectorAll('.arch-layer[data-layer]').forEach(el => el.classList.remove('active'));
}

// Requires _currentState and _currentConfig to be set by the demo
let _currentState = null;
let _currentConfig = null;

// ============================================================
// State Tensor Heatmap
// ============================================================

const regionColors = {
    instructions: [59, 130, 246],   // blue
    memory:       [16, 185, 129],   // green
    scratchpad:   [139, 92, 246],   // purple
    pc:           [245, 158, 11],   // amber
    posEnc:       [107, 114, 128],  // gray
    buffer:       [239, 68, 68],    // red
    indicator:    [99, 102, 241],   // indigo
};

function getRowColor(row) {
    if (row < 30) return regionColors.instructions;
    if (row < 38) return regionColors.memory;
    if (row < 84) return regionColors.scratchpad;
    if (row < 94) return regionColors.pc;
    if (row < 104) return regionColors.posEnc;
    if (row < 146) return regionColors.buffer;
    return regionColors.indicator;
}

function _renderHeatmapRegion(canvas, state, config, colStart, colEnd) {
    if (!state || !config) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;
    const n = config.n;
    const d = config.d_model;
    const srcCols = colEnd - colStart;
    const colStep = srcCols / w;

    for (let py = 0; py < h && py < d; py++) {
        const color = getRowColor(py);
        for (let px = 0; px < w; px++) {
            const srcCol = colStart + Math.floor(px * colStep);
            const v = state[py * n + srcCol];
            const idx = (py * w + px) * 4;
            const intensity = (v + 1) / 2;
            const bright = intensity * intensity;
            const alpha = bright * 0.85;
            data[idx]     = Math.floor(color[0] * alpha + 240 * (1 - alpha));
            data[idx + 1] = Math.floor(color[1] * alpha + 240 * (1 - alpha));
            data[idx + 2] = Math.floor(color[2] * alpha + 240 * (1 - alpha));
            data[idx + 3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function renderHeatmap(canvasOrEl, state, config) {
    // Support both old single-canvas API and new split layout
    const scratchCanvas = document.getElementById('stateCanvasScratch');
    const dataCanvas = document.getElementById('stateCanvasData');
    if (scratchCanvas && dataCanvas) {
        const s = config ? config.s : 32;
        _renderHeatmapRegion(scratchCanvas, state, config, 0, s);
        _renderHeatmapRegion(dataCanvas, state, config, s, config ? config.n : 1024);
    } else if (canvasOrEl && canvasOrEl.getContext) {
        // Fallback: single canvas
        _renderHeatmapRegion(canvasOrEl, state, config, 0, config ? config.n : 1024);
    }
}

// ============================================================
// Status Helpers
// ============================================================

function setStatus(type, text) {
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    if (dot) dot.className = 'status-dot ' + type;
    if (txt) txt.textContent = text;
}

function updateStat(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// ============================================================
// State Loading
// ============================================================

async function loadInitialState(url) {
    const resp = await fetch(url);
    const data = await resp.json();
    const arr2d = data.state;
    const rows = arr2d.length;
    const cols = arr2d[0].length;
    const flat = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            flat[i * cols + j] = arr2d[i][j];
        }
    }
    return flat;
}

/**
 * Build state tensor from compile result (mirrors init_state_v4 in Python).
 * Used by the REPL to avoid torch dependency.
 */
function buildStateTensor(compileResult) {
    const cfg = compileResult.config;
    const memory = compileResult.memory;
    const commands = compileResult.commands;
    const d = cfg.d_model;
    const n = cfg.n;
    const N = cfg.N;
    const logn = cfg.logn;
    const s = cfg.s;
    const m = cfg.m;
    const idx_memory = cfg.idx_memory;
    const idx_pc = cfg.idx_pc;
    const idx_pos_enc = cfg.idx_pos_enc;
    const idx_indicator = d - 1;

    const flat = new Float32Array(d * n);

    // Indicator row: +1 for scratchpad columns (0..s-1)
    for (let c = 0; c < s; c++) {
        flat[idx_indicator * n + c] = 1.0;
    }

    // Position encoding for columns 1..n-1
    for (let col = 1; col < n; col++) {
        const bits = toBipolar(col, logn);
        for (let i = 0; i < logn; i++) {
            flat[(idx_pos_enc + i) * n + col] = bits[i];
        }
    }

    // Memory
    for (let i = 0; i < memory.length && i < m; i++) {
        const col = s + i;
        let val = memory[i];
        if (val < 0) val += (1 << N);
        for (let b = 0; b < N; b++) {
            const bit = (val >> (N - 1 - b)) & 1;
            flat[(idx_memory + b) * n + col] = bit ? 1.0 : -1.0;
        }
    }

    // Address tags: each memory column stores its own index (for LOAD/FIND)
    const idx_tag = cfg.idx_tag;
    if (idx_tag !== undefined) {
        for (let i = 0; i < m; i++) {
            const col = s + i;
            const bits = toBipolar(i, N);
            for (let b = 0; b < N; b++) {
                flat[(idx_tag + b) * n + col] = bits[b];
            }
        }
    }

    // Commands (instructions)
    for (let i = 0; i < commands.length; i++) {
        const col = s + m + i;
        if (col >= n) break;
        const [a, b, c] = commands[i];
        const aBits = toBipolar(a, logn);
        const bBits = toBipolar(b, logn);
        const cBits = toBipolar(c, logn);
        for (let j = 0; j < logn; j++) {
            flat[j * n + col] = aBits[j];
            flat[(logn + j) * n + col] = bBits[j];
            flat[(2 * logn + j) * n + col] = cBits[j];
        }
    }

    // PC = s + m (first instruction address)
    const pc = s + m;
    const pcBits = toBipolar(pc, logn);
    for (let i = 0; i < logn; i++) {
        flat[(idx_pc + i) * n + 0] = pcBits[i];
    }

    return flat;
}

async function loadConfig(url) {
    const resp = await fetch(url);
    const config = await resp.json();
    // Derive row indices if not present
    if (!config.idx_scratchpad) config.idx_scratchpad = config.idx_memory + config.N;
    if (!config.idx_pos_enc) config.idx_pos_enc = config.idx_pc + config.logn;
    if (!config.idx_buffer) config.idx_buffer = config.idx_pos_enc + config.logn;
    if (!config.idx_tag) config.idx_tag = config.idx_buffer + 3 * config.N + config.N + config.logn;
    return config;
}

// ============================================================
// Architecture Diagram HTML Generator
// ============================================================

function buildArchDiagramHTML() {
    const layers = [
        { id: 'input', label: 'IN', name: 'State Tensor <span style="color:#374151">155&times;1024</span>', color: '#10b981' },
        { id: '1', label: 'L1', name: 'Fetch Instruction', color: '#3b82f6' },
        { id: '2', label: 'L2', name: 'Read Operands + Decode', color: '#3b82f6' },
        { id: '3', label: 'L3', name: 'Indirect Read + Correction', color: '#8b5cf6' },
        { id: '4', label: 'L4', name: 'Subtract (Direct)', color: '#f59e0b' },
        { id: '5', label: 'L5', name: 'Write Memory', color: '#10b981' },
        { id: '6', label: 'L6', name: 'Branch Flag + PC+1', color: '#ef4444' },
        { id: '7', label: 'L7', name: 'Branch Select', color: '#ef4444' },
        { id: '8', label: 'L8', name: 'Error Correction', color: '#6366f1' },
        { id: 'output', label: 'OUT', name: 'Updated State', color: '#10b981' },
    ];

    return layers.map((l, i) => {
        const layer = `<div class="arch-layer" data-layer="${l.id}" style="border-left:3px solid ${l.color}">
            <div class="layer-glow"></div>
            <span class="layer-id">${l.label}</span>
            <span class="layer-name">${l.name}</span>
        </div>`;
        const connector = i < layers.length - 1 ? '<div class="arch-connector"></div>' : '';
        return layer + connector;
    }).join('\n');
}

// ============================================================
// Network Panel HTML Generator
// ============================================================

function buildNetworkPanelHTML(extraPanels = '') {
    return `
    <div class="panel">
        <div class="panel-title">Transformer Architecture: 8 Layers</div>
        <div class="arch-layers" id="archLayers">
            ${buildArchDiagramHTML()}
        </div>
    </div>

    <div class="panel">
        <div class="panel-title">State Tensor: 155 &times; 1024</div>
        <div style="display:flex;gap:0;align-items:stretch">
            <div style="flex:0 0 80px">
                <div style="font-size:0.6rem;color:#9ca3af;text-align:center;margin-bottom:4px">Cols 0–31</div>
                <canvas id="stateCanvasScratch" width="32" height="155" style="width:100%;height:120px;border-radius:4px 0 0 4px;image-rendering:pixelated;background:#f3f4f6;display:block;border:1px solid #e5e7eb;border-right:none"></canvas>
            </div>
            <div style="flex:0 0 2px;background:repeating-linear-gradient(to bottom,#d1d5db 0,#d1d5db 4px,transparent 4px,transparent 8px);margin-top:18px;height:120px"></div>
            <div style="flex:1;min-width:0">
                <div style="font-size:0.6rem;color:#9ca3af;text-align:center;margin-bottom:4px">Cols 32–1023</div>
                <canvas id="stateCanvasData" width="272" height="155" style="width:100%;height:120px;border-radius:0 4px 4px 0;image-rendering:pixelated;background:#f3f4f6;display:block;border:1px solid #e5e7eb;border-left:none"></canvas>
            </div>
        </div>
        <div style="display:flex;gap:0;margin-top:4px">
            <div style="flex:0 0 80px;font-size:0.55rem;color:#6b7280;text-align:center;line-height:1.3">Scratchpad<br>columns</div>
            <div style="flex:0 0 2px"></div>
            <div style="flex:1;font-size:0.55rem;color:#6b7280;text-align:center;line-height:1.3">Memory (32–95) &middot; Instructions (96+)</div>
        </div>
        <div class="heatmap-labels" style="margin-top:8px">
            <div class="heatmap-label"><div class="heatmap-swatch" style="background:#3b82f6"></div><span class="heatmap-label-text">Instructions (rows 0-29)</span></div>
            <div class="heatmap-label"><div class="heatmap-swatch" style="background:#10b981"></div><span class="heatmap-label-text">Memory (rows 30-37)</span></div>
            <div class="heatmap-label"><div class="heatmap-swatch" style="background:#8b5cf6"></div><span class="heatmap-label-text">Scratchpad (rows 38-83)</span></div>
            <div class="heatmap-label"><div class="heatmap-swatch" style="background:#f59e0b"></div><span class="heatmap-label-text">PC (rows 84-93)</span></div>
            <div class="heatmap-label"><div class="heatmap-swatch" style="background:#6b7280"></div><span class="heatmap-label-text">Position Encoding (rows 94-103)</span></div>
            <div class="heatmap-label"><div class="heatmap-swatch" style="background:#ef4444"></div><span class="heatmap-label-text">Buffer (rows 104-145)</span></div>
            <div class="heatmap-label"><div class="heatmap-swatch" style="background:#64748b"></div><span class="heatmap-label-text">Tags + Indicator (rows 146-154)</span></div>
        </div>
        <div style="font-size:0.6rem;color:#9ca3af;line-height:1.4;margin-top:8px;padding-top:8px;border-top:1px solid #e5e7eb">
            Each row is a feature dimension, each column a position in the state.
            Column 0 is the working register; all computation (fetch, decode,
            execute) happens here, changing every step. Columns 1–31 are reserved
            scratchpad slots but unused by the current ISA. Columns 32–95 hold memory;
            96+ hold program instructions. Both update only on memory writes.
        </div>
    </div>

    ${extraPanels}

    <div class="panel">
        <div class="panel-title">Inference Stats</div>
        <div class="step-info">
            <div class="step-stat">
                <span class="step-stat-label">Steps</span>
                <span class="step-stat-value" id="statSteps">0</span>
            </div>
            <div class="step-stat">
                <span class="step-stat-label">ms/Step</span>
                <span class="step-stat-value" id="statMs">0</span>
            </div>
            <div class="step-stat">
                <span class="step-stat-label">Total</span>
                <span class="step-stat-value" id="statTotal">0</span>
            </div>
            <div class="step-stat">
                <span class="step-stat-label">Backend</span>
                <span class="step-stat-value" id="statBackend">--</span>
            </div>
        </div>
    </div>`;
}

// ============================================================
// Simple C Syntax Highlighter
// ============================================================

function highlightC(code) {
    // Escape HTML
    let s = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    // Comments
    s = s.replace(/(\/\/.*)/g, '<span class="cmt">$1</span>');
    // Strings
    s = s.replace(/("(?:[^"\\]|\\.)*")/g, '<span class="str">$1</span>');
    // Types
    s = s.replace(/\b(int|void|char)\b/g, '<span class="type">$1</span>');
    // Keywords
    s = s.replace(/\b(if|else|while|for|return|break|continue)\b/g, '<span class="kw">$1</span>');
    // Numbers
    s = s.replace(/\b(\d+)\b/g, '<span class="num">$1</span>');
    return s;
}

function buildCodeView(source, containerId) {
    const lines = source.split('\n');
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    lines.forEach((line, i) => {
        const div = document.createElement('div');
        div.className = 'code-line';
        div.dataset.line = i + 1;
        div.innerHTML = `<span class="code-lineno">${i + 1}</span><span class="code-text">${highlightC(line)}</span>`;
        container.appendChild(div);
    });
}

function highlightCodeLine(containerId, lineNum) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.querySelectorAll('.code-line').forEach(el => {
        el.classList.remove('active');
    });
    if (lineNum > 0) {
        const line = container.querySelector(`[data-line="${lineNum}"]`);
        if (line) {
            line.classList.add('active');
            line.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }
    }
}

// ============================================================
// V5 (Fast Engine) — Separated ROM Architecture
// ============================================================

let _v5Rom = null;       // Float32Array, row-major [nrows_cmds x num_commands]
let _v5RomCols = 0;      // number of instruction columns in ROM
let _v5RomRows = 0;      // nrows_cmds = 3 * logn

/**
 * Load ROM from JSON. Returns { rom, rows, cols }.
 */
async function loadV5ROM(url) {
    const resp = await fetch(url);
    const data = await resp.json();
    const arr2d = data.rom;
    _v5RomRows = arr2d.length;
    _v5RomCols = arr2d[0].length;
    _v5Rom = new Float32Array(_v5RomRows * _v5RomCols);
    for (let i = 0; i < _v5RomRows; i++) {
        for (let j = 0; j < _v5RomCols; j++) {
            _v5Rom[i * _v5RomCols + j] = arr2d[i][j];
        }
    }
    return { rom: _v5Rom, rows: _v5RomRows, cols: _v5RomCols };
}

/**
 * Read memory from V5 state (same as V4 but uses w columns).
 */
function readMemoryV5(flat, config, addr) {
    const col = config.s + addr;
    const w = config.w;
    const N = config.N;
    let value = 0;
    for (let i = 0; i < N; i++) {
        if (flat[(config.idx_memory + i) * w + col] > 0) {
            value |= (1 << (N - 1 - i));
        }
    }
    if (value >= (1 << (N - 1))) value -= (1 << N);
    return value;
}

function writeMemoryV5(flat, config, addr, value) {
    const col = config.s + addr;
    const w = config.w;
    const N = config.N;
    if (value < 0) value += (1 << N);
    for (let i = 0; i < N; i++) {
        flat[(config.idx_memory + i) * w + col] = ((value >> (N - 1 - i)) & 1) ? 1.0 : -1.0;
    }
}

function getPCV5(flat, config) {
    const w = config.w;
    const logn = config.logn;
    let pc = 0;
    for (let i = 0; i < logn; i++) {
        if (flat[(config.idx_pc + i) * w] > 0) {
            pc |= (1 << (logn - 1 - i));
        }
    }
    return pc;
}

/**
 * Single V5 step: ROM fetch + buffer clear + instruction write + ONNX.
 * State is flat Float32Array of shape (d_model, w).
 */
async function runStepV5(flat, config) {
    const w = config.w;
    const d = config.d_model;
    const logn = config.logn;
    const nrows_inst = config.nrows_cmds;  // 3 * logn
    const cmdStart = config.s + config.m;

    // Read PC
    let pc = getPCV5(flat, config);
    const romIdx = pc - cmdStart;

    // Clear buffer rows in column 0
    const bufStart = config.idx_buffer;
    const bufEnd = bufStart + 3 * config.N + config.N + logn;
    for (let r = bufStart; r < bufEnd; r++) {
        flat[r * w] = 0.0;
    }

    // Write instruction from ROM to scratch_cmd rows of column 0
    const scrStart = config.idx_scratch_cmd;
    if (romIdx >= 0 && romIdx < _v5RomCols) {
        for (let r = 0; r < nrows_inst; r++) {
            flat[(scrStart + r) * w] = _v5Rom[r * _v5RomCols + romIdx];
        }
    } else {
        // Out of range: write all -1 (HALT)
        for (let r = 0; r < nrows_inst; r++) {
            flat[(scrStart + r) * w] = -1.0;
        }
    }

    // Run ONNX model (layers 2-8)
    const inputTensor = new ort.Tensor('float32', flat, [d, w]);
    const results = await _onnxSession.run({ state: inputTensor });
    return new Float32Array(results.new_state.data);
}

/**
 * Fast V5 execution loop. Reuses single buffer, no allocations per step.
 */
async function runUntilStopFastV5(inputState, config, maxSteps = 5000) {
    const w = config.w;
    const d = config.d_model;
    const logn = config.logn;
    const N = config.N;
    const nrows_inst = config.nrows_cmds;
    const cmdStart = config.s + config.m;
    const scrStart = config.idx_scratch_cmd;
    const bufStart = config.idx_buffer;
    const bufEnd = bufStart + 3 * N + N + logn;
    const t0 = performance.now();
    let steps = 0;

    // Pre-compute PC row offsets
    const pcOffsets = new Array(logn);
    for (let i = 0; i < logn; i++) pcOffsets[i] = (config.idx_pc + i) * w;

    // Single reusable buffer
    const buf = new Float32Array(inputState);
    const tensor = new ort.Tensor('float32', buf, [d, w]);

    while (steps < maxSteps) {
        // Inline PC check
        let pc = 0;
        for (let i = 0; i < logn; i++) {
            if (buf[pcOffsets[i]] > 0) pc |= (1 << (logn - 1 - i));
        }
        if (pc === 0) return { state: buf, steps, halted: true, ms: performance.now() - t0 };

        // Clear buffer rows in column 0
        for (let r = bufStart; r < bufEnd; r++) buf[r * w] = 0.0;

        // Write instruction from ROM
        const romIdx = pc - cmdStart;
        if (romIdx >= 0 && romIdx < _v5RomCols) {
            for (let r = 0; r < nrows_inst; r++) {
                buf[(scrStart + r) * w] = _v5Rom[r * _v5RomCols + romIdx];
            }
        } else {
            for (let r = 0; r < nrows_inst; r++) {
                buf[(scrStart + r) * w] = -1.0;
            }
        }

        // Run ONNX (tensor.data IS buf)
        const results = await _onnxSession.run({ state: tensor });
        buf.set(results.new_state.data);
        steps++;
    }

    return { state: buf, steps, halted: false, ms: performance.now() - t0 };
}
