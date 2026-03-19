/**
 * Sparse Argmax Transformer Engine
 *
 * Optimized for the V4 architecture's extreme sparsity:
 *   - 8,019 nonzero weights (99.9% sparse)
 *   - Argmax attention: no softmax, no n×n matrix, no exp()
 *   - 3-zone biases: (col0, scratchpad, memory)
 *   - 4 of 8 layers are pure FFN (attention skipped)
 *   - Pre-allocated scratch buffers (zero GC pressure)
 *
 * Target: <1ms per step at 146×512 (vs ~10ms for ONNX)
 */

class SparseArgmaxEngine {
    constructor(weightsJSON, n) {
        this.d = weightsJSON.d;
        this.s = weightsJSON.s;
        this.n = n;

        // Parse layers
        this.layers = weightsJSON.layers.map(l => this._parseLayer(l));

        // Pre-allocate all scratch buffers
        let maxQ = 0, maxW1 = 0;
        for (const layer of this.layers) {
            if (layer.qRows > maxQ) maxQ = layer.qRows;
            if (layer.w1Rows > maxW1) maxW1 = layer.w1Rows;
        }
        this._QX = new Float32Array(maxQ * n);
        this._KX = new Float32Array(maxQ * n);
        this._VX = new Float32Array(this.d * n);
        this._VXsrc = new Float32Array(this.d * n);
        this._ff = new Float32Array(maxW1 * n);
    }

    _parseLayer(l) {
        const [numHeads, qRows, w1Rows, headsData, W1_coo, b1_3z, W2_coo, b2_3z] = l;
        const heads = [];
        for (const h of headsData) {
            const [Q_coo, K_coo, V_coo] = h;
            if (Q_coo[0].length === 0 && K_coo[0].length === 0 && V_coo[0].length === 0) {
                heads.push(null);
                continue;
            }
            heads.push({
                Q: this._toCOO(Q_coo),
                K: this._toCOO(K_coo),
                V: this._toCOO(V_coo),
            });
        }
        return {
            numHeads, qRows, w1Rows, heads,
            W1: this._toCOO(W1_coo),
            W2: this._toCOO(W2_coo),
            b1: this._toBias3z(b1_3z, w1Rows),
            b2: this._toBias3z(b2_3z, this.d),
        };
    }

    _toCOO(coo) {
        const [rows, cols, vals] = coo;
        return {
            nnz: rows.length,
            rows: new Int32Array(rows),
            cols: new Int32Array(cols),
            vals: new Float32Array(vals),
        };
    }

    _toBias3z(b3z, rows) {
        const [col0, scr, mem] = b3z;
        return {
            col0: new Float32Array(col0),
            scr: new Float32Array(scr),
            mem: new Float32Array(mem),
        };
    }

    /**
     * One full transformer step (8 layers). X is modified in-place.
     * Trim to nt active columns for attention (skip empty instruction slots).
     */
    step(X, nt) {
        const d = this.d, n = this.n, s = this.s;
        const QX = this._QX, KX = this._KX, VX = this._VX;
        const VXsrc = this._VXsrc, ff = this._ff;
        // nt = number of active columns (s + m + num_instructions)
        // Attention only needs to scan nt columns, not all n
        if (!nt || nt > n) nt = n;

        for (let li = 0; li < this.layers.length; li++) {
            const layer = this.layers[li];
            const qRows = layer.qRows;
            const w1Rows = layer.w1Rows;
            const dn = d * n;

            // ---- Attention (argmax, no n×n matrix) ----
            for (let i = 0; i < dn; i++) VX[i] = 0;

            for (let hi = 0; hi < layer.heads.length; hi++) {
                const head = layer.heads[hi];
                if (!head) continue;

                // QX = Q @ X  (only nt columns matter)
                const qLen = qRows * n;
                for (let i = 0; i < qLen; i++) QX[i] = 0;
                this._spmv_trim(head.Q, X, n, QX, n, nt);

                // KX = K @ X
                for (let i = 0; i < qLen; i++) KX[i] = 0;
                this._spmv_trim(head.K, X, n, KX, n, nt);

                // VXsrc = V @ X (only source columns that have nonzero KX)
                for (let i = 0; i < dn; i++) VXsrc[i] = 0;
                this._spmv_trim(head.V, X, n, VXsrc, n, nt);

                // Argmax: scan only nt columns
                this._argmaxScatter(QX, KX, qRows, n, nt, VXsrc, d, VX);
            }

            // Residual: X += VX
            for (let i = 0; i < dn; i++) X[i] += VX[i];

            // ---- FFN (trimmed to nt columns) ----
            const ffLen = w1Rows * n;
            for (let i = 0; i < ffLen; i++) ff[i] = 0;
            this._spmv_trim(layer.W1, X, n, ff, n, nt);
            this._addBias3z_trim(ff, w1Rows, n, s, layer.b1, nt);
            for (let r = 0; r < w1Rows; r++) {
                const off = r * n;
                for (let j = 0; j < nt; j++) if (ff[off + j] < 0) ff[off + j] = 0;
            }
            this._spmv_trim(layer.W2, ff, n, X, n, nt);
            this._addBias3z_trim(X, d, n, s, layer.b2, nt);
        }
    }

    /**
     * One step with per-layer tracing. Returns array of 8 layer traces.
     * Each trace contains FULL 2D snapshots (d x nt):
     *   input:     Float32Array(d*nt) state entering this layer
     *   attnDelta: Float32Array(d*nt) what attention added
     *   delta:     Float32Array(d*nt) total change (output - input)
     *   output:    Float32Array(d*nt) state leaving this layer
     * All stored row-major with stride = n (original state layout).
     */
    stepTraced(X, nt) {
        const d = this.d, n = this.n, s = this.s;
        const QX = this._QX, KX = this._KX, VX = this._VX;
        const VXsrc = this._VXsrc, ff = this._ff;
        if (!nt || nt > n) nt = n;

        const traces = [];

        for (let li = 0; li < this.layers.length; li++) {
            const layer = this.layers[li];
            const qRows = layer.qRows;
            const w1Rows = layer.w1Rows;
            const dn = d * n;

            // Snapshot input (full 2D, trimmed to nt columns)
            const input = new Float32Array(d * nt);
            for (let r = 0; r < d; r++)
                for (let c = 0; c < nt; c++)
                    input[r * nt + c] = X[r * n + c];

            // Attention
            for (let i = 0; i < dn; i++) VX[i] = 0;
            for (let hi = 0; hi < layer.heads.length; hi++) {
                const head = layer.heads[hi];
                if (!head) continue;
                const qLen = qRows * n;
                for (let i = 0; i < qLen; i++) QX[i] = 0;
                this._spmv_trim(head.Q, X, n, QX, n, nt);
                for (let i = 0; i < qLen; i++) KX[i] = 0;
                this._spmv_trim(head.K, X, n, KX, n, nt);
                for (let i = 0; i < dn; i++) VXsrc[i] = 0;
                this._spmv_trim(head.V, X, n, VXsrc, n, nt);
                this._argmaxScatter(QX, KX, qRows, n, nt, VXsrc, d, VX);
            }

            // Snapshot attention delta (full 2D)
            const attnDelta = new Float32Array(d * nt);
            for (let r = 0; r < d; r++)
                for (let c = 0; c < nt; c++)
                    attnDelta[r * nt + c] = VX[r * n + c];

            for (let i = 0; i < dn; i++) X[i] += VX[i];

            // FFN
            const ffLen = w1Rows * n;
            for (let i = 0; i < ffLen; i++) ff[i] = 0;
            this._spmv_trim(layer.W1, X, n, ff, n, nt);
            this._addBias3z_trim(ff, w1Rows, n, s, layer.b1, nt);
            for (let r = 0; r < w1Rows; r++) {
                const off = r * n;
                for (let j = 0; j < nt; j++) if (ff[off + j] < 0) ff[off + j] = 0;
            }
            this._spmv_trim(layer.W2, ff, n, X, n, nt);
            this._addBias3z_trim(X, d, n, s, layer.b2, nt);

            // Snapshot output + delta (full 2D)
            const output = new Float32Array(d * nt);
            const delta = new Float32Array(d * nt);
            for (let r = 0; r < d; r++)
                for (let c = 0; c < nt; c++) {
                    output[r * nt + c] = X[r * n + c];
                    delta[r * nt + c] = output[r * nt + c] - input[r * nt + c];
                }

            traces.push({ input, attnDelta, delta, output, nt });
        }

        return traces;
    }

    /**
     * Sparse matrix × dense vector (all n columns).
     */
    _spmv(coo, X, xStride, out, outStride) {
        const { nnz, rows, cols, vals } = coo;
        for (let k = 0; k < nnz; k++) {
            const r = rows[k], c = cols[k], v = vals[k];
            const outOff = r * outStride;
            const xOff = c * xStride;
            for (let j = 0; j < xStride; j++) {
                out[outOff + j] += v * X[xOff + j];
            }
        }
    }

    /**
     * Sparse matrix × dense vector, only first nt columns.
     */
    _spmv_trim(coo, X, xStride, out, outStride, nt) {
        const { nnz, rows, cols, vals } = coo;
        for (let k = 0; k < nnz; k++) {
            const r = rows[k], c = cols[k], v = vals[k];
            const outOff = r * outStride;
            const xOff = c * xStride;
            for (let j = 0; j < nt; j++) {
                out[outOff + j] += v * X[xOff + j];
            }
        }
    }

    /**
     * Argmax attention scatter, scanning only nt columns.
     */
    _argmaxScatter(QX, KX, qRows, n, nt, VXsrc, d, VXout) {
        for (let i = 0; i < nt; i++) {
            // Check if KX column i is zero
            let isZero = true;
            for (let q = 0; q < qRows; q++) {
                if (KX[q * n + i] !== 0) { isZero = false; break; }
            }
            if (isZero) continue;

            // Find top-2 (only scan nt columns)
            let best1v = -1e30, best2v = -1e30;
            let best1j = 0, best2j = 0;
            for (let j = 0; j < nt; j++) {
                let score = 0;
                for (let q = 0; q < qRows; q++) {
                    score += KX[q * n + i] * QX[q * n + j];
                }
                if (score > best1v) {
                    best2v = best1v; best2j = best1j;
                    best1v = score; best1j = j;
                } else if (score > best2v) {
                    best2v = score; best2j = j;
                }
            }

            // Scatter
            const tied = (best1v - best2v) < 1.0;
            const w = tied ? 0.5 : 1.0;
            for (let r = 0; r < d; r++) {
                const val = VXsrc[r * n + i];
                VXout[r * n + best1j] += w * val;
                if (tied) VXout[r * n + best2j] += 0.5 * val;
            }
        }
    }

    /**
     * 3-zone bias: col0 for column 0, scr for columns 1..s-1, mem for columns s..n-1
     */
    _addBias3z(out, rows, n, s, bias) {
        const { col0, scr, mem } = bias;
        for (let r = 0; r < rows; r++) {
            const off = r * n;
            out[off] += col0[r];
            const sv = scr[r];
            if (sv !== 0) {
                for (let j = 1; j < s; j++) out[off + j] += sv;
            }
            const mv = mem[r];
            if (mv !== 0) {
                for (let j = s; j < n; j++) out[off + j] += mv;
            }
        }
    }

    /**
     * 3-zone bias, only first nt columns.
     */
    _addBias3z_trim(out, rows, n, s, bias, nt) {
        const { col0, scr, mem } = bias;
        for (let r = 0; r < rows; r++) {
            const off = r * n;
            out[off] += col0[r];
            const sv = scr[r];
            if (sv !== 0) {
                const end = Math.min(s, nt);
                for (let j = 1; j < end; j++) out[off + j] += sv;
            }
            const mv = mem[r];
            if (mv !== 0) {
                for (let j = s; j < nt; j++) out[off + j] += mv;
            }
        }
    }

    /**
     * Read PC from state (bipolar encoded).
     */
    getPC(X, idxPc, logn) {
        const n = this.n;
        let pc = 0;
        for (let i = 0; i < logn; i++) {
            if (X[(idxPc + i) * n] > 0) pc |= (1 << (logn - 1 - i));
        }
        return pc;
    }

    /**
     * Run until halt (PC=0) or maxSteps, with optional yield for UI.
     * nt = number of active columns (s + m + numInstructions).
     * Returns { steps, halted, ms }.
     */
    async run(X, config, maxSteps, yieldEvery = 100, nt = 0) {
        const idxPc = config.idx_pc;
        const logn = config.logn;
        if (!nt) nt = config.n;
        const t0 = performance.now();
        let steps = 0;

        while (steps < maxSteps) {
            if (this.getPC(X, idxPc, logn) === 0) {
                return { steps, halted: true, ms: performance.now() - t0 };
            }
            this.step(X, nt);
            steps++;
            if (yieldEvery > 0 && steps % yieldEvery === 0) {
                await new Promise(r => setTimeout(r, 0));
            }
        }
        return { steps, halted: false, ms: performance.now() - t0 };
    }
}

/**
 * Load sparse engine from JSON weights file.
 */
async function loadSparseArgmax(weightsPath, n) {
    const resp = await fetch(weightsPath);
    const json = await resp.json();
    return new SparseArgmaxEngine(json, n);
}
