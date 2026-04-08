# Fused Softmax — Design & Implementation

## What it does

Numerically stable softmax in a **single kernel launch**:

```
softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
```

## The problem with 5 kernels

The Stage 1 softmax ([examples/softmax_test.py](../examples/softmax_test.py)) uses 5 separate kernel launches. Each launch writes its output to global memory (GPU DRAM), which the next launch reads back:

```
Launch 1: reduce_max(x)          → writes max_val   to DRAM
Launch 2: sub_scalar(x, max_val) → writes shifted   to DRAM   (reads x, max_val)
Launch 3: exp(shifted)           → writes exps      to DRAM   (reads shifted)
Launch 4: reduce_sum(exps)       → writes sum_val   to DRAM   (reads exps)
Launch 5: div_scalar(exps, sum)  → writes out       to DRAM   (reads exps, sum_val)

Total: 5 launches, 9 DRAM round trips, 5 kernel startup overheads
```

At N=27 (microgpt vocab_size), each launch does ~27 floating-point operations. The launch overhead (~150µs) dwarfs the compute (~0.1µs). This is why the GPU ran 487x slower than CPU in Stage 2.

## The fused kernel

All intermediate values stay in **thread registers** — no global memory writes between steps:

```
load x → reduce_max → sub → exp → reduce_sum → div → store out

Total: 1 launch, 2 DRAM accesses (one load, one store), 1 kernel startup overhead
```

```python
@tt.jit
def fused_softmax_kernel(src, dst, N, n_masked_ptr):
    tid  = tt.arange(0, 64)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask)   # load all elements; masked → 0.0
    mx   = tt.reduce_max(x)                # warp-shuffle max → scalar
    e    = tt.exp(x - mx)                  # per-thread: exp(x[i] - max)
    s    = tt.reduce_sum(e)                # includes masked threads' exp(-mx)
    nm   = tt.load(n_masked_ptr)           # scalar: float(64 - N)
    s    = s - nm * tt.exp(0.0 - mx)      # subtract masked contribution
    tt.store(dst + tid, e / s, mask=mask)  # normalize and write

def fused_softmax(x, out, N):
    n_masked = np.array([float(64 - N)], dtype=np.float32)
    fused_softmax_kernel[(1,)](x, out, N, n_masked)   # always 1 block
```

No new builtins or C++ changes. Uses three existing patterns:
- `reduce_max`/`reduce_sum` returning scalars for further arithmetic (same as the existing composed kernels)
- Scalar broadcast load `tt.load(n_masked_ptr)` — same pattern as `n_arr` in `kern_rsqrt_mean`
- Float constant `0.0` in arithmetic expression

## How it executes on the GPU

With N=27, `grid=(1,)`, 64 threads per block:

```
Threads 0-26:  load x[0..26] into registers
Threads 27-63: load 0.0 (masked hardware default)

Warp shuffle (reduce_max):
               all 64 threads → mx = max(x[0..26], 0) = max(max_real, 0)

Threads 0-26:  e[i] = exp(x[i] - mx)     in registers
Threads 27-63: e[i] = exp(0.0 - mx) = exp(-mx)   (non-zero, must be corrected)

Warp shuffle (reduce_sum):
               s_raw = sum(e[0..63]) = sum_real + 37 * exp(-mx)

Correction:
               nm = load(n_masked_ptr) = 37.0        (= 64 - 27)
               s  = s_raw - 37.0 * exp(-mx)
                  = sum_real                          (correct normalization)

Threads 0-26:  store e[i] / s_corrected → dst[i]    (correct softmax)
Threads 27-63: masked, no store
```

The `n_masked_ptr` is a 1-element float array `[float(64 - N)]` computed on the host and passed as a scalar broadcast argument (same pattern as `n_arr` in `kern_rsqrt_mean`).

## Constraint: N <= 64

The fused kernel uses a single block of 64 threads. All N elements must fit in the registers of one block.

Microgpt use cases — all satisfy N <= 64:

| Use site | N | Fits? |
|----------|---|-------|
| Attention scores per head | seq_len (1–16) | Yes |
| Output logits | vocab_size = 27 | Yes |
| Inference sampling | vocab_size = 27 | Yes |

For N > 64, the existing 5-kernel version is still required.

## Memory comparison

| Version | DRAM reads | DRAM writes | Launches |
|---------|-----------|-------------|---------|
| 5-kernel | 5×N + 4 scalars | 4×N + 2 scalars | 5 |
| Fused | N | N | 1 |

## Files

| File | What |
|------|------|
| `docs/fused-softmax.md` | This design doc |
| `examples/fused_softmax_test.py` | Standalone test vs NumPy |
