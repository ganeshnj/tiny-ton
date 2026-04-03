# rmsnorm (Root Mean Square Layer Normalization) — Design & Implementation

## What it does

RMSNorm normalizes a vector by its root-mean-square value:

```
rms = sqrt(mean(x^2)) = sqrt(sum(x^2) / N)
rmsnorm(x) = x / rms  = x * rsqrt(sum(x^2) / N + eps)
```

The epsilon (`1e-5`) prevents division by zero when all inputs are zero.

In microgpt:

```python
def rmsnorm(x, w):
    n = sum(xi * xi for xi in x) / len(x)
    val = 1.0 / math.sqrt(n + 1e-5)
    return [wi * xi * val for wi, xi in zip(w, x)]
```

RMSNorm is applied before every attention and MLP block — it is the most frequently called normalization in the model.

## 4-kernel decomposition

RMSNorm is composed from 4 separate kernel launches — no fusion, no new MLIR ops:

```
Step 1:  sq    = x * x                    →  kern_square       (N/64 blocks)
Step 2:  s     = reduce_sum(sq)           →  kern_reduce_sum   (1 block)
Step 3:  scale = rsqrt(s / N + eps)       →  kern_rsqrt_mean   (1 block, scalar → scalar)
Step 4:  out   = x * scale               →  kern_mul_scalar    (N/64 blocks)
```

Steps 1 and 4 are element-wise. Steps 2 and 3 operate on scalars (1-element arrays).

## Kernel code

Kernel 2 (`kern_reduce_sum`) already exists in the codebase. The new kernels:

```python
@tt.jit
def kern_square(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    tt.store(dst + off, x * x, mask=mask)

@tt.jit
def kern_rsqrt_mean(sum_ptr, n_ptr, out_ptr):
    tid = tt.arange(0, 64)
    s = tt.load(sum_ptr)
    n = tt.load(n_ptr)
    mean_eps = s / n + 1e-5
    scale = tt.rsqrt(mean_eps)
    tt.store(out_ptr, scale)

@tt.jit
def kern_mul_scalar(src, scalar_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    s = tt.load(scalar_ptr)
    tt.store(dst + off, x * s, mask=mask)
```

`kern_rsqrt_mean` receives `N` as a 1-element f32 array to avoid int/float type mixing in the JIT. The Python orchestrator pre-converts `N` to float.

`kern_mul_scalar` uses the same scalar broadcast pattern as softmax's `kern_div_scalar`.

## Python orchestrator

```python
def rmsnorm(x, out, N):
    grid = (max(1, (N + 63) // 64),)
    tmp_sq  = np.zeros(N, dtype=x.dtype)
    tmp_sum = np.zeros(1, dtype=x.dtype)
    tmp_scl = np.zeros(1, dtype=x.dtype)
    n_arr   = np.array([float(N)], dtype=x.dtype)

    kern_square[grid](x, tmp_sq, N)
    kern_reduce_sum[(1,)](tmp_sq, tmp_sum, N)
    kern_rsqrt_mean[(1,)](tmp_sum, n_arr, tmp_scl)
    kern_mul_scalar[grid](x, tmp_scl, out, N)
```

## Constraint: N <= 64

Same as softmax — `reduce_sum` produces one scalar per block. microgpt uses `n_embd = 16`.

## How it maps to microgpt

microgpt applies rmsnorm before every attention and MLP block:

```python
x = rmsnorm(x, state_dict[f'layers.{l}.norm1'])
```

With tiny-ton, this becomes 4 kernel launches. The weight scaling (`w * x * val` in microgpt) would be an additional element-wise multiply kernel, but for Stage 1 we test the unweighted version first.

## How it flows through the stack

```
Python orchestrator: 4 x kernel[grid](...)
    │
    ▼
JIT (for each kernel): existing builtins
    │  mul (square)    → tinyton.mul
    │  reduce_sum      → tinyton.reduce_sum
    │  div + rsqrt     → tinyton.div + tinyton.rsqrt
    │  mul (broadcast) → tinyton.mul
    │
    ▼
GPU lowering (existing patterns)
    │  mul/div      → arith.mulf / arith.divf
    │  reduce_sum   → gpu.shuffle + shared memory + gpu.barrier
    │  rsqrt        → math.rsqrt → __nv_rsqrtf (libdevice)
    │
    ▼
PTX: 4 separate kernel launches via cuLaunchKernel
```

## Files changed

| File | What |
|------|------|
| `docs/rmsnorm.md` | This design doc |
| `examples/rmsnorm_test.py` | Standalone test: 4-launch rmsnorm vs NumPy |

No C++ files changed. No new builtins in `jit.py`. All kernels are user-written.

## Testing strategy

`examples/rmsnorm_test.py` creates random f32 vectors of length 16 and 32, runs the 4-launch rmsnorm, and compares against the NumPy reference: `x * (1 / sqrt(mean(x^2) + 1e-5))`. Tolerance: `atol=1e-5` (float accumulation across 4 kernels).
