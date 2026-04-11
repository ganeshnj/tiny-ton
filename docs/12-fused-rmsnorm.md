# Fused RMSNorm — Design & Implementation

## What it does

Root Mean Square Layer Normalization in a **single kernel launch**:

```
rmsnorm(x)[i] = x[i] / sqrt(mean(x^2) + eps)
```

## The problem with 4 kernels

The Stage 1 rmsnorm ([examples/rmsnorm_test.py](../examples/rmsnorm_test.py)) uses 4 separate kernel launches:

```
Launch 1: kern_square(x)           → writes x^2      to DRAM
Launch 2: kern_reduce_sum(x^2)     → writes sum       to DRAM   (reads x^2)
Launch 3: kern_rsqrt_mean(sum, N)  → writes scale     to DRAM   (reads sum, N)
Launch 4: kern_mul_scalar(x, scale)→ writes out       to DRAM   (reads x, scale)

Total: 4 launches, 7 DRAM round trips, 4 kernel startup overheads
```

## The fused kernel

All intermediate values stay in **thread registers**:

```
load x → square → reduce_sum → /N → +eps → rsqrt → scale → store

Total: 1 launch, 2 DRAM accesses (one load x, one store out), 1 kernel startup overhead
```

```python
@tt.jit
def fused_rmsnorm_kernel(src, dst, N, n_ptr):
    tid  = tt.arange(0, 64)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask)
    sq   = x * x
    s    = tt.reduce_sum(sq)          # warp-shuffle sum of squares
    n    = tt.load(n_ptr)             # scalar: float(N)
    scale = tt.rsqrt(s / n + 1e-5)   # 1 / sqrt(mean + eps)
    tt.store(dst + tid, x * scale, mask=mask)

def fused_rmsnorm(x, out, N):
    n_arr = np.array([float(N)], dtype=np.float32)
    fused_rmsnorm_kernel[(1,)](x, out, N, n_arr)  # always 1 block
```

No `other` parameter needed — masked threads load `0.0`, and `0^2 = 0` contributes nothing to `reduce_sum(sq)`.

## Constraint: N <= 64

Microgpt use case: n_embd = 16. Fits in one 64-thread block.

For N > 64 the existing 4-kernel version is still required.

## Memory comparison

| Version | DRAM reads | DRAM writes | Launches |
|---------|-----------|-------------|---------|
| 4-kernel | 3×N + 2 scalars | 2×N + 2 scalars | 4 |
| Fused | N + 1 scalar | N | 1 |

## Files

| File | What |
|------|------|
| `docs/12-fused-rmsnorm.md` | This design doc |
| `examples/fused_rmsnorm_test.py` | Standalone test vs NumPy and 4-kernel version |
