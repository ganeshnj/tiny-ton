# Fused Attention — Design & Implementation

## What it does

Single-head scaled dot-product attention with Q/K/V/O projections, reducing kernel launches from 12 to 7 by fusing the score-scaling and softmax steps.

## The problem with 12 launches

The Stage 1 attention ([examples/attention_test.py](../examples/attention_test.py)) decomposes into 12 separate kernel launches:

```
Launches  1-3:  linear_kernel x3           (Wq@x, Wk@x, Wv@x)     — 3 launches
Launch    4:    matvec_kernel              (K @ q → scores)        — 1 launch
Launch    5:    kern_div_scalar            (scores / sqrt(d))      — 1 launch
Launches  6-10: softmax (5 kernels)        (reduce_max, sub, exp,  — 5 launches
                                            reduce_sum, div)
Launch   11:    matvec_kernel              (V^T @ weights)         — 1 launch
Launch   12:    linear_kernel              (Wo @ attn_out)         — 1 launch
                                                            Total:  12 launches
```

## What we fuse

### Step A: Replace 5-kernel softmax with fused softmax (5 → 1)

The fused softmax kernel ([examples/fused_softmax_test.py](../examples/fused_softmax_test.py)) collapses reduce_max + sub + exp + reduce_sum + div into a single launch using `other=-float('inf')`.

### Step B: Fold score scaling into softmax (2 → 1)

The `kern_div_scalar` (launch 5) divides scores by `sqrt(d)`. This is folded into a `fused_scaled_softmax_kernel` that loads raw scores, divides by sqrt_d, then does the full softmax — all in one kernel:

```python
@tt.jit
def fused_scaled_softmax_kernel(src, dst, N, sqrt_d_ptr):
    tid  = tt.arange(0, 64)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask, other=-float('inf'))
    sd   = tt.load(sqrt_d_ptr)
    x    = x / sd
    mx   = tt.reduce_max(x)
    e    = tt.exp(x - mx)
    s    = tt.reduce_sum(e)
    tt.store(dst + tid, e / s, mask=mask)
```

### Combined result: 12 → 7 launches

```
Launches  1-3:  linear_kernel x3              — 3 launches (unchanged)
Launch    4:    matvec_kernel (K @ q)          — 1 launch  (unchanged)
Launch    5:    fused_scaled_softmax           — 1 launch  (was 6: div + 5 softmax)
Launch    6:    matvec_kernel (V^T @ w)        — 1 launch  (unchanged)
Launch    7:    linear_kernel (Wo @ out)       — 1 launch  (unchanged)
                                        Total:  7 launches
```

## What cannot be fused (without C++ changes)

The 3 linear projections and 2 matvec launches each use `reduce_sum(w * x)` per output element — each block produces one scalar. Fusing multiple rows into one kernel would require shared memory and multi-row processing (Layer 4: tiled matmul).

## Constraint

Score vector size = seq_len <= 64. Microgpt: max seq_len = 16.

## Files

| File | What |
|------|------|
| `docs/13-fused-attention.md` | This design doc |
| `examples/fused_attention_test.py` | Standalone test vs NumPy |
