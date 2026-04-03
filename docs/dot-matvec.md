# dot / matvec (Dot Product and Matrix-Vector Multiply) — Design & Implementation

## What they do

**Dot product**: given two vectors `a` and `b` of length `N`, compute `sum(a[i] * b[i])` — a single scalar output.

**Matvec (matrix-vector multiply)**: given a matrix `W` of shape `(out_features, in_features)` and a vector `x` of length `in_features`, compute `y = W @ x` — a vector of length `out_features` where each element is a dot product of one row of `W` with `x`.

In microgpt, this is the `linear` function:

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

## Kernel design

### Dot product

A single block computes the full dot product. Each thread multiplies one pair of elements, then `tt.reduce_sum` combines across all threads:

```python
@tt.jit
def dot_kernel(a_ptr, b_ptr, dst_ptr, N):
    tid = tt.arange(0, 64)
    mask = tid < N
    a = tt.load(a_ptr + tid, mask=mask)
    b = tt.load(b_ptr + tid, mask=mask)
    result = tt.reduce_sum(a * b)
    tt.store(dst_ptr, result)
```

Launched with `grid = (1,)`.

### Matvec

Each block computes one output element — the dot product of one row of `W` with `x`:

```python
@tt.jit
def matvec_kernel(W_ptr, x_ptr, y_ptr, in_features):
    pid = tt.program_id(0)
    tid = tt.arange(0, 64)
    mask = tid < in_features
    w = tt.load(W_ptr + pid * in_features + tid, mask=mask)
    x = tt.load(x_ptr + tid, mask=mask)
    dot = tt.reduce_sum(w * x)
    tt.store(y_ptr + pid, dot)
```

Launched with `grid = (out_features,)` — one block per output row.

## Why no new MLIR op

Both operations compose existing primitives:

- `tt.load` — load vector elements
- `*` (via `tt.mul`) — element-wise multiply
- `tt.reduce_sum` — sum across threads in the block

No new entry in `TinyTonOps.td`, no builder method, no GPU lowering, no simulator opcode.

## How it flows through the stack

```
Python: tt.reduce_sum(w * x)
    │
    ▼
JIT AST visitor
    │  emit_load(W_ptr + pid*in_features + tid)  → tinyton.load (row of W)
    │  emit_load(x_ptr + tid)                     → tinyton.load (input x)
    │  emit_mul(w, x)                             → tinyton.mul
    │  emit_reduce_sum(product)                   → tinyton.reduce_sum
    │
    ▼
GPU lowering (existing patterns)
    │  tinyton.mul        → arith.mulf
    │  tinyton.reduce_sum → gpu.shuffle + shared memory + gpu.barrier
    │
    ▼
PTX: mul.f32 + shfl.sync.bfly + st.shared/ld.shared + bar.sync
```

## Constraint: in_features <= 64

The current block size is 64 threads. For `in_features > 64`, a single block cannot load the full row. microgpt uses `n_embd = 16` by default, well within this limit.

For larger dimensions, future work would add tiled loops or multi-block reductions.

## How it maps to microgpt

microgpt's `linear(x, w)` computes `out_features` dot products. With tiny-ton:

```python
# W is (out_features, in_features), stored as flat f32 array (row-major)
# x is (in_features,) f32 vector
# y is (out_features,) f32 output
matvec_kernel[(out_features,)](W_flat, x, y, in_features)
```

All attention projections (`attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`), MLP layers (`mlp_fc1`, `mlp_fc2`), and the LM head use this pattern.

## Files changed

| File | What |
|------|------|
| `docs/dot-matvec.md` | This design doc |
| `examples/dot_matvec_test.py` | Standalone test vs `np.dot` and `W @ x` |

No C++ files changed. No new builtins — these are user-written kernels.

## Testing strategy

`examples/dot_matvec_test.py` tests:

1. **Dot product**: two random f32 vectors, compare scalar output against `np.dot(a, b)` with `atol=1e-4` (float accumulation tolerance).
2. **Matvec**: random f32 matrix `W` and vector `x`, compare output vector against `W @ x` with `atol=1e-4`.
