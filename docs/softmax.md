# softmax — Design & Implementation

## What it does

`softmax(x)` converts a vector of raw scores into a probability distribution. Each output element is `exp(x_i - max(x)) / sum(exp(x_j - max(x)))`. Subtracting the max first ensures numerical stability (prevents overflow in `exp`).

In microgpt, softmax is used in attention (to normalize attention scores) and in cross-entropy loss (to convert logits to probabilities).

## 5-kernel decomposition

Softmax is composed from 5 separate kernel launches — no fusion, no new MLIR ops:

```
Step 1:  m = reduce_max(x)          →  kern_reduce_max   (1 block)
Step 2:  shifted = x - m            →  kern_sub_scalar    (N/64 blocks)
Step 3:  exps = exp(shifted)        →  kern_exp           (N/64 blocks)
Step 4:  s = reduce_sum(exps)       →  kern_reduce_sum    (1 block)
Step 5:  out = exps / s             →  kern_div_scalar    (N/64 blocks)
```

Steps 1 and 4 produce a single scalar (stored in a 1-element array).
Steps 2, 3, 5 are element-wise — they operate on every element independently.

## Kernel code

Kernels 1, 3, 4 already exist in the codebase. Kernels 2 and 5 introduce a **scalar broadcast** pattern:

```python
@tt.jit
def kern_sub_scalar(src, scalar_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    s = tt.load(scalar_ptr)          # all threads load same scalar
    tt.store(dst + off, x - s, mask=mask)

@tt.jit
def kern_div_scalar(src, scalar_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    s = tt.load(scalar_ptr)          # all threads load same scalar
    tt.store(dst + off, x / s, mask=mask)
```

`tt.load(scalar_ptr)` without a mask is a maskless load — every thread loads from the same address, broadcasting the scalar to all threads.

## Python orchestrator

```python
def softmax(x, out, N):
    grid = (max(1, (N + 63) // 64),)
    tmp_max = np.zeros(1, dtype=x.dtype)
    tmp_exp = np.zeros(N, dtype=x.dtype)
    tmp_sum = np.zeros(1, dtype=x.dtype)

    kern_reduce_max[(1,)](x, tmp_max, N)
    kern_sub_scalar[grid](x, tmp_max, tmp_exp, N)
    kern_exp[grid](tmp_exp, tmp_exp, N)
    kern_reduce_sum[(1,)](tmp_exp, tmp_sum, N)
    kern_div_scalar[grid](tmp_exp, tmp_sum, out, N)
```

## Constraint: N <= 64

`reduce_max` and `reduce_sum` produce one scalar per block. With block size 64, the entire vector must fit in a single block for the reductions. microgpt uses `n_embd = 16` and `n_heads` (small), so this is not a limitation. Multi-block reductions are Stage 3 work.

## How it maps to microgpt

microgpt's softmax:

```python
def softmax(x):
    max_val = max(x)
    e = [math.exp(xi - max_val) for xi in x]
    s = sum(e)
    return [ei / s for ei in e]
```

With tiny-ton, this becomes 5 kernel launches as shown above.

## How it flows through the stack

```
Python orchestrator: 5 x kernel[grid](...)
    │
    ▼
JIT (for each kernel): existing builtins
    │  reduce_max  → tinyton.reduce_max
    │  load + sub  → tinyton.load + tinyton.sub
    │  exp         → tinyton.exp
    │  reduce_sum  → tinyton.reduce_sum
    │  load + div  → tinyton.load + tinyton.div
    │
    ▼
GPU lowering (existing patterns)
    │  reduce_max/sum → gpu.shuffle + shared memory + gpu.barrier
    │  sub/div/exp    → arith.subf / arith.divf / math.exp
    │
    ▼
PTX: 5 separate kernel launches via cuLaunchKernel
```

## Files changed

| File | What |
|------|------|
| `docs/softmax.md` | This design doc |
| `examples/softmax_test.py` | Standalone test: 5-launch softmax vs NumPy |

No C++ files changed. No new builtins in `jit.py`. All kernels are user-written.

## Testing strategy

`examples/softmax_test.py` creates random f32 vectors of length 16 (matching microgpt's `n_embd`), runs the 5-launch softmax, and compares against the NumPy reference: `exp(x - max(x)) / sum(exp(x - max(x)))`. Tolerance: `atol=1e-5` (float accumulation across 5 kernels). Also verifies the output sums to 1.0 and all values are in [0, 1].
