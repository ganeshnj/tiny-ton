# tt.constexpr — Compile-Time Block Size

## The problem

Every tiny-ton kernel runs with a fixed block size of 64 threads, set by the
literal in `tt.arange(0, 64)`. This is fine for `n_embd=64`, but wasteful at
smaller sizes.

For example, at `n_embd=16`:

```
64 threads launched
48 threads masked out by `mask = tid < 16`
75% of GPU occupancy wasted
```

The mask prevents incorrect results, but idle threads still consume GPU
resources and dispatch slots.

## Solution: `PARAM: tt.constexpr`

Annotate a parameter with `tt.constexpr` to make it a compile-time constant.
Pass the desired block size as its value at the call site:

```python
@tt.jit
def fused_softmax_kernel(src, dst, N, BLOCK: tt.constexpr):
    tid  = tt.arange(0, BLOCK)   # block size set at compile time
    mask = tid < N
    x    = tt.load(src + tid, mask=mask, other=-float('inf'))
    mx   = tt.reduce_max(x)
    e    = tt.exp(x - mx)
    s    = tt.reduce_sum(e)
    tt.store(dst + tid, e / s, mask=mask)

# 16 threads, 0% idle
fused_softmax_kernel[(1,)](x, out, 16, 16)

# 64 threads (original behaviour)
fused_softmax_kernel[(1,)](x, out, 64, 64)
```

Each `BLOCK` value produces a **separate compiled kernel** — they are cached
independently and emit different PTX with different `blockX` in
`cuLaunchKernel`.

## Rules

- `BLOCK: tt.constexpr` must be an integer at call time.
- `tt.arange(0, BLOCK)` is the only place it may be used as a variable; all
  arithmetic on `BLOCK` must happen at the Python level before the call.
- The constexpr value is **not passed as a runtime kernel argument** —
  `cuLaunchKernel` never sees it. Only pointers and plain integer scalars
  (like `N`) are passed.
- Existing kernels with `tt.arange(0, 64)` (literal) are unchanged —
  `tt.constexpr` is purely additive.

## Thread utilization

| n_embd | Old (hardcoded 64) | New (BLOCK=n_embd) |
|---|---|---|
| 16 | 25% utilization | 100% |
| 32 | 50% utilization | 100% |
| 64 | 100% utilization | 100% |

## How it works

The change is entirely in [`python/tiny_ton/jit.py`](../python/tiny_ton/jit.py).
No C++ or MLIR changes are needed.

```
@tt.jit decoration
  └─ JITFunction.__init__
       └─ _find_constexpr_params()    ← parse AST annotations once

kernel[(grid,)](x, out, N, 16)
  └─ _make_key(args)                  ← ('constexpr', 16) in key
  └─ _compile(args)                   ← extract constexpr_values = {'BLOCK': 16}
       └─ KernelVisitor(constexpr_values=...)
            ├─ symbols['BLOCK'] = 16  (Python int, not MLIR Value)
            ├─ skip emit_arg for BLOCK
            └─ arange handler reads BLOCK from symbols → block_size = 16
  └─ cuLaunchKernel(gridX, blockX=16, ...)   ← 16 threads launched
```

### Cache key

Constexpr values are part of the cache key. `BLOCK=16` and `BLOCK=64` produce
separate cache entries and separate PTX:

```python
# cache key for BLOCK=16
(('ndarray', (16,), float32), ..., ('constexpr', 16))

# cache key for BLOCK=64
(('ndarray', (64,), float32), ..., ('constexpr', 64))
```

### Kernel argument indexing

Constexpr params are excluded from the IR argument list. With:

```python
def kernel(src, dst, N, BLOCK: tt.constexpr):
```

The PTX kernel only has three parameters: `src` (ptr, arg 0), `dst` (ptr, arg 1),
`N` (i32, arg 2). `BLOCK` is embedded in the compiled code as the block size
passed to `cuLaunchKernel` and as the range of `tt.arange`.

## Example: using BLOCK=N

For single-block reduction kernels where the entire vector fits in one block,
set `BLOCK = N` at the call site:

```python
def gpu_softmax(x, N):
    out = np.zeros(N, dtype=np.float32)
    fused_softmax_kernel[(1,)](x, out, N, N)   # exactly N threads
    return out
```

This is the optimal choice when `N <= 64`. For `N > 64`, split into multiple
blocks using `program_id` and pass a fixed `BLOCK` that divides the work.

## Test

[`examples/constexpr_test.py`](../examples/constexpr_test.py) covers:

- `BLOCK == N` (no idle threads)
- `BLOCK > N` (masked, correct output)
- Multiple BLOCK values → separate cache entries
- Backward compatibility with literal `arange(0, 64)` kernels
- Fused softmax correctness for N=4, 16, 27
