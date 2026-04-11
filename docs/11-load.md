# tt.load — Memory Load Operation

## Signature

```python
tt.load(ptr, mask=None, other=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `int` / address expression | Memory address to read from |
| `mask` | `int` (0 or 1), optional | Per-thread predicate: load only if nonzero |
| `other` | `float` / `int`, optional | Fallback value for masked-out threads (default `0.0` / `0`) |

Returns: the loaded value (or `other` if the thread is masked out).

## Two usage patterns

### 1. Vector load with tail masking

The standard pattern for loading N elements into a 64-thread block:

```python
tid  = tt.arange(0, 64)     # thread index 0..63
mask = tid < N               # only threads 0..N-1 are valid
x    = tt.load(src + tid, mask=mask, other=-float('inf'))
```

Threads with `tid >= N` get `other` instead of reading memory. This is necessary because the block size (64) is typically larger than the data size N.

### 2. Scalar broadcast load

Load a single value and broadcast to all threads:

```python
s = tt.load(scalar_ptr)      # no mask, no other — every thread reads the same address
```

Used to broadcast scalars like reduction results or constants. Examples: `kern_sub_scalar`, `kern_div_scalar`, `kern_rsqrt_mean`.

## The `mask` parameter

When `mask` is provided:
- Threads where `mask != 0` execute the memory load
- Threads where `mask == 0` skip the load and get `other` (or `0.0`/`0` if `other` is not specified)

This serves as **tail masking**: when N < block_size, threads beyond the data boundary must not read out-of-bounds memory.

## The `other` parameter

The fallback value for masked-out threads. This matters when masked threads participate in subsequent collective operations like `reduce_max` or `reduce_sum`.

### Why it matters

Without `other`, masked threads get `0.0`. This causes two bugs:

**Bug 1: reduce_max with all-negative inputs**

```python
x = tt.load(src + tid, mask=mask)   # masked threads get 0.0
mx = tt.reduce_max(x)               # 0.0 > all negatives → wrong max!
```

Fix: `other=-float('inf')` — negative infinity never wins a max comparison.

**Bug 2: fused reductions (softmax)**

```python
x  = tt.load(src + tid, mask=mask)  # masked threads get 0.0
mx = tt.reduce_max(x)
e  = tt.exp(x - mx)                 # masked: exp(0.0 - mx) ≠ 0
s  = tt.reduce_sum(e)               # sum polluted by (64-N) × exp(-mx)
```

Fix: `other=-float('inf')` — `exp(-inf - mx) = 0.0`, contributing nothing to the sum.

## Supported dtypes

The dtype is inferred from the kernel's argument types (determined at JIT compile time):

| Dtype | `other` default | `-float('inf')` representation |
|-------|----------------|-------------------------------|
| `f32` | `0.0f` | `0xFF800000` (IEEE 754 -inf) |
| `f16` | `0.0` (half) | `0xFC00` (IEEE 754 -inf) |
| `i32` | `0` | N/A (inf is float-only) |

## Lowering path

```
Python kernel source
  │  @tt.jit decorator captures AST
  ▼
jit.py _eval_call("load")
  │  Parses ptr, mask=, other= kwargs
  │  Evaluates other via _eval (handles -float('inf'))
  ▼
python_bindings.cpp emit_load(addr, mask, other, dtype)
  │  Converts PyValue → mlir::Value
  ▼
Builder.cpp emitLoad(addr, mask, other, elemType)
  │  Creates tinyton::LoadOp with optional other operand
  ▼
MLIR: tinyton.load %addr, %mask, %other : f32
  │
  ├──▶ GPU path (TinyTonToGPU.cpp):
  │      if mask:
  │        cf.cond_branch mask → then_block / merge_block(other)
  │        then_block: llvm.load → branch merge_block(loaded)
  │        merge_block: result = block_arg (loaded or other)
  │      else:
  │        llvm.load
  │
  └──▶ Simulator path (CodeGen.cpp):
         if mask:
           MOV rd, r_other   (or FCONST/CONST if no other)
           BZ r_mask, #1     (skip load if mask=0)
         LDR rd, [r_addr]
```

## Before / after: fused softmax

### Before (`other` not available)

```python
@tt.jit
def fused_softmax_kernel(src, dst, N, n_masked_ptr):
    tid  = tt.arange(0, 64)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask)          # masked → 0.0
    mx   = tt.reduce_max(x)
    e    = tt.exp(x - mx)
    s    = tt.reduce_sum(e)
    nm   = tt.load(n_masked_ptr)                  # float(64 - N)
    s    = s - nm * tt.exp(0.0 - mx)              # subtract masked pollution
    tt.store(dst + tid, e / s, mask=mask)

def fused_softmax(x, out, N):
    n_masked = np.array([float(64 - N)], dtype=np.float32)
    fused_softmax_kernel[(1,)](x, out, N, n_masked)   # 4 args
```

### After (`other=-float('inf')`)

```python
@tt.jit
def fused_softmax_kernel(src, dst, N):
    tid  = tt.arange(0, 64)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask, other=-float('inf'))
    mx   = tt.reduce_max(x)
    e    = tt.exp(x - mx)                         # masked: exp(-inf) = 0.0
    s    = tt.reduce_sum(e)                        # only real values
    tt.store(dst + tid, e / s, mask=mask)

def fused_softmax(x, out, N):
    fused_softmax_kernel[(1,)](x, out, N)             # 3 args, no correction
```

No extra host-side scalar. No correction formula.

## Edge cases

- **`float('inf')` in the JIT**: The expression `-float('inf')` is parsed as `UnaryOp(USub, Call(float, 'inf'))`. The JIT handles `float('inf')` as a special-case call that emits an `fconst(inf)`, and the `USub` negates it.
- **`other` without `mask`**: Ignored — if there's no mask, every thread does the load.
- **`other` type mismatch**: The `other` value must match the kernel's element type. Passing an int `other` to an f32 kernel is an error.
