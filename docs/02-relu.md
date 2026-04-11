# relu (Rectified Linear Unit) — Design & Implementation

## What it does

`tt.relu(x)` is an element-wise activation function: each thread computes `max(x, 0.0)` on its own scalar value. Negative values become zero; positive values pass through unchanged.

```python
@tt.jit
def kernel(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    y = tt.relu(x)
    tt.store(dst + off, y, mask=mask)
```

## Why no new MLIR op

`relu(x)` is `max(x, 0.0)`. We already have `tinyton.max` working end-to-end (GPU + simulator, f32 + f16). Rather than adding a dedicated `TinyTon_ReluOp`, relu is composed at the Python JIT level by emitting:

```
zero = tinyton.fconst 0.0       (or tinyton.hconst 0.0 for f16)
result = tinyton.max x, zero
```

This keeps the C++ stack unchanged — no new op in `TinyTonOps.td`, no new builder method, no new GPU lowering pattern, no new simulator opcode.

## How it flows through the stack

```
Python: tt.relu(x)
    │
    ▼
JIT AST visitor (_eval_call)
    │  emit_fconst(0.0)  →  tinyton.fconst 0.0
    │  emit_max(x, zero) →  tinyton.max %x, %zero
    │
    ▼
GPU lowering (TinyTonToGPU.cpp, existing MaxOp pattern)
    │  tinyton.max → arith.maxnumf (float) or arith.maxsi (int)
    │
    ▼
NVVM/PTX (CombinedGPULoweringPass)
    │  arith.maxnumf → max.f32 in PTX
    │
    ▼
GPU execution: each thread computes max(x, 0.0) independently
```

For the simulator path:

```
Python: tt.relu(x)
    │
    ▼
CodeGen (CodeGen.cpp, existing MaxOp)
    │  FCONST 0.0 + FMAX opcodes
    │
    ▼
Simulator (Simulator.cpp)
    │  std::fmax(x, 0.0f)
```

## f16 handling

When `_kernel_dtype` is `"f16"`, the JIT emits `emit_hconst(0.0)` instead of `emit_fconst(0.0)`. The rest of the pipeline handles f16 identically to how it handles any `tinyton.max` on f16 operands (promote to f32 for the GPU math, truncate back).

## Files changed

| File | What |
|------|------|
| `python/tiny_ton/jit.py` | Add `"relu"` to `_BUILTINS`, dispatch in `_eval_call` |
| `python/tiny_ton/__init__.py` | Add `relu` stub + export |
| `examples/relu_test.py` | Standalone test vs `np.maximum(x, 0)` |

No C++ files changed.

## Testing strategy

`examples/relu_test.py` tests relu on f32 arrays containing a mix of positive, negative, and zero values. The expected output is `np.maximum(src, 0)`. Tolerance: exact match (relu has no numerical approximation).
