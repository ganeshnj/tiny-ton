# 17 — K0: Naive GEMM

## Goal

Implement the simplest correct GEMM kernel in the tiny-ton Python DSL — a single thread
per output element, no shared memory, no tiling.  This is the starting point for the
"Road to cuBLAS" optimization series and mirrors the K0 baseline in Modular's matmul
blog series.

---

## Kernel structure

```
Grid: (M * N,)        — one block (= one thread group) per output element
Block: TILE_K threads — each thread handles one element of the K reduction

C[row, col] = sum_k A[row, k] * B[k, col]
```

The flat `pid` from `tt.program_id(0)` is decomposed into a 2-D index using the
new `//` and `%` operators:

```python
@tt.jit
def naive_gemm(A_ptr, B_ptr, C_ptr,
               M: tt.constexpr, N: tt.constexpr, K: tt.constexpr,
               TILE_K: tt.constexpr):
    pid  = tt.program_id(0)
    row  = pid // N          # ← requires ast.FloorDiv support in jit.py
    col  = pid % N           # ← requires ast.Mod support in jit.py
    tid  = tt.arange(0, TILE_K)
    acc  = 0.0
    for k0 in range(0, K, TILE_K):
        a_tile = tt.load(A_ptr + row * K + k0 + tid)
        b_tile = tt.load(B_ptr + (k0 + tid) * N + col)
        acc    = acc + tt.reduce_sum(a_tile * b_tile)
    tt.store(C_ptr + row * N + col, acc)
```

Key properties:

| Property | Value |
|----------|-------|
| Output parallelism | M × N blocks run in parallel |
| K reduction | constexpr-unrolled loop over TILE_K-wide tiles |
| Memory access | all global memory, no sharing between blocks |
| Compiler requirements | `//`, `%`, existing `reduce_sum`, `arange`, `load`, `store` |

---

## Why `//` and `%` are needed

The 1D `pid` must be mapped to `(row, col)` coordinates.  Without these operators
every kernel that operates on 2D or higher-dimensional outputs needs to express that
mapping with division and modulo.

The `jit.py` BinOp handler previously raised `NotImplementedError` for `ast.FloorDiv`
and `ast.Mod`.  The fix maps them onto the already-available `emit_div` and synthesises
`%` as `a - (a // b) * b` (correct for non-negative indices):

```python
# jit.py — _eval, ast.BinOp branch
if isinstance(node.op, ast.FloorDiv):
    return self.builder.emit_div(lhs, rhs)          # arith::DivSIOp
if isinstance(node.op, ast.Mod):
    q = self.builder.emit_div(lhs, rhs)
    return self.builder.emit_sub(lhs, self.builder.emit_mul(q, rhs))
```

No C++ rebuild is required — the change is pure Python.

---

## Expected performance (Jetson Orin Nano, sm_87)

K0 is deliberately not fast.  The purpose is correctness and a measurable baseline.

| Shape | Expected TFLOPS | cuBLAS FP32 | Gap |
|-------|----------------|-------------|-----|
| 64³   | < 0.001        | ~0.005      | > 5× |
| 128³  | < 0.001        | ~0.03       | > 30× |
| 256³  | ~0.001         | ~0.2        | > 200× |

The dominant cost at small sizes is the Python host loop + H2D + D2H overhead per
call, not actual arithmetic.  This gap closes in later stages when tile sizes grow.

---

## What K1 adds

K1 (`row_gemm`) moves from a flat grid to one block per output row and uses a
compile-time loop over K tiles.  It is the existing `tiled_matmul_kernel` renamed.

See `16-tiled-gemm.md` for the K1 implementation.
