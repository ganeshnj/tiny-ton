# 20 — K4 Vec GEMM: 128-bit Vectorized Global Loads

## Overview

K4 extends K3 by widening global loads from 32-bit scalars to 128-bit vectors
(4 × f32 per thread). This maps to the PTX `ld.global.v4.f32` instruction
(`LDG.128`), reducing global-memory transaction count by 4× for the A tile.

| Feature | What it unlocks |
|---|---|
| **`LoadVec4Op` in TinyTon dialect** | 128-bit global load → 4 f32 results |
| **`tt.load4` builtin in JIT** | Python-level `a0,a1,a2,a3 = tt.load4(addr)` |

---

## Problem: narrow global loads in K3

Each K3 thread issues one 32-bit `LDG.32` per k-step for A:

```python
a_val = tt.load(a_row + k0 + tid)   # 1 float = 4 bytes per thread
```

With BK=16 threads, that is 16 separate load instructions for 64 bytes of A
data. The GPU memory controller coalesces these into cache-line transactions,
but the 16 individual instructions still consume instruction-issue slots and
register-write ports.

---

## Concrete example — BK = 4 threads, loading 16 floats

**K3 (scalar):**

```
tid=0  → load A[row, k0+0]   LDG.32   4 bytes
tid=1  → load A[row, k0+1]   LDG.32   4 bytes
tid=2  → load A[row, k0+2]   LDG.32   4 bytes
tid=3  → load A[row, k0+3]   LDG.32   4 bytes
  → 4 load instructions for 16 bytes
```

**K4 (vec4):**

```
tid=0  → load4(A[row, k0+0]) → (v0,v1,v2,v3)   LDG.128   16 bytes
  → 1 load instruction for 16 bytes (same data, 4× fewer instructions)
```

Each `LDG.128` fetches 4 consecutive floats in a single 128-bit transaction.
The address must be 16-byte aligned — guaranteed because `cudaMalloc` returns
256-byte-aligned pointers and the k-loop step is a multiple of 4.

---

## Why B stays scalar

B is accessed column-wise: `B[k, bn]` → stride N between elements. Adjacent
floats in memory belong to different columns, so a 128-bit load would fetch
3 unwanted elements for every 1 wanted. B stays `LDG.32` until a future kernel
transposes B into a layout where columns are contiguous.

---

## Algorithm (diff from K3)

Only the A-load changes; B-load, swizzle, shmem, accumulation, and store are
identical.

**K3:**
```python
a_val = tt.load(a_row + k0 + tid)
# ... shmem A: BK slots
# K-loop step: BK
```

**K4:**
```python
a0, a1, a2, a3 = tt.load4(a_row + k0 + tid * 4)
# store 4 A-values into swizzled shmem: slots 0 .. 4*BK-1
for i, av in enumerate([a0, a1, a2, a3]):
    sw = (tid * 4 + i) ^ (((tid * 4 + i) >> 3) & 7)
    tt.shared_store(sw, av, buffer_size=8*BK)
# B: 4 scalar loads for 4*BK values, stored into 4*BK .. 8*BK-1
for j in range(4):
    b_val = tt.load(b_col + (k0 + j * BK + tid) * N)
    bidx = j * BK + tid
    bsw = bidx ^ ((bidx >> 3) & 7)
    tt.shared_store(4*BK + bsw, b_val, buffer_size=8*BK)
# shmem: 8*BK floats (4*BK A + 4*BK B)
# K-loop step: 4*BK
```

---

## Architecture: Python DSL → PTX

```
tt.load4(addr)  →  jit.py _eval_call → emit_load_vec4(addr)
                →  TinyTon IR:  tinyton.load_vec4 %addr → f32, f32, f32, f32
                →  TinyTonToGPU.cpp:
                     LLVM::LoadOp <4 x f32> from ptr addrspace(1)
                     4 × LLVM::ExtractElementOp (indices 0..3)
                →  NVPTX backend:
                     ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%ptr]
```

---

## Compiler changes

| File | Change |
|---|---|
| `include/tiny-ton/Dialect/TinyTon/TinyTonOps.td` | Add `LoadVec4Op` (addr → 4 × f32) |
| `include/tiny-ton/IR/Builder.h` | Declare `emitLoadVec4` returning `std::array<Value,4>` |
| `lib/IR/Builder.cpp` | Implement `emitLoadVec4` — create `LoadVec4Op`, return 4 results |
| `lib/Conversion/TinyTonToGPU.cpp` | Lower to `LLVM::LoadOp` with `vector<4xf32>` + 4 `ExtractElementOp` |
| `bindings/python_bindings.cpp` | Expose `emit_load_vec4` returning list of 4 PyValues |
| `python/tiny_ton/jit.py` | Add `tt.load4` builtin + tuple-unpack in `visit_Assign` |

---

## Kernel source

```python
@tt.jit
def vec_gemm(A_ptr, B_ptr, C_ptr, N, K,
             BM: tt.constexpr, BN: tt.constexpr, BK: tt.constexpr):
    # A is M×K, B is K×N, C is M×N
    # Grid (M//BM, N//BN)  ·  BK threads/block
    # Each thread loads 4 A-floats per k-step via LDG.128
    # Shmem: 8×BK floats (4×BK for A vec4 + 4×BK for B scalar)
    bm  = tt.program_id(0)
    bn  = tt.program_id(1)
    tid = tt.arange(0, BK)

    for tm in range(BM):
        for tn in range(BN):
            a_row = A_ptr + (bm * BM + tm) * K
            b_col = B_ptr + (bn * BN + tn)
            c_out = C_ptr + (bm * BM + tm) * N + (bn * BN + tn)

            acc = 0.0
            for k0 in range(0, K, 4 * BK):     # step = 4×BK
                # Vectorized A load: 4 floats per thread
                a0, a1, a2, a3 = tt.load4(a_row + k0 + tid * 4)

                # Swizzled shmem store for A (4 slots per thread)
                sw0 = (tid * 4)     ^ (((tid * 4)     >> 3) & 7)
                sw1 = (tid * 4 + 1) ^ (((tid * 4 + 1) >> 3) & 7)
                sw2 = (tid * 4 + 2) ^ (((tid * 4 + 2) >> 3) & 7)
                sw3 = (tid * 4 + 3) ^ (((tid * 4 + 3) >> 3) & 7)
                tt.shared_store(sw0, a0, buffer_size=8*BK)
                tt.shared_store(sw1, a1, buffer_size=8*BK)
                tt.shared_store(sw2, a2, buffer_size=8*BK)
                tt.shared_store(sw3, a3, buffer_size=8*BK)

                # Scalar B loads: 4 batches of BK (stride-N, can't vectorize)
                for j in range(4):
                    b_val = tt.load(b_col + (k0 + j * BK + tid) * N)
                    bidx = j * BK + tid
                    bsw = bidx ^ ((bidx >> 3) & 7)
                    tt.shared_store(4 * BK + bsw, b_val, buffer_size=8*BK)
                tt.sync()

                # Accumulate: 4 sub-steps, each reduce_sum over BK threads
                for j in range(4):
                    aidx = j * BK + tid
                    asw = aidx ^ ((aidx >> 3) & 7)
                    a_sh = tt.shared_load(asw, buffer_size=8*BK)
                    bidx = j * BK + tid
                    bsw = bidx ^ ((bidx >> 3) & 7)
                    b_sh = tt.shared_load(4 * BK + bsw, buffer_size=8*BK)
                    acc = acc + tt.reduce_sum(a_sh * b_sh)
                tt.sync()
            tt.store(c_out, acc)
```

---

## Key result

**Correctness:** max_err identical to K3 — same data, loaded differently.

**Load efficiency:**

| | K3 | K4 |
|---|---|---|
| A load instruction | `LDG.32` | `LDG.128` |
| Instructions per A-tile (BK=16) | 16 | 4 |
| Bytes per instruction | 4 | 16 |

---

## Next: K5 Pipelined GEMM

`cp.async` to overlap global-to-shared loads with computation, hiding the
global-memory latency behind arithmetic in the previous k-step.
