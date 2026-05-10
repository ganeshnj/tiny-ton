# 19 — K3 Swizzled GEMM: XOR Shmem Swizzle

## Overview

K3 extends K2 by remapping shared-memory addresses with a bitwise XOR swizzle,
eliminating bank conflicts that arise when multiple threads access the same shmem
bank in parallel. It also adds bitwise integer ops (`^`, `&`, `>>`) to the
TinyTon dialect so the swizzle formula can be expressed in the Python DSL.

| Feature | What it unlocks |
|---|---|
| **Bitwise ops in TinyTon dialect** — `BitXorOp / BitAndOp / BitShrOp` | XOR/AND/SHR on i32 indices |
| **XOR swizzle pattern** — `sw = tid ^ ((tid >> 3) & 7)` | Conflict-free shmem bank mapping |

---

## Background: shmem banks

NVIDIA shared memory is divided into **32 banks**, each 4 bytes wide. A float at
slot `i` lives in bank `i % 32`. A warp of 32 threads can read from **all 32
banks simultaneously in one cycle**. When two or more threads in the same warp
target the *same* bank, the hardware serialises those accesses:
a *k*-way conflict takes *k* cycles instead of 1.

---

## Problem: conflict in the 2D thread layout (K5 prerequisite)

K3 is preparatory infrastructure for K5, which assigns `BM` threads to compute
`BM` output rows of C in parallel. In that layout, thread `t` stores A row `t`
into shared memory with a row-major stride of `BK` floats per row:

```
slot(thread t, step k) = t * BK + k      (row-major, BK floats per row)
bank = slot % 32 = (t * BK + k) % 32
```

**Concrete example — BM = 8 threads, BK = 8, k = 0**
(using 8 banks for clarity; the pattern scales to 32 on real hardware)

```
t   slot = t*8   bank = slot%8
0      0          0
1      8          0   ← conflict with t=0!
2     16          0   ← conflict!
3     24          0   ← conflict!
4     32          0
5     40          0
6     48          0
7     56          0   ← all 8 threads hit bank 0 → 8-way conflict
```

Root cause: stride `BK = 8` is a multiple of the bank count 8, so every row
lands on the same bank. On real hardware (32 banks, `BK = 16`):

```
stride 16 % 32 = 16  →  rows 0, 2, 4, … share bank k%32
                         rows 1, 3, 5, … share bank (k+16)%32
                     →  8 threads fight over 2 banks → 8-way conflict per A read
```

K2's `BK`-thread sequential layout avoids this because `tid = 0..15` maps to
banks `0..15` (all distinct). The conflict only surfaces once we add parallel
`BM`-thread row access in K5. K3 adds the swizzle now so K5 gets it for free.

---

## Swizzle formula — worked example

```
swizzled(i) = i ^ ((i >> 3) & 7)
```

The shift `>> 3` extracts the *row* index (for BK=8, 8 slots per row = 3 bits),
and XOR-ing it back into the *column* bits rotates columns differently per row,
spreading them across all banks.

**For i = 0..15 (BK = 16, physical bank = swizzled % 32):**

```
i   i>>3  &7   swizzled   bank
0     0    0      0         0
1     0    0      1         1
2     0    0      2         2
3     0    0      3         3
4     0    0      4         4
5     0    0      5         5
6     0    0      6         6
7     0    0      7         7
8     1    1      9         9
9     1    1      8         8
10    1    1     11        11
11    1    1     10        10
12    1    1     13        13
13    1    1     12        12
14    1    1     15        15
15    1    1     14        14
```

All 16 swizzled values are **distinct** → 16 different banks → **0 conflicts**.

**Re-running the 2D-thread example with swizzle:**

```
swizzled slot for (t, k=0) = swizzled(t * BK)
t=0 → swizzled(0)  =  0 → bank  0
t=1 → swizzled(8)  =  9 → bank  9   ← different from t=0!
t=2 → swizzled(16) = 17 → bank 17
t=3 → swizzled(24) = 27 → bank 27
...  all distinct → 1 transaction
```

**Bijection proof:** XOR with a constant is its own inverse, so loading with
the same swizzle formula retrieves the correct element:
`swizzled(swizzled(i)) = i`.

---

## Algorithm (diff from K2)

Only the shared-memory index changes; global loads, accumulation, and stores
are identical to K2.

**K2:**
```python
tt.shared_store(tid,      a_val, buffer_size=2*BK)
a_sh = tt.shared_load(tid,      buffer_size=2*BK)
tt.shared_store(BK + tid, b_val, buffer_size=2*BK)
b_sh = tt.shared_load(BK + tid, buffer_size=2*BK)
```

**K3:**
```python
sw = tid ^ ((tid >> 3) & 7)
tt.shared_store(sw,      a_val, buffer_size=2*BK)
a_sh = tt.shared_load(sw,      buffer_size=2*BK)
tt.shared_store(BK + sw, b_val, buffer_size=2*BK)
b_sh = tt.shared_load(BK + sw, buffer_size=2*BK)
```

---

## Architecture: Python DSL → PTX

```
tid >> 3   →  ast.RShift  → BitShrOp → arith.shrui → shr.u32
& 7        →  ast.BitAnd  → BitAndOp → arith.andi  → and.b32
^ (...)    →  ast.BitXor  → BitXorOp → arith.xori  → xor.b32
st/ld.shared use swizzled address → conflict-free
```

---

## Compiler changes

| File | Change |
|---|---|
| `include/tiny-ton/Dialect/TinyTon/TinyTonOps.td` | Add `BitXorOp`, `BitAndOp`, `BitShrOp` |
| `include/tiny-ton/IR/Builder.h` | Declare `emitBitXor`, `emitBitAnd`, `emitBitShr` |
| `lib/IR/Builder.cpp` | Implement the three `emit*` methods |
| `lib/Conversion/TinyTonToGPU.cpp` | Lower to `arith.xori`, `arith.andi`, `arith.shrui` |
| `bindings/python_bindings.cpp` | Expose `emit_bit_xor`, `emit_bit_and`, `emit_bit_shr` |
| `python/tiny_ton/jit.py` | Handle `ast.BitXor`, `ast.BitAnd`, `ast.RShift` in `_eval` |

---

## Kernel source

```python
@tt.jit
def swizzled_gemm(A_ptr, B_ptr, C_ptr, N, K,
                  BM: tt.constexpr, BN: tt.constexpr, BK: tt.constexpr):
    # A is M×K, B is K×N, C is M×N
    # Grid (M//BM, N//BN)  ·  BK threads/block
    # Shmem: 2×BK floats — swizzled layout eliminates bank conflicts
    bm  = tt.program_id(0)
    bn  = tt.program_id(1)
    tid = tt.arange(0, BK)

    for tm in range(BM):         # constexpr — unrolled
        for tn in range(BN):     # constexpr — unrolled
            a_row = A_ptr + (bm * BM + tm) * K
            b_col = B_ptr + (bn * BN + tn)
            c_out = C_ptr + (bm * BM + tm) * N + (bn * BN + tn)

            acc = 0.0
            for k0 in range(0, K, BK):    # runtime scf.for
                a_val = tt.load(a_row + k0 + tid)
                b_val = tt.load(b_col + (k0 + tid) * N)
                sw = tid ^ ((tid >> 3) & 7)
                tt.shared_store(sw,      a_val, buffer_size=2*BK)
                tt.shared_store(BK + sw, b_val, buffer_size=2*BK)
                tt.sync()
                a_sh = tt.shared_load(sw,      buffer_size=2*BK)
                b_sh = tt.shared_load(BK + sw, buffer_size=2*BK)
                acc  = acc + tt.reduce_sum(a_sh * b_sh)
                tt.sync()
            tt.store(c_out, acc)
```

---

## Key result

**Correctness:** `max_err` identical to K2 (swizzle is a bijection — same
data, different layout).

**Conflict reduction:**

| Layout | BM=8, BK=8 | BM=16, BK=16 |
|---|---|---|
| Linear (K2) | 8-way conflict | 8-way conflict |
| XOR swizzled (K3) | 0 conflicts | 0 conflicts |

---

## Next: K4 Vec-load GEMM

128-bit vectorised global loads (`float4`, 4 floats per transaction) to saturate
memory bandwidth from global memory:
```python
a0, a1, a2, a3 = tt.load4(a_row + k0 + tid * 4)
```
This requires a `LoadVec4Op` in the TinyTon dialect and LLVM vector lowering.
