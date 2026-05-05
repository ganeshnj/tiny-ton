# 18 — K2 Shmem GEMM: 2D Grid + Runtime scf.for + Shared Memory Tiling

## Overview

K2 is the third GEMM kernel in tiny-ton's progression toward cuBLAS-level performance.
It introduces three new compiler features that are independently useful:

| Feature | What it unlocks |
|---|---|
| **2D grid launch** — `tt.program_id(1)` | Natural 2D indexing for matrix tiles |
| **Runtime `scf.for` loop** — `ForRangeOp` | K can be a runtime argument; no IR explosion |
| **Shared memory tiling** — `tt.shared_store / tt.shared_load` | Block-local caching of A and B tiles |

## Algorithm

```
Grid  : (M // TM, N // TN)   one block per output tile
Block : TK threads

For each block (bm, bn):
  acc = 0
  for k0 in range(0, K, TK):         ← runtime loop, K is not constexpr
    shmem_A[0..TK-1]  = A[bm, k0..k0+TK-1]   ← global → shared
    shmem_B[TK..2TK-1] = B[k0..k0+TK-1, bn]  ← global → shared
    sync()
    acc += reduce_sum(shmem_A * shmem_B)
    sync()
  C[bm, bn] = acc
```

With TM = TN = 1 (current), each block computes **one scalar** of C.
Increasing TM / TN (K3+) harvests the shared memory reuse benefit.

## Architecture: Python DSL → PTX

```
@tt.jit kernel
    │
    ▼  jit.py: visit_For detects runtime bounds → ForRangeOp
TinyTon IR
    tinyton.for_range %start, %K, %step iter_args(%acc = %init) {
      ^bb(%iv: i32, %acc_: f32): ... tinyton.yield %acc2
    }
    │
    ▼  TinyTonToGPU.cpp: lowerOneOp (recursive lambda)
GPU dialect + SCF
    scf.for %iv = %start to %K step %step iter_args(%acc = %init) -> f32 {
      ... scf.yield %acc2
    }
    │
    ▼  Pipeline.cpp: createConvertSCFToCFPass()
CF dialect
    cf.cond_br %done, ^exit, ^body
    ^body: ... cf.br ^header
    │
    ▼  CombinedGPULoweringPass → NVPTX backend
PTX
    $loop_header:
      setp.ge.s32 %p0, %k0, %K
      @%p0 bra $loop_exit
      ...
      add.s32 %k0, %k0, 16
      bra $loop_header
```

## Compiler changes

### Workstream A — 2D Grid

| File | Change |
|---|---|
| `CUDARuntime.h/.cpp` | `launch(gridX, gridY, blockX, ...)` — `gridY` was hardcoded 1 |
| `Simulator.h/.cpp` | `run(gridX, gridY, threadsPerBlock)` — PID axis=1 → `blockId / gridX` |
| `python_bindings.cpp` | Updated `launch` and `run` bindings |
| `jit.py` | `_launch_cuda` / `_launch_simulator` read `grid[1]` |

### Workstream B — Runtime scf.for

| File | Change |
|---|---|
| `TinyTonOps.td` | `ForRangeOp` (body region + iter_args) + `YieldOp` |
| `Builder.h/.cpp` | `beginForRange` / `endForRange` |
| `TinyTonToGPU.cpp` | Recursive `lowerOneOp` lambda converts `ForRangeOp` → `scf.for` |
| `CMakeLists.txt` (Conversion) | `MLIRSCFDialect` |
| `Pipeline.cpp` | `createConvertSCFToCFPass()` before `CombinedGPULoweringPass` |
| `CMakeLists.txt` (Compiler) | `MLIRSCFToControlFlow` |
| `python_bindings.cpp` | `begin_for_range` / `end_for_range` |
| `jit.py` | `visit_For` falls through to `ForRangeOp` when bounds are runtime |

## Kernel source

```python
@tt.jit
def shmem_gemm(A_ptr, B_ptr, C_ptr, M, N, K,
               TM: tt.constexpr, TN: tt.constexpr, TK: tt.constexpr):
    bm  = tt.program_id(0)
    bn  = tt.program_id(1)
    tid = tt.arange(0, TK)
    acc = 0.0
    for k0 in range(0, K, TK):
        a_val = tt.load(A_ptr + bm * TM * K + k0 + tid)
        tt.shared_store(tid, a_val, buffer_size=2*TK)
        b_val = tt.load(B_ptr + (k0 + tid) * N + bn * TN)
        tt.shared_store(TK + tid, b_val, buffer_size=2*TK)
        tt.sync()
        a_sh = tt.shared_load(tid,      buffer_size=2*TK)
        b_sh = tt.shared_load(TK + tid, buffer_size=2*TK)
        acc  = acc + tt.reduce_sum(a_sh * b_sh)
        tt.sync()
    tt.store(C_ptr + bm * TM * N + bn * TN, acc)
```

## Key result: compile time vs K1

| Kernel | K=128 compile | K=256 compile |
|---|---|---|
| K1 Row GEMM (constexpr unroll) | ~200 ms | **hangs (>60 s)** |
| K2 Shmem GEMM (runtime scf.for) | ~200 ms | ~200 ms |

K2's IR size is **constant** regardless of K — `scf.for` emits a single loop
instead of N×(K/TK) unrolled copies of the body.

## Next: K3 Swizzled GEMM

XOR-swizzle the shared memory addresses to eliminate bank conflicts:
```python
swizzled_idx = idx ^ ((idx >> 3) & 0x7)
tt.shared_store(swizzled_idx, val, buffer_size=2*TK)
```
This requires an address-swizzle helper in the JIT and no new compiler passes.
