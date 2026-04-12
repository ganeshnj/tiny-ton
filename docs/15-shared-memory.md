# Shared Memory: tt.sync / tt.shared_store / tt.shared_load

## The problem

In the current `linear_kernel`, each block computes one output row of `y = W @ x`.
Every block loads the full `x` vector from global memory independently:

```
Block 0: load x[0..N-1] from global → dot with W[0,:]
Block 1: load x[0..N-1] from global → dot with W[1,:]
Block 2: load x[0..N-1] from global → dot with W[2,:]
...
```

The same `x` data is read `out_features` times from global memory. On real
GPUs the L2 cache usually handles this for small vectors, but for larger data
the redundant reads become a bottleneck.

Shared memory solves this by loading `x` once and letting all threads in a
block reuse it from fast on-chip storage.

## New primitives

```python
tt.sync()                    # barrier — all threads in the block wait here
tt.shared_store(idx, val)    # write val to shared memory at position idx
val = tt.shared_load(idx)    # read from shared memory at position idx
```

Shared memory is **per-block**: each block has its own buffer, sized
automatically to `BLOCK` (from `tt.arange(0, BLOCK)`).

## Execution model

```
Thread 0: load x[0] from global → shared_store(0, x[0])
Thread 1: load x[1] from global → shared_store(1, x[1])
...
Thread N-1: load x[N-1] from global → shared_store(N-1, x[N-1])

          ╔═══════════╗
          ║  tt.sync() ║  ← all threads wait here
          ╚═══════════╝

Thread 0: x_sh = shared_load(0..N-1)  → compute dot with W row
Thread 1: x_sh = shared_load(0..N-1)  → compute dot with W row
```

## Example: tiled 2-row-per-block matvec

Instead of 1 output row per block, each block computes 2 rows. The `x` vector
is loaded into shared memory once and reused for both dot products:

```python
@tt.jit
def tiled_linear_kernel(W_ptr, x_ptr, y_ptr, in_features, BLOCK: tt.constexpr):
    pid  = tt.program_id(0)
    tid  = tt.arange(0, BLOCK)
    mask = tid < in_features

    x_val = tt.load(x_ptr + tid, mask=mask)
    tt.shared_store(tid, x_val)
    tt.sync()
    x_sh = tt.shared_load(tid)

    w0   = tt.load(W_ptr + (pid * 2)     * in_features + tid, mask=mask)
    w1   = tt.load(W_ptr + (pid * 2 + 1) * in_features + tid, mask=mask)
    dot0 = tt.reduce_sum(w0 * x_sh)
    dot1 = tt.reduce_sum(w1 * x_sh)
    tt.store(y_ptr + pid * 2,     dot0)
    tt.store(y_ptr + pid * 2 + 1, dot1)
```

Launch with `grid = (out_features // 2,)` — half the blocks, each doing 2x work.

## MLIR lowering

| Python | TinyTon IR | GPU dialect |
|---|---|---|
| `tt.sync()` | `tinyton.sync` | `gpu.barrier` |
| `tt.shared_store(idx, val)` | `tinyton.shared_store %idx, %val size 64` | `memref.store` to workgroup memref |
| `tt.shared_load(idx)` | `tinyton.shared_load %idx size 64` | `memref.load` from workgroup memref |

The `size` attribute is the buffer size, baked in at compile time from
`block_size` (captured via `tt.arange`). The GPU lowering allocates a
`memref<size x f32, #gpu.address_space<workgroup>>` as a second workgroup
attribution (separate from the 32-element buffer used by `reduce_sum`/
`reduce_max`).

## Simulator

The simulator maintains a 256-element `sharedMem` vector per block. Instructions
are distinguished from regular `LDR`/`STR` by flag bits in the encoding:

- `SHMEM_STR`: opcode 0x8 with rd=1 (global STR has rd=0)
- `SHMEM_LDR`: opcode 0x7 with rt=1 (global LDR has rt=0)
- `SYNC`: opcode 0xF with imm=1 (RET has imm=0); pauses all threads and
  resumes them in the next phase, matching GPU barrier semantics.

## Files changed

| File | Change |
|---|---|
| `include/tiny-ton/Dialect/TinyTon/TinyTonOps.td` | `SyncOp`, `SharedStoreOp`, `SharedLoadOp` |
| `include/tiny-ton/IR/Builder.h` | `emitSync`, `emitSharedStore`, `emitSharedLoad` |
| `lib/IR/Builder.cpp` | Implementation |
| `lib/Conversion/TinyTonToGPU.cpp` | Pre-scan for buffer size, second workgroup memref, lowering |
| `lib/Compiler/CodeGen.cpp` | Simulator instruction encoding |
| `lib/Runtime/Simulator.cpp` | `sharedMem` buffer, `StepResult::Sync`, flag-based dispatch |
| `bindings/python_bindings.cpp` | Python bindings |
| `python/tiny_ton/jit.py` | `_BUILTINS`, `_eval_call` handlers |
| `python/tiny_ton/__init__.py` | Stubs |
| `examples/tiled_matvec_test.py` | Round-trip + tiled matvec tests |
| `docs/15-shared-memory.md` | This design doc |

## What this does NOT include

Full tiled GEMM requires iterating over K tiles inside the kernel via
`tt.for_range`, which generates `scf.for` in MLIR. That is a separate
future addition. This plan provides all the shared memory building blocks
that tiled GEMM needs.
