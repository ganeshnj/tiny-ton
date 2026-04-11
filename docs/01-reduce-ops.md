# reduce_sum and reduce_max — Design & Implementation

## What they do

`tt.reduce_sum(x)` and `tt.reduce_max(x)` are cross-thread reduction operations. Every thread in a block contributes one scalar value; every thread receives the fully reduced result (the sum or max of all values across all threads in the block).

```python
@tt.jit
def kernel(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)   # 64 threads per block
    x = tt.load(src + off, mask=off < N)
    total = tt.reduce_sum(x)             # all 64 threads get the same sum
    tt.store(dst + pid, total)
```

## Architecture overview

The implementation spans four layers:

```
Python API (tt.reduce_sum)
    │
    ▼
TinyTon MLIR dialect (tinyton.reduce_sum)
    │
    ▼
GPU dialect lowering (gpu.shuffle + memref + gpu.barrier)
    │
    ▼
NVVM/PTX (nvvm.shfl.sync + st.shared/ld.shared + bar.sync)
```

### Layer 1: Python frontend

**Files:** `python/tiny_ton/__init__.py`, `python/tiny_ton/jit.py`

The `KernelVisitor` (AST visitor) recognizes `tt.reduce_sum(x)` and `tt.reduce_max(x)` as builtins and calls `builder.emit_reduce_sum(x)` / `builder.emit_reduce_max(x)` on the C++ `IRBuilder`.

### Layer 2: TinyTon MLIR dialect

**Files:** `include/tiny-ton/Dialect/TinyTon/TinyTonOps.td`, `include/tiny-ton/IR/Builder.h`, `lib/IR/Builder.cpp`

Two ops defined in TableGen:

```tablegen
def TinyTon_ReduceSumOp : TinyTon_Op<"reduce_sum",
    [AllTypesMatch<["operand", "result"]>]> {
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
}
```

`AllTypesMatch` tells MLIR the result type equals the operand type, avoiding the need for explicit type inference.

### Layer 3: GPU dialect lowering (the interesting part)

**File:** `lib/Conversion/TinyTonToGPU.cpp`

This is a two-phase reduction: intra-warp shuffle, then cross-warp via shared memory.

#### Why two phases?

A GPU block with 64 threads has 2 warps (warp = 32 threads). The `gpu.shuffle xor` instruction only communicates within a single warp. To reduce across the full block, we need shared memory to exchange partial results between warps.

#### Phase 1: Intra-warp butterfly reduction

```
for offset in {1, 2, 4, 8, 16}:
    shuffled = gpu.shuffle xor val, offset, 32
    val = val + shuffled       (or max)
```

This is a "butterfly" pattern. After 5 rounds, every thread in a warp holds that warp's partial sum. The XOR shuffle ensures all threads receive the result (not just lane 0).

**Example** (4 threads for simplicity, values [1, 2, 3, 4]):
```
Step 1: shuffle xor offset=1
  Thread 0 gets Thread 1's value → adds: 1+2 = 3
  Thread 1 gets Thread 0's value → adds: 2+1 = 3
  Thread 2 gets Thread 3's value → adds: 3+4 = 7
  Thread 3 gets Thread 2's value → adds: 4+3 = 7

Step 2: shuffle xor offset=2
  Thread 0 gets Thread 2's value → adds: 3+7 = 10
  Thread 1 gets Thread 3's value → adds: 3+7 = 10
  Thread 2 gets Thread 0's value → adds: 7+3 = 10
  Thread 3 gets Thread 1's value → adds: 7+3 = 10
```

All threads end up with 10 (the full sum). On real hardware this runs for 5 rounds (offsets 1,2,4,8,16) to cover all 32 lanes.

#### Phase 2: Cross-warp reduction via shared memory

After Phase 1, each warp has its own partial sum. With BLOCK=64 (2 warps):
- Warp 0 holds partial_0 (sum of threads 0-31)
- Warp 1 holds partial_1 (sum of threads 32-63)

```
warp_id = threadIdx.x / 32
lane_id = threadIdx.x % 32

shmem[warp_id] = val              // each warp writes its partial
gpu.barrier                        // __syncthreads()

num_warps = blockDim.x / 32
safe_idx = (lane_id < num_warps) ? lane_id : 0
val = (lane_id < num_warps) ? shmem[safe_idx] : identity

// second butterfly over the per-warp partials
for offset in {1, 2, 4, 8, 16}:
    shuffled = gpu.shuffle xor val, offset, 32
    val = val + shuffled
```

**Key insight:** after the barrier, *every* warp loads the same pattern from shared memory (partial_0 at index 0, partial_1 at index 1, identity elsewhere). So every warp runs the same second butterfly and arrives at the same correct final result. No broadcast step needed — all threads in all warps end up with the complete reduction.

#### Shared memory allocation

Shared memory is declared via `gpu.func` workgroup attributions:

```cpp
auto shmemTy = MemRefType::get(
    {32}, f32Ty, MemRefLayoutAttrInterface{}, addrSpace);
shmemArg = gpuFunc.addWorkgroupAttribution(shmemTy, loc);
```

This allocates 32 × f32 = 128 bytes of shared memory (enough for up to 32 warps = 1024 threads). The `#gpu.address_space<workgroup>` attribute maps to LLVM address space 3, which the NVPTX backend emits as `__shared__` memory.

#### Address space mapping

The `LLVMTypeConverter` must know the GPU-to-LLVM address space mapping, otherwise shared memory pointers end up in the wrong address space (causing GPU crashes):

```cpp
populateGpuMemorySpaceAttributeConversions(
    converter, [](gpu::AddressSpace space) -> unsigned {
      switch (space) {
      case gpu::AddressSpace::Global:    return 1;
      case gpu::AddressSpace::Workgroup: return 3;  // shared memory
      case gpu::AddressSpace::Private:   return 5;
      }
      return 0;
    });
```

**File:** `lib/Compiler/Pipeline.cpp`

#### Identity values

| Operation | Float identity | Int identity |
|-----------|---------------|-------------|
| reduce_sum | 0.0 | 0 |
| reduce_max | -infinity | INT_MIN |

Out-of-bounds lanes (where `lane_id >= num_warps`) get the identity value so they don't affect the result.

#### f16 handling

f16 values are promoted to f32 before reduction (the shuffle operates on f32), then truncated back to f16 after the final result. Shared memory is always f32.

### Layer 4: NVVM/PTX lowering

**File:** `lib/Compiler/Pipeline.cpp` (`CombinedGPULoweringPass`)

The pass pipeline converts GPU dialect ops to NVVM:
- `gpu.shuffle xor` → `nvvm.shfl.sync xor` → `shfl.sync.bfly.b32` in PTX
- `gpu.barrier` → `nvvm.barrier0` → `bar.sync 0` in PTX
- `memref.store/load` on workgroup memory → `st.shared.f32` / `ld.shared.f32` in PTX
- `gpu.func` workgroup attributions → `@shmem = addrspace(3) global [32 x float]`

## Simulator support

**File:** `lib/Runtime/Simulator.cpp`

The CPU simulator uses a two-phase execution model:
1. Run all threads until they hit a reduce op (each thread records its value)
2. Compute the reduction across all threads
3. Resume all threads with the reduced value

This simulates the barrier-synchronized collective without actual parallelism.

**File:** `lib/Compiler/CodeGen.cpp`

Reduce ops use extended opcode `0x0F` with type flags to distinguish sum vs max and float vs int.

## Files changed (complete list)

| File | What |
|------|------|
| `include/tiny-ton/Dialect/TinyTon/TinyTonOps.td` | Op definitions |
| `include/tiny-ton/IR/Builder.h` | Builder API declarations |
| `lib/IR/Builder.cpp` | Builder API implementations |
| `bindings/python_bindings.cpp` | Python bindings |
| `python/tiny_ton/__init__.py` | Python stubs |
| `python/tiny_ton/jit.py` | AST visitor (`_eval_call`, `visit_AnnAssign`) |
| `lib/Conversion/TinyTonToGPU.cpp` | GPU lowering (shuffle + shmem + barrier) |
| `lib/Compiler/Pipeline.cpp` | NVVM pass pipeline (memref-to-LLVM, address space mapping) |
| `lib/Compiler/CodeGen.cpp` | Simulator bytecode generation |
| `lib/Runtime/Simulator.cpp` | Two-phase simulator execution |
| `lib/Conversion/CMakeLists.txt` | Link `MLIRMemRefDialect` |
| `lib/Compiler/CMakeLists.txt` | Link `MLIRMemRefToLLVM` |
| `test/test_reduce_e2e.cpp` | Simulator E2E tests |
| `test/test_gpu_lowering.cpp` | GPU MLIR lowering tests |

## Gotchas we hit

1. **`populateGpuAllReducePatterns` silently fails** on some MLIR 18 builds (Colab apt packages). We bypassed it entirely by emitting the shuffle tree ourselves.

2. **Address space mapping must be registered** on the `LLVMTypeConverter` via `populateGpuMemorySpaceAttributeConversions`. Without it, shared memory pointers get LLVM addrspace 0 instead of 3, causing GPU crashes with no error message — just kernel death.

3. **`ast.AnnAssign` vs `ast.Assign`** — Python type annotations like `x: int = 5` produce `AnnAssign` nodes, not `Assign`. The `KernelVisitor` needed a `visit_AnnAssign` handler.

4. **`AllTypesMatch` trait** is required on the TableGen op definition so MLIR can infer the result type from the operand type without a custom builder.
