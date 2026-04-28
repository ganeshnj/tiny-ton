# tiny-ton

A Triton-inspired GPU kernel compiler. Write GPU kernels in Python, compile them via MLIR to real hardware instructions.

```python
import tiny_ton as tt
import numpy as np

@tt.jit
def vector_add(a_ptr, b_ptr, c_ptr, N):
    pid = tt.program_id(0)
    offsets = pid * 64 + tt.arange(0, 64)
    mask = offsets < N
    a = tt.load(a_ptr + offsets, mask=mask)
    b = tt.load(b_ptr + offsets, mask=mask)
    tt.store(c_ptr + offsets, a + b, mask=mask)

a = np.array([1, 2, 3, 4], dtype=np.int32)
b = np.array([10, 20, 30, 40], dtype=np.int32)
c = np.zeros(4, dtype=np.int32)

vector_add[(1,)](a, b, c, len(a))
print(c)  # [11, 22, 33, 44]
```

## Architecture

```
Python (@jit) → AST capture → pybind11 → C++ IRBuilder → MLIR (TinyTon dialect)
    → Register Allocation → CodeGen → Runtime/Simulator → Execution
```

## Building

### Prerequisites

- CMake 3.20+
- C++17 compiler
- LLVM/MLIR 18
- pybind11
- Python 3.10+

### Build

```bash
# Docker (recommended)
docker build -t tiny-ton .
docker run tiny-ton ttc --emit asm examples/vector_add.tgc

# Native
brew install cmake ninja llvm@18
rm -rf build
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@18/lib/cmake/mlir \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@18/lib/cmake/llvm \
  -DTTN_ENABLE_PYTHON=OFF
cmake --build build
./build/bin/ttc --help
```

### Python package

```bash
cd python
pip install -e .
```

## Roadmap — microgpt on GPU

Goal: run [Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) forward pass on GPU via tiny-ton JIT kernels.

### Done

- [x] Element-wise arithmetic: `add`, `sub`, `mul`, `div` (i32/f32/f16)
- [x] Math intrinsics: `exp`, `log`, `sqrt`, `rsqrt`, `abs`, `max` (f32/f16)
- [x] Masked load/store with `program_id` threading
- [x] NVIDIA GPU backend: MLIR → PTX via combined pass + libdevice
- [x] Google Colab CI: build + test on T4 GPU

### Stage 1 — Standalone GPU kernels (one op at a time)

Each operation is a single kernel, tested independently against NumPy.

- [x] `tt.reduce_sum` — warp-shuffle / `gpu.all_reduce` reduction
- [x] `tt.reduce_max` — same as above with max
- [x] `tt.relu` — element-wise `max(x, 0)`
- [x] `tt.gather` — embedding lookup by index
- [x] `tt.dot` / matvec — dot product via `reduce_sum`
- [x] `softmax` — composed: `reduce_max` → `sub` → `exp` → `reduce_sum` → `div` (5 launches)
- [x] `rmsnorm` — composed: `square` → `reduce_sum` → `rsqrt` → `scale` (4 launches)
- [x] `linear` — matvec using dot (one output per block)
- [x] `cross_entropy` — composed: `softmax` → `gather` → `-log`
- [x] `attention` — composed: linear projections + dot + softmax + weighted sum

### Stage 2 — Wire into microgpt

Replace microgpt's Python ops one by one with tiny-ton GPU kernels.
Each op is still a separate launch — no fusion yet.

- [x] Replace `softmax()`, `rmsnorm()`, `linear()` with GPU kernels
- [x] Replace attention + MLP with composed GPU launches
- [x] Full forward pass end-to-end on GPU
- [x] Benchmark vs Python CPU baseline

### Stage 3 — Optimize for GPU

Reduce launch overhead, fuse kernels, improve throughput.

**Benchmark context:** Stage 2 ran 8,800 kernel launches for 20 inference samples at n_embd=16. Overhead dominated (~150µs/launch × 8,800 = ~1,320s). GPU ran 487x slower than CPU. Every item below attacks this.

#### Layer 1 — Reduce launch count (pure Python kernel work)

- [x] Fused softmax — 5 launches → 1 (warp shuffle: reduce max, sub, exp, reduce sum, div all in registers) — [`examples/fused_softmax_test.py`](examples/fused_softmax_test.py), [`docs/10-fused-softmax.md`](docs/10-fused-softmax.md)
- [x] Fused rmsnorm — 4 launches → 1 (warp shuffle: square, reduce sum, rsqrt, scale in registers) — [`examples/fused_rmsnorm_test.py`](examples/fused_rmsnorm_test.py), [`docs/12-fused-rmsnorm.md`](docs/12-fused-rmsnorm.md)
- [x] Fused per-head attention — 12 launches → 7 (score scaling + softmax fused into one kernel) — [`examples/fused_attention_test.py`](examples/fused_attention_test.py), [`docs/13-fused-attention.md`](docs/13-fused-attention.md)
- [x] NumPy training — replaced scalar `Value` autograd with vectorized NumPy forward + manual backward (full BPTT through KV cache) + Adam; 1000 steps in ~1s vs ~minutes

Expected: ~3x fewer kernel launches, ~3x speedup.

#### Layer 2 — Scale up model size (no code changes)

- [ ] Test at n_embd=64, n_embd=128 to find the GPU crossover point
  - n_embd=16: GPU 487x slower (4x useful work per launch)
  - n_embd=64: estimated ~30x slower
  - n_embd=512+: GPU wins

#### Layer 3 — Configurable block size (compiler change)

- [x] Make block size a kernel `constexpr` parameter — today it is hardcoded to 64, so 75% of threads are idle at n_embd=16 — [`examples/constexpr_test.py`](examples/constexpr_test.py), [`docs/14-constexpr.md`](docs/14-constexpr.md)
- [x] Implemented in `jit.py` (parse `PARAM: tt.constexpr` annotation, separate cache key per value, exclude from IR args); no C++ or MLIR changes required

#### Layer 4 — Ampere GEMM: Road to cuBLAS (Jetson Orin Nano, sm_87)

Mirrors [Modular's Blackwell series](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction), adapted for Ampere sm_87. Each kernel adds one hardware concept. Target: match cuBLAS FP32 (~2 TFLOPS), then FP16 with tensor cores (~12 TFLOPS).

**Hardware context (Jetson Orin Nano):** 16 SMs · 48 KB shmem/SM · 68 GB/s · FP32 peak ~2 TFLOPS · FP16 tensor core peak ~12 TFLOPS

| Kernel | Technique | Expected TFLOPS | Compiler change |
|--------|-----------|-----------------|-----------------|
| K0: Naive GEMM | One block per output element, global memory only | ~0.001 | Add `//`, `%` to JIT |
| K1: Row GEMM | One block per row, A reused across N cols | ~0.005 | None (rename tiled_matmul_kernel) |
| K2: Shmem GEMM | A + B tiles in shared memory, 2D grid | ~0.1 | `program_id(1)` + `scf.for` runtime loop |
| K3: Swizzled GEMM | XOR-swizzle shmem layout, eliminate 8-way bank conflicts | ~0.2 | Swizzle address helper in JIT |
| K4: Vectorized GEMM | `LDG.128` — load 4 floats per instruction | ~0.5 | New `tt.load_v4` IR op |
| K5: Pipelined GEMM | `cp.async` — overlap load with compute (Ampere) | ~1.0 | New `tt.async_copy` IR op |
| K6: Tensor Core GEMM | `mma.sync.m16n8k16` — FP16 tensor cores | ~6–12 | New `tt.dot` tile op |

**Progress:**

- [x] Correctness: `tiled_gemm_test.py` — all loop_sum, tiled_dot, tiled_matmul tests pass on Jetson
- [x] Bug fix: `reduce_sum` partial-warp shuffle (`blockSize < 32`) — passes correct `width` to `gpu::ShuffleOp` instead of hardcoded 32
- [x] Bug fix: `emit_mul` scalar promotion — `_promote_scalar` in `jit.py` prevents `TypeError` when a constexpr int is passed as an IR operand
- [x] Benchmark notebook — `examples/gemm_benchmark.ipynb` with cuBLAS reference numbers and gap analysis
- [ ] K0: Naive GEMM — add `//` (FloorDiv) and `%` (Mod) to JIT `BinOp` handler
- [ ] K1: Row GEMM — rename/reframe `tiled_matmul_kernel` in notebook
- [ ] K2: Shmem GEMM — `program_id(1)` (2D grid) + `scf.for` runtime K-loop
- [ ] K3: Swizzled GEMM — 128-byte XOR swizzle to eliminate shmem bank conflicts
- [ ] K4: Vectorized GEMM — `LDG.128` vectorized loads
- [ ] K5: Pipelined GEMM — `cp.async` to overlap load and compute
- [ ] K6: Tensor Core GEMM — `mma.sync.aligned.m16n8k16` via `tt.dot`

See [`examples/gemm_benchmark.ipynb`](examples/gemm_benchmark.ipynb) for the live benchmark notebook and [`docs/16-tiled-gemm.md`](docs/16-tiled-gemm.md) for the current tiling design.

#### Layer 5 — Flash Attention (algorithmic, longer sequences)

- [ ] Flash Attention style — tiles the KV cache into chunks, accumulates softmax numerator/denominator across chunks; needed when seq_len > block_size (64)

#### Layer 6 — Automatic fusion pass (compiler infrastructure)

- [ ] Pattern-matching fusion pass on the `tinyton` MLIR dialect — detects `exp` → `reduce_sum` → `div` etc. and merges them automatically, like XLA/TVM/torch.compile

## License

MIT — see [LICENSE](LICENSE).
