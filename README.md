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

- [x] Fused softmax — 5 launches → 1 (warp shuffle: reduce max, sub, exp, reduce sum, div all in registers) — [`examples/fused_softmax_test.py`](examples/fused_softmax_test.py), [`docs/fused-softmax.md`](docs/fused-softmax.md)
- [x] Fused rmsnorm — 4 launches → 1 (warp shuffle: square, reduce sum, rsqrt, scale in registers) — [`examples/fused_rmsnorm_test.py`](examples/fused_rmsnorm_test.py), [`docs/fused-rmsnorm.md`](docs/fused-rmsnorm.md)
- [x] Fused per-head attention — 12 launches → 7 (score scaling + softmax fused into one kernel) — [`examples/fused_attention_test.py`](examples/fused_attention_test.py), [`docs/fused-attention.md`](docs/fused-attention.md)
- [x] NumPy training — replaced scalar `Value` autograd with vectorized NumPy forward + manual backward (full BPTT through KV cache) + Adam; 1000 steps in ~1s vs ~minutes

Expected: ~3x fewer kernel launches, ~3x speedup.

#### Layer 2 — Scale up model size (no code changes)

- [ ] Test at n_embd=64, n_embd=128 to find the GPU crossover point
  - n_embd=16: GPU 487x slower (4x useful work per launch)
  - n_embd=64: estimated ~30x slower
  - n_embd=512+: GPU wins

#### Layer 3 — Configurable block size (compiler change)

- [x] Make block size a kernel `constexpr` parameter — today it is hardcoded to 64, so 75% of threads are idle at n_embd=16 — [`examples/constexpr_test.py`](examples/constexpr_test.py), [`docs/constexpr.md`](docs/constexpr.md)
- [x] Implemented in `jit.py` (parse `PARAM: tt.constexpr` annotation, separate cache key per value, exclude from IR args); no C++ or MLIR changes required

#### Layer 4 — Tiled matmul with shared memory (C++ MLIR change)

- [ ] Tiled matmul with shared memory + barriers — reuse each weight row across multiple threads via shared memory; impactful at n_embd >= 64
- [ ] Requires adding `gpu.barrier` and shared memory ops to the MLIR pipeline

#### Layer 5 — Flash Attention (algorithmic, longer sequences)

- [ ] Flash Attention style — tiles the KV cache into chunks, accumulates softmax numerator/denominator across chunks; needed when seq_len > block_size (64)

#### Layer 6 — Automatic fusion pass (compiler infrastructure)

- [ ] Pattern-matching fusion pass on the `tinyton` MLIR dialect — detects `exp` → `reduce_sum` → `div` etc. and merges them automatically, like XLA/TVM/torch.compile

## License

MIT — see [LICENSE](LICENSE).
