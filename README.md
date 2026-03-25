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

- [ ] `tt.reduce_sum` — warp-shuffle / `gpu.all_reduce` reduction
- [ ] `tt.reduce_max` — same as above with max
- [ ] `tt.relu` — element-wise `max(x, 0)`
- [ ] `tt.gather` — embedding lookup by index
- [ ] `tt.dot` / matvec — dot product via `reduce_sum`
- [ ] `softmax` — composed: `reduce_max` → `sub` → `exp` → `reduce_sum` → `div` (5 launches)
- [ ] `rmsnorm` — composed: `square` → `reduce_sum` → `rsqrt` → `scale` (4 launches)
- [ ] `linear` — matvec using dot (one output per block)
- [ ] `cross_entropy` — composed: `softmax` → `gather` → `-log`
- [ ] `attention` — composed: linear projections + dot + softmax + weighted sum

### Stage 2 — Wire into microgpt

Replace microgpt's Python ops one by one with tiny-ton GPU kernels.
Each op is still a separate launch — no fusion yet.

- [ ] Replace `softmax()`, `rmsnorm()`, `linear()` with GPU kernels
- [ ] Replace attention + MLP with composed GPU launches
- [ ] Full forward pass end-to-end on GPU
- [ ] Benchmark vs Python CPU baseline

### Stage 3 — Optimize for GPU

Reduce launch overhead, fuse kernels, improve throughput.

- [ ] Fused softmax (single kernel)
- [ ] Fused rmsnorm (single kernel)
- [ ] Tiled matmul with shared memory + barriers
- [ ] Fused attention (Flash-attention style)
- [ ] Fused MLP + transformer block
- [ ] Automatic operator fusion pass in the compiler

## License

MIT — see [LICENSE](LICENSE).
