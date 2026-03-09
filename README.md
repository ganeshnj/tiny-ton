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

## License

MIT — see [LICENSE](LICENSE).
