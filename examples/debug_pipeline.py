"""Step through the entire compilation pipeline stage by stage.

Run: PYTHONPATH="build/bindings:python" python3 examples/debug_pipeline.py
"""

import ast
import inspect
import os
import textwrap
import numpy as np
import _tiny_ton_core as core
import tiny_ton as tt
from tiny_ton.jit import KernelVisitor

SM_VERSION = os.environ.get("TTN_SM_VERSION", "sm_87")


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ── Stage 0: Python source ──────────────────────────────────

@tt.jit
def vector_add(a_ptr, b_ptr, c_ptr, N):
    pid = tt.program_id(0)
    offsets = pid * 64 + tt.arange(0, 64)
    mask = offsets < N
    a = tt.load(a_ptr + offsets, mask=mask)
    b = tt.load(b_ptr + offsets, mask=mask)
    tt.store(c_ptr + offsets, a + b, mask=mask)


separator("STAGE 0: Python Source")
source = textwrap.dedent(inspect.getsource(vector_add.fn))
print(source)


# ── Stage 1: Python AST ─────────────────────────────────────

separator("STAGE 1: Python AST")
tree = ast.parse(source)
print(ast.dump(tree, indent=2)[:1500], "...\n")


# ── Stage 2: TinyTon MLIR (our dialect) ──────────────────────

separator("STAGE 2: TinyTon MLIR (our dialect)")

args = (np.zeros(256, np.int32), np.zeros(256, np.int32),
        np.zeros(256, np.int32), 256)

func_def = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        func_def = node
        break

param_names = [a.arg for a in func_def.args.args]
arg_is_pointer = [isinstance(a, np.ndarray) for a in args]

builder = core.IRBuilder()
builder.begin_function(func_def.name)

visitor = KernelVisitor(builder, param_names, arg_is_pointer)
visitor.visit(func_def)

mlir_ir = builder.dump_mlir()
print(mlir_ir)


# ── Stage 3: Simulator assembly (16-bit ISA) ────────────────

separator("STAGE 3: Simulator Assembly (16-bit ISA)")
result = builder.compile()
assert result.success, result.error
print(result.output)


# ── Stage 4: Simulator binary (hex) ─────────────────────────

separator("STAGE 4: Simulator Binary (hex)")
binary = result.get_binary()
for i, inst in enumerate(binary[:10]):
    print(f"  [{i:2d}]  0x{inst:04X}")
if len(binary) > 10:
    print(f"  ... ({len(binary)} instructions total)")


# ── Stage 5: NVPTX / PTX ────────────────────────────────────

separator(f"STAGE 5: NVPTX (PTX assembly for {SM_VERSION})")

# Need a fresh builder since the module was consumed by compile()
builder2 = core.IRBuilder()
builder2.begin_function(func_def.name)
visitor2 = KernelVisitor(builder2, param_names, arg_is_pointer)
visitor2.visit(func_def)

nvptx = builder2.compile_to_nvptx(sm_version=SM_VERSION)
if nvptx.success:
    print(nvptx.ptx)
    print(f"  kernel_name: {nvptx.kernel_name}")
else:
    print(f"  PTX compilation failed: {nvptx.error}")


# ── Stage 6: Execute on simulator ───────────────────────────

separator("STAGE 6: Execute on Simulator")

N = 256
a = np.arange(N, dtype=np.int32)
b = np.arange(N, dtype=np.int32)
c = np.zeros(N, dtype=np.int32)

grid = (N // 64,)
vector_add[grid](a, b, c, N)

print(f"  a[:8] = {a[:8]}")
print(f"  b[:8] = {b[:8]}")
print(f"  c[:8] = {c[:8]}")
expected = a + b
if np.array_equal(c, expected):
    print("  PASS")
else:
    print("  FAIL!")
    print(f"  expected[:8] = {expected[:8]}")


# ── Summary ─────────────────────────────────────────────────

separator("Pipeline Summary")
print("""  Python source
       │  (ast.parse)
       ▼
  Python AST
       │  (KernelVisitor)
       ▼
  TinyTon MLIR           ← our custom dialect
       │
       ├──→ RegisterAlloc + CodeGen → 16-bit binary → Simulator
       │
       └──→ TinyTonToGPU → gpu.module + arith + llvm
                 │  (arith-to-llvm, gpu-to-nvvm, reconcile-casts)
                 ▼
            LLVM IR (nvptx64)
                 │  (TargetMachine)
                 ▼
            PTX assembly  → CUDA driver API → real GPU
""")
