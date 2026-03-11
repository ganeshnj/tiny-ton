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
from tiny_ton.jit import KernelVisitor, _DTYPE_MAP

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


def run_pipeline(label, args):
    """Run the full pipeline for a given set of arguments."""
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    param_names = [a.arg for a in func_def.args.args]
    arg_is_pointer = [isinstance(a, np.ndarray) for a in args]
    arg_dtypes = []
    for a in args:
        if isinstance(a, np.ndarray) and a.dtype in _DTYPE_MAP:
            arg_dtypes.append(_DTYPE_MAP[a.dtype])
        else:
            arg_dtypes.append("i32")

    # -- MLIR --
    separator(f"STAGE 2: TinyTon MLIR ({label})")
    builder = core.IRBuilder()
    builder.begin_function(func_def.name)
    visitor = KernelVisitor(builder, param_names, arg_is_pointer, arg_dtypes)
    visitor.visit(func_def)
    print(builder.dump_mlir())

    # -- Simulator assembly --
    separator(f"STAGE 3: Simulator Assembly ({label})")
    result = builder.compile()
    assert result.success, result.error
    print(result.output)

    # -- PTX --
    separator(f"STAGE 4: PTX for {SM_VERSION} ({label})")
    builder2 = core.IRBuilder()
    builder2.begin_function(func_def.name)
    visitor2 = KernelVisitor(builder2, param_names, arg_is_pointer,
                             arg_dtypes)
    visitor2.visit(func_def)
    nvptx = builder2.compile_to_nvptx(sm_version=SM_VERSION)
    if nvptx.success:
        print(nvptx.ptx)
    else:
        print(f"  PTX compilation failed: {nvptx.error}")


# ── i32 pipeline ────────────────────────────────────────────

i32_args = (np.zeros(256, np.int32), np.zeros(256, np.int32),
            np.zeros(256, np.int32), 256)
run_pipeline("i32", i32_args)

# ── f32 pipeline ────────────────────────────────────────────

f32_args = (np.zeros(256, np.float32), np.zeros(256, np.float32),
            np.zeros(256, np.float32), 256)
run_pipeline("f32", f32_args)

# ── f16 pipeline ────────────────────────────────────────────

f16_args = (np.zeros(256, np.float16), np.zeros(256, np.float16),
            np.zeros(256, np.float16), 256)
run_pipeline("f16", f16_args)


# ── Execute on simulator (all types) ────────────────────────

separator("STAGE 5: Execute i32 on Simulator")

N = 256
a_i = np.arange(N, dtype=np.int32)
b_i = np.arange(N, dtype=np.int32)
c_i = np.zeros(N, dtype=np.int32)
vector_add[(N // 64,)](a_i, b_i, c_i, N)

print(f"  a[:8] = {a_i[:8]}")
print(f"  b[:8] = {b_i[:8]}")
print(f"  c[:8] = {c_i[:8]}")
assert np.array_equal(c_i, a_i + b_i), "i32 FAIL!"
print("  PASS")


separator("STAGE 6: Execute f32 on Simulator")

a_f = np.array([1.5, 2.5, 3.5, 0.1] * 64, dtype=np.float32)
b_f = np.array([0.5, 0.5, 0.5, 0.9] * 64, dtype=np.float32)
c_f = np.zeros(N, dtype=np.float32)
vector_add[(N // 64,)](a_f, b_f, c_f, N)

print(f"  a[:4] = {a_f[:4]}")
print(f"  b[:4] = {b_f[:4]}")
print(f"  c[:4] = {c_f[:4]}")
assert np.allclose(c_f, a_f + b_f, atol=1e-6), "f32 FAIL!"
print("  PASS")


separator("STAGE 7: Execute f16 on Simulator")

a_h = np.array([1.5, 2.5, 3.5, 0.1] * 64, dtype=np.float16)
b_h = np.array([0.5, 0.5, 0.5, 0.9] * 64, dtype=np.float16)
c_h = np.zeros(N, dtype=np.float16)
vector_add[(N // 64,)](a_h, b_h, c_h, N)

print(f"  a[:4] = {a_h[:4]}")
print(f"  b[:4] = {b_h[:4]}")
print(f"  c[:4] = {c_h[:4]}")
assert np.allclose(c_h, a_h + b_h, atol=1e-3), "f16 FAIL!"
print("  PASS")


# ── Summary ─────────────────────────────────────────────────

separator("Pipeline Summary")
print("""  Python source (same kernel code for i32, f32, and f16!)
       |  (ast.parse)
       v
  Python AST
       |  (KernelVisitor -- type inferred from numpy dtype)
       v
  TinyTon MLIR           <- i32 / f32 / f16 ops
       |
       +---> RegisterAlloc + CodeGen -> 16-bit binary -> Simulator
       |    (HADD/HMUL for f16, FADD/FMUL for f32, ADD/MUL for i32)
       |
       +---> TinyTonToGPU -> gpu.module + arith + llvm
                 |  addi/addf depending on type (f16/f32 both use addf)
                 v
            PTX assembly  -> CUDA driver API -> real GPU
""")
