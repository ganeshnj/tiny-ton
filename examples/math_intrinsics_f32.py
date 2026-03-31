"""Math intrinsics on f32 -- exp, log, sqrt, rsqrt, abs, max.

Shows progressive lowering: Python → MLIR → simulator assembly → execution.

Run: PYTHONPATH="build/bindings:python" python3 examples/math_intrinsics_f32.py
"""

import numpy as np
import tiny_ton as tt


@tt.jit
def kern_exp(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    tt.store(dst + off, tt.exp(tt.load(src + off, mask=mask)), mask=mask)


@tt.jit
def kern_log(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    tt.store(dst + off, tt.log(tt.load(src + off, mask=mask)), mask=mask)


@tt.jit
def kern_sqrt(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    tt.store(dst + off, tt.sqrt(tt.load(src + off, mask=mask)), mask=mask)


@tt.jit
def kern_rsqrt(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    tt.store(dst + off, tt.rsqrt(tt.load(src + off, mask=mask)), mask=mask)


@tt.jit
def kern_abs(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    tt.store(dst + off, tt.abs(tt.load(src + off, mask=mask)), mask=mask)


@tt.jit
def kern_max(a_ptr, b_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    a = tt.load(a_ptr + off, mask=mask)
    b = tt.load(b_ptr + off, mask=mask)
    tt.store(dst + off, tt.max(a, b), mask=mask)


def test_unary(name, kernel, np_fn, inp):
    N = len(inp)
    dst = np.zeros(N, dtype=np.float32)
    kernel[(N // 64,)](inp.copy(), dst, N)
    expected = np_fn(inp)
    ok = np.allclose(dst, expected, atol=1e-5, rtol=1e-2)
    print(f"  {name}: dst[:4]={dst[:4]}  expected[:4]={expected[:4]}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    N = 256
    np.random.seed(42)
    pos = np.abs(np.random.randn(N).astype(np.float32)) + 0.01
    mixed = (np.random.randn(N) * 2).astype(np.float32)

    all_ok = True
    all_ok &= test_unary("exp",   kern_exp,   np.exp,  mixed)
    all_ok &= test_unary("log",   kern_log,   np.log,  pos)
    all_ok &= test_unary("sqrt",  kern_sqrt,  np.sqrt, pos)
    all_ok &= test_unary("rsqrt", kern_rsqrt,
                         lambda x: (1.0 / np.sqrt(x)).astype(np.float32), pos)
    all_ok &= test_unary("abs",   kern_abs,   np.abs,  mixed)

    a = (np.random.randn(N) * 5).astype(np.float32)
    b = (np.random.randn(N) * 5).astype(np.float32)
    dst = np.zeros(N, dtype=np.float32)
    kern_max[(N // 64,)](a.copy(), b.copy(), dst, N)
    expected = np.maximum(a, b)
    ok = np.allclose(dst, expected, atol=1e-5)
    print(f"  max:   dst[:4]={dst[:4]}  expected[:4]={expected[:4]}  {'PASS' if ok else 'FAIL'}")
    all_ok &= ok

    print()
    print("All PASSED." if all_ok else "Some FAILED!")


if __name__ == "__main__":
    main()
