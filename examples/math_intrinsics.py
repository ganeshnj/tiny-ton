"""Test math intrinsics: exp, log, sqrt, rsqrt, abs, max on f32 and f16.

Run: PYTHONPATH="build/bindings:python" python3 examples/math_intrinsics.py
"""

import numpy as np
import tiny_ton as tt


# ---------------------------------------------------------------------------
# Kernels -- one per intrinsic
# ---------------------------------------------------------------------------

@tt.jit
def kern_exp(src, dst, N):
    pid = tt.program_id(0)
    offsets = pid * 64 + tt.arange(0, 64)
    mask = offsets < N
    x = tt.load(src + offsets, mask=mask)
    tt.store(dst + offsets, tt.exp(x), mask=mask)


@tt.jit
def kern_log(src, dst, N):
    pid = tt.program_id(0)
    offsets = pid * 64 + tt.arange(0, 64)
    mask = offsets < N
    x = tt.load(src + offsets, mask=mask)
    tt.store(dst + offsets, tt.log(x), mask=mask)


@tt.jit
def kern_sqrt(src, dst, N):
    pid = tt.program_id(0)
    offsets = pid * 64 + tt.arange(0, 64)
    mask = offsets < N
    x = tt.load(src + offsets, mask=mask)
    tt.store(dst + offsets, tt.sqrt(x), mask=mask)


@tt.jit
def kern_rsqrt(src, dst, N):
    pid = tt.program_id(0)
    offsets = pid * 64 + tt.arange(0, 64)
    mask = offsets < N
    x = tt.load(src + offsets, mask=mask)
    tt.store(dst + offsets, tt.rsqrt(x), mask=mask)


@tt.jit
def kern_abs(src, dst, N):
    pid = tt.program_id(0)
    offsets = pid * 64 + tt.arange(0, 64)
    mask = offsets < N
    x = tt.load(src + offsets, mask=mask)
    tt.store(dst + offsets, tt.abs(x), mask=mask)


@tt.jit
def kern_max(a_ptr, b_ptr, dst, N):
    pid = tt.program_id(0)
    offsets = pid * 64 + tt.arange(0, 64)
    mask = offsets < N
    a = tt.load(a_ptr + offsets, mask=mask)
    b = tt.load(b_ptr + offsets, mask=mask)
    tt.store(dst + offsets, tt.max(a, b), mask=mask)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_unary(name, kernel, np_fn, inp, dtype, atol):
    N = len(inp)
    src = inp.copy()
    dst = np.zeros(N, dtype=dtype)
    grid = (N // 64,)
    kernel[grid](src, dst, N)
    expected = np_fn(inp)
    ok = np.allclose(dst, expected, atol=atol, rtol=1e-2)
    status = "PASS" if ok else "FAIL"
    if not ok:
        maxd = np.max(np.abs(dst.astype(np.float32) - expected.astype(np.float32)))
        print(f"  [{status}] {name} ({dtype.__name__})  max_diff={maxd:.6f}")
    else:
        print(f"  [{status}] {name} ({dtype.__name__})")
    return ok


def run_binary(name, kernel, np_fn, a, b, dtype, atol):
    N = len(a)
    dst = np.zeros(N, dtype=dtype)
    grid = (N // 64,)
    kernel[grid](a.copy(), b.copy(), dst, N)
    expected = np_fn(a, b)
    ok = np.allclose(dst, expected, atol=atol, rtol=1e-2)
    status = "PASS" if ok else "FAIL"
    if not ok:
        maxd = np.max(np.abs(dst.astype(np.float32) - expected.astype(np.float32)))
        print(f"  [{status}] {name} ({dtype.__name__})  max_diff={maxd:.6f}")
    else:
        print(f"  [{status}] {name} ({dtype.__name__})")
    return ok


def main():
    N = 256
    np.random.seed(42)

    all_pass = True

    for dtype, atol in [(np.float32, 1e-5), (np.float16, 5e-2)]:
        print(f"\n=== dtype={dtype.__name__} ===")

        pos = np.abs(np.random.randn(N).astype(dtype)) + 0.01
        mixed = (np.random.randn(N) * 2).astype(dtype)

        all_pass &= run_unary("exp",   kern_exp,   np.exp,  mixed, dtype, atol)
        all_pass &= run_unary("log",   kern_log,   np.log,  pos,   dtype, atol)
        all_pass &= run_unary("sqrt",  kern_sqrt,  np.sqrt, pos,   dtype, atol)

        rsqrt_fn = lambda x: np.float32(1.0) / np.sqrt(x.astype(np.float32))
        rsqrt_expected = rsqrt_fn(pos).astype(dtype)
        all_pass &= run_unary("rsqrt", kern_rsqrt,
                              lambda x: (np.float32(1.0) / np.sqrt(x.astype(np.float32))).astype(dtype),
                              pos, dtype, atol)

        all_pass &= run_unary("abs",   kern_abs,   np.abs,  mixed, dtype, atol)

        a = (np.random.randn(N) * 5).astype(dtype)
        b = (np.random.randn(N) * 5).astype(dtype)
        all_pass &= run_binary("max", kern_max, np.maximum, a, b, dtype, atol)

    print()
    if all_pass:
        print("All math intrinsic tests PASSED.")
    else:
        print("Some tests FAILED!")


if __name__ == "__main__":
    main()
