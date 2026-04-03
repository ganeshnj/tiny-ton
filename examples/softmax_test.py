"""Composed softmax (5 kernel launches) -- verified against NumPy.

Run: PYTHONPATH="build/bindings:python" python3 examples/softmax_test.py
"""

import numpy as np
import tiny_ton as tt


# --- kernel 1: reduce_max ---------------------------------------------------

@tt.jit
def kern_reduce_max(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    mx = tt.reduce_max(x)
    tt.store(dst + pid, mx)


# --- kernel 2: subtract scalar (broadcast) ----------------------------------

@tt.jit
def kern_sub_scalar(src, scalar_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    s = tt.load(scalar_ptr)
    tt.store(dst + off, x - s, mask=mask)


# --- kernel 3: exp ----------------------------------------------------------

@tt.jit
def kern_exp(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    tt.store(dst + off, tt.exp(x), mask=mask)


# --- kernel 4: reduce_sum ---------------------------------------------------

@tt.jit
def kern_reduce_sum(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    total = tt.reduce_sum(x)
    tt.store(dst + pid, total)


# --- kernel 5: divide by scalar (broadcast) ---------------------------------

@tt.jit
def kern_div_scalar(src, scalar_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    s = tt.load(scalar_ptr)
    tt.store(dst + off, x / s, mask=mask)


# --- orchestrator ------------------------------------------------------------

def softmax(x, out, N):
    """Host-side softmax: 5 kernel launches."""
    grid = (max(1, (N + 63) // 64),)
    tmp_max = np.zeros(1, dtype=x.dtype)
    tmp_exp = np.zeros(N, dtype=x.dtype)
    tmp_sum = np.zeros(1, dtype=x.dtype)

    kern_reduce_max[(1,)](x, tmp_max, N)
    kern_sub_scalar[grid](x, tmp_max, tmp_exp, N)
    kern_exp[grid](tmp_exp, tmp_exp, N)
    kern_reduce_sum[(1,)](tmp_exp, tmp_sum, N)
    kern_div_scalar[grid](tmp_exp, tmp_sum, out, N)


def numpy_softmax(x):
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def main():
    np.random.seed(42)
    all_ok = True

    for trial, N in enumerate([16, 32]):
        x = np.random.randn(N).astype(np.float32)
        expected = numpy_softmax(x)

        out = np.zeros(N, dtype=np.float32)
        softmax(x.copy(), out, N)

        ok = np.allclose(out, expected, atol=1e-5)
        sums_to_one = np.allclose(np.sum(out), 1.0, atol=1e-5)
        in_range = np.all(out >= 0.0) and np.all(out <= 1.0)

        tag = "PASS" if (ok and sums_to_one and in_range) else "FAIL"
        print(f"softmax N={N}: {tag}  sum={np.sum(out):.6f}")
        if not ok:
            for i in range(min(8, N)):
                print(f"  [{i}] got={out[i]:.6f}  expected={expected[i]:.6f}")
            all_ok = False
        if not sums_to_one:
            print(f"  sum != 1.0: {np.sum(out):.6f}")
            all_ok = False

    assert all_ok, "softmax test failed"
    print("All softmax tests passed.")


if __name__ == "__main__":
    main()
