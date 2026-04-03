"""Composed rmsnorm (4 kernel launches) -- verified against NumPy.

Run: PYTHONPATH="build/bindings:python" python3 examples/rmsnorm_test.py
"""

import numpy as np
import tiny_ton as tt


# --- kernel 1: square -------------------------------------------------------

@tt.jit
def kern_square(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    tt.store(dst + off, x * x, mask=mask)


# --- kernel 2: reduce_sum ---------------------------------------------------

@tt.jit
def kern_reduce_sum(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    total = tt.reduce_sum(x)
    tt.store(dst + pid, total)


# --- kernel 3: rsqrt of mean + epsilon (scalar -> scalar) -------------------

@tt.jit
def kern_rsqrt_mean(sum_ptr, n_ptr, out_ptr):
    tid = tt.arange(0, 64)
    s = tt.load(sum_ptr)
    n = tt.load(n_ptr)
    mean_eps = s / n + 1e-5
    scale = tt.rsqrt(mean_eps)
    tt.store(out_ptr, scale)


# --- kernel 4: multiply by scalar (broadcast) -------------------------------

@tt.jit
def kern_mul_scalar(src, scalar_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    s = tt.load(scalar_ptr)
    tt.store(dst + off, x * s, mask=mask)


# --- orchestrator ------------------------------------------------------------

def rmsnorm(x, out, N):
    """Host-side rmsnorm: 4 kernel launches."""
    grid = (max(1, (N + 63) // 64),)
    tmp_sq = np.zeros(N, dtype=x.dtype)
    tmp_sum = np.zeros(1, dtype=x.dtype)
    tmp_scl = np.zeros(1, dtype=x.dtype)
    n_arr = np.array([float(N)], dtype=x.dtype)

    kern_square[grid](x, tmp_sq, N)
    kern_reduce_sum[(1,)](tmp_sq, tmp_sum, N)
    kern_rsqrt_mean[(1,)](tmp_sum, n_arr, tmp_scl)
    kern_mul_scalar[grid](x, tmp_scl, out, N)


def numpy_rmsnorm(x):
    eps = 1e-5
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms


def main():
    np.random.seed(42)
    all_ok = True

    for N in [16, 32]:
        x = np.random.randn(N).astype(np.float32)
        expected = numpy_rmsnorm(x)

        out = np.zeros(N, dtype=np.float32)
        rmsnorm(x.copy(), out, N)

        ok = np.allclose(out, expected, atol=1e-5)
        tag = "PASS" if ok else "FAIL"
        print(f"rmsnorm N={N}: {tag}")
        if not ok:
            for i in range(min(8, N)):
                print(f"  [{i}] got={out[i]:.6f}  expected={expected[i]:.6f}")
            all_ok = False

    assert all_ok, "rmsnorm test failed"
    print("All rmsnorm tests passed.")


if __name__ == "__main__":
    main()
