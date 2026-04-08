"""Fused rmsnorm (1 kernel launch) -- verified against NumPy and 4-kernel version.

Run: PYTHONPATH="build/bindings:python" python3 examples/fused_rmsnorm_test.py
"""

import numpy as np
import tiny_ton as tt


# --- fused rmsnorm: 1 kernel launch ------------------------------------------

@tt.jit
def fused_rmsnorm_kernel(src, dst, N, n_ptr):
    tid  = tt.arange(0, 64)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask)
    sq   = x * x
    s    = tt.reduce_sum(sq)
    n    = tt.load(n_ptr)
    scale = tt.rsqrt(s / n + 1e-5)
    tt.store(dst + tid, x * scale, mask=mask)


def fused_rmsnorm(x, out, N):
    """Single kernel launch. Requires N <= 64."""
    n_arr = np.array([float(N)], dtype=np.float32)
    fused_rmsnorm_kernel[(1,)](x, out, N, n_arr)


# --- 4-kernel rmsnorm (Stage 1 baseline) ------------------------------------

@tt.jit
def kern_square(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    tt.store(dst + off, x * x, mask=mask)


@tt.jit
def kern_reduce_sum(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    total = tt.reduce_sum(x)
    tt.store(dst + pid, total)


@tt.jit
def kern_rsqrt_mean(sum_ptr, n_ptr, out_ptr):
    tid = tt.arange(0, 64)
    s = tt.load(sum_ptr)
    n = tt.load(n_ptr)
    mean_eps = s / n + 1e-5
    scale = tt.rsqrt(mean_eps)
    tt.store(out_ptr, scale)


@tt.jit
def kern_mul_scalar(src, scalar_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    s = tt.load(scalar_ptr)
    tt.store(dst + off, x * s, mask=mask)


def four_kernel_rmsnorm(x, out, N):
    """4-kernel reference: each step writes to global memory."""
    grid = (max(1, (N + 63) // 64),)
    tmp_sq = np.zeros(N, dtype=x.dtype)
    tmp_sum = np.zeros(1, dtype=x.dtype)
    tmp_scl = np.zeros(1, dtype=x.dtype)
    n_arr = np.array([float(N)], dtype=x.dtype)

    kern_square[grid](x, tmp_sq, N)
    kern_reduce_sum[(1,)](tmp_sq, tmp_sum, N)
    kern_rsqrt_mean[(1,)](tmp_sum, n_arr, tmp_scl)
    kern_mul_scalar[grid](x, tmp_scl, out, N)


# --- NumPy reference ---------------------------------------------------------

def numpy_rmsnorm(x):
    eps = 1e-5
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms


# --- test --------------------------------------------------------------------

def main():
    np.random.seed(42)
    all_ok = True

    for N in [4, 16, 32]:
        x = np.random.randn(N).astype(np.float32)
        expected = numpy_rmsnorm(x)

        out_fused = np.zeros(N, dtype=np.float32)
        fused_rmsnorm(x.copy(), out_fused, N)

        out_4k = np.zeros(N, dtype=np.float32)
        four_kernel_rmsnorm(x.copy(), out_4k, N)

        ok_numpy = np.allclose(out_fused, expected, atol=1e-4)
        ok_4k    = np.allclose(out_fused, out_4k,  atol=1e-4)

        ok = ok_numpy and ok_4k
        tag = "PASS" if ok else "FAIL"
        print(f"fused_rmsnorm N={N}: {tag}  vs_numpy={ok_numpy}  vs_4k={ok_4k}")
        if not ok:
            for i in range(min(8, N)):
                print(f"  [{i}] fused={out_fused[i]:.6f}  "
                      f"4k={out_4k[i]:.6f}  numpy={expected[i]:.6f}")
            all_ok = False

    assert all_ok, "fused_rmsnorm test failed"
    print("All fused_rmsnorm tests passed.")
    print()
    print("Launch count comparison (per rmsnorm call):")
    print("  4-kernel version: 4 launches, 7 DRAM round trips")
    print("  fused version:    1 launch,   2 DRAM round trips (1 load + 1 store)")


if __name__ == "__main__":
    main()
