"""Fused softmax (1 kernel launch) -- verified against NumPy and 5-kernel version.

Run: PYTHONPATH="build/bindings:python" python3 examples/fused_softmax_test.py
"""

import numpy as np
import tiny_ton as tt


# --- fused softmax: 1 kernel launch -----------------------------------------

@tt.jit
def fused_softmax_kernel(src, dst, N):
    tid  = tt.arange(0, 64)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask, other=-float('inf'))
    mx   = tt.reduce_max(x)
    e    = tt.exp(x - mx)
    s    = tt.reduce_sum(e)
    tt.store(dst + tid, e / s, mask=mask)


def fused_softmax(x, out, N):
    """Single kernel launch. Requires N <= 64."""
    fused_softmax_kernel[(1,)](x, out, N)


# --- 5-kernel softmax (Stage 1 baseline) ------------------------------------

@tt.jit
def kern_reduce_max(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    mx = tt.reduce_max(x)
    tt.store(dst + pid, mx)


@tt.jit
def kern_sub_scalar(src, scalar_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    s = tt.load(scalar_ptr)
    tt.store(dst + off, x - s, mask=mask)


@tt.jit
def kern_exp(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    tt.store(dst + off, tt.exp(x), mask=mask)


@tt.jit
def kern_reduce_sum(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    total = tt.reduce_sum(x)
    tt.store(dst + pid, total)


@tt.jit
def kern_div_scalar(src, scalar_ptr, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    s = tt.load(scalar_ptr)
    tt.store(dst + off, x / s, mask=mask)


def five_kernel_softmax(x, out, N):
    """5-kernel reference: each step writes to global memory."""
    grid = (max(1, (N + 63) // 64),)
    tmp_max = np.zeros(1, dtype=x.dtype)
    tmp_exp = np.zeros(N, dtype=x.dtype)
    tmp_sum = np.zeros(1, dtype=x.dtype)
    kern_reduce_max[(1,)](x, tmp_max, N)
    kern_sub_scalar[grid](x, tmp_max, tmp_exp, N)
    kern_exp[grid](tmp_exp, tmp_exp, N)
    kern_reduce_sum[(1,)](tmp_exp, tmp_sum, N)
    kern_div_scalar[grid](tmp_exp, tmp_sum, out, N)


# --- NumPy reference ---------------------------------------------------------

def numpy_softmax(x):
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


# --- test --------------------------------------------------------------------

def main():
    np.random.seed(42)
    all_ok = True

    for N in [4, 16, 27]:
        x = np.random.randn(N).astype(np.float32)
        expected = numpy_softmax(x)

        # fused (1 launch)
        out_fused = np.zeros(N, dtype=np.float32)
        fused_softmax(x.copy(), out_fused, N)

        # 5-kernel reference
        out_5k = np.zeros(N, dtype=np.float32)
        five_kernel_softmax(x.copy(), out_5k, N)

        ok_numpy  = np.allclose(out_fused, expected, atol=1e-5)
        ok_5k     = np.allclose(out_fused, out_5k,  atol=1e-5)
        sums_to_1 = np.allclose(np.sum(out_fused), 1.0, atol=1e-5)
        in_range  = bool(np.all(out_fused >= 0.0) and np.all(out_fused <= 1.0))

        ok = ok_numpy and ok_5k and sums_to_1 and in_range
        tag = "PASS" if ok else "FAIL"
        print(f"fused_softmax N={N}: {tag}  sum={np.sum(out_fused):.6f}  "
              f"vs_numpy={ok_numpy}  vs_5k={ok_5k}")
        if not ok:
            for i in range(min(8, N)):
                print(f"  [{i}] fused={out_fused[i]:.6f}  "
                      f"5k={out_5k[i]:.6f}  numpy={expected[i]:.6f}")
            all_ok = False

    assert all_ok, "fused_softmax test failed"
    print("All fused_softmax tests passed.")
    print()
    print("Launch count comparison (per softmax call):")
    print("  5-kernel version: 5 launches, 9 DRAM round trips")
    print("  fused version:    1 launch,   2 DRAM round trips (1 load + 1 store)")


if __name__ == "__main__":
    main()
