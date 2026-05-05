"""Correctness test for the K2 Shmem GEMM kernel."""

import sys
import os

sys.path.insert(0, os.path.expanduser("~/tiny-ton/python"))
sys.path.insert(0, os.path.expanduser("~/tiny-ton/build/bindings"))

import numpy as np
import tiny_ton as tt


@tt.jit
def shmem_gemm(A_ptr, B_ptr, C_ptr, M, N, K,
               TM: tt.constexpr, TN: tt.constexpr, TK: tt.constexpr):
    """K2 — 2D grid + runtime scf.for + shared memory tiling.

    Grid  : (M // TM, N // TN)   each block computes one C[bm*TM, bn*TN]
    Block : TK threads
    """
    bm  = tt.program_id(0)   # row-tile index
    bn  = tt.program_id(1)   # col-tile index
    tid = tt.arange(0, TK)   # thread index within block [0, TK)

    acc = 0.0

    for k0 in range(0, K, TK):           # runtime scf.for — no IR explosion
        # Load A[bm*TM, k0..k0+TK-1] into shared memory
        a_val = tt.load(A_ptr + bm * TM * K + k0 + tid)
        tt.shared_store(tid, a_val, buffer_size=2*TK)

        # Load B[k0..k0+TK-1, bn*TN] into shared memory
        b_val = tt.load(B_ptr + (k0 + tid) * N + bn * TN)
        tt.shared_store(TK + tid, b_val, buffer_size=2*TK)

        tt.sync()

        # Accumulate dot product from shared memory
        a_sh = tt.shared_load(tid,      buffer_size=2*TK)
        b_sh = tt.shared_load(TK + tid, buffer_size=2*TK)
        acc  = acc + tt.reduce_sum(a_sh * b_sh)

        tt.sync()

    tt.store(C_ptr + bm * TM * N + bn * TN, acc)


def run_test(M, N, K, TK=16):
    TM, TN = 1, 1
    assert K % TK == 0, f"K={K} must be divisible by TK={TK}"

    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    grid = (M // TM, N // TN)
    shmem_gemm[grid](A, B, C, M, N, K, TM, TN, TK)

    ref = A @ B
    max_err = float(np.max(np.abs(C - ref)))
    ok = max_err < 1e-2
    status = "PASS" if ok else "FAIL"
    print(f"  M={M:4d} N={N:4d} K={K:4d} TK={TK:2d}  max_err={max_err:.2e}  {status}")
    return ok


if __name__ == "__main__":
    print("K2 Shmem GEMM correctness tests")
    print("=" * 50)

    all_pass = True
    for M, N, K in [(16, 16, 16), (32, 32, 32), (64, 64, 64), (64, 64, 128)]:
        all_pass &= run_test(M, N, K, TK=16)

    print()
    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)
