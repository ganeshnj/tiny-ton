"""Correctness test for the K2 Shmem GEMM kernel."""

import sys
import os

sys.path.insert(0, os.path.expanduser("~/tiny-ton/python"))
sys.path.insert(0, os.path.expanduser("~/tiny-ton/build/bindings"))

import numpy as np
import tiny_ton as tt


@tt.jit
def shmem_gemm(A_ptr, B_ptr, C_ptr, N, K,
               BM: tt.constexpr, BN: tt.constexpr, BK: tt.constexpr):
    """K2 — 2D grid + runtime scf.for + shared memory tiling.

    A is M×K, B is K×N, C is M×N.
    Grid  : (M // BM, N // BN)
    Block : BK threads
    Each block owns the BM×BN output sub-tile of C.
    """
    bm = tt.program_id(0)
    bn = tt.program_id(1)
    tid = tt.arange(0, BK)

    for tm in range(BM):        # constexpr — unrolled
        for tn in range(BN):    # constexpr — unrolled
            a_row = A_ptr + (bm * BM + tm) * K
            b_col = B_ptr + (bn * BN + tn)
            c_out = C_ptr + (bm * BM + tm) * N + (bn * BN + tn)

            acc = 0.0
            for k0 in range(0, K, BK):   # runtime scf.for
                a_val = tt.load(a_row + k0 + tid)
                tt.shared_store(tid, a_val, buffer_size=2*BK)
                b_val = tt.load(b_col + (k0 + tid) * N)
                tt.shared_store(BK + tid, b_val, buffer_size=2*BK)
                tt.sync()
                a_sh = tt.shared_load(tid,      buffer_size=2*BK)
                b_sh = tt.shared_load(BK + tid, buffer_size=2*BK)
                acc  = acc + tt.reduce_sum(a_sh * b_sh)
                tt.sync()
            tt.store(c_out, acc)


def run_test(M, N, K, BM=1, BN=1, BK=16):
    assert K % BK == 0, f"K={K} must be divisible by BK={BK}"
    assert M % BM == 0, f"M={M} must be divisible by BM={BM}"
    assert N % BN == 0, f"N={N} must be divisible by BN={BN}"

    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    grid = (M // BM, N // BN)
    shmem_gemm[grid](A, B, C, N, K, BM, BN, BK)

    ref = A @ B
    max_err = float(np.max(np.abs(C - ref)))
    ok = max_err < 1e-2
    status = "PASS" if ok else "FAIL"
    print(f"  M={M:4d} N={N:4d} K={K:4d}  BM={BM} BN={BN} BK={BK:2d}  "
          f"max_err={max_err:.2e}  {status}")
    return ok


if __name__ == "__main__":
    print("K2 Shmem GEMM correctness tests")
    print("=" * 60)

    all_pass = True

    print("BM=BN=1 (one output element per block):")
    for M, N, K in [(16, 16, 16), (32, 32, 32), (64, 64, 64), (64, 64, 128)]:
        all_pass &= run_test(M, N, K, BM=1, BN=1, BK=16)

    print("\nBM=BN=2 (2×2 output tile per block):")
    for M, N, K in [(16, 16, 16), (32, 32, 32), (64, 64, 64)]:
        all_pass &= run_test(M, N, K, BM=2, BN=2, BK=16)

    print("\nBM=BN=4 (4×4 output tile per block, 16 unrolled K-loops):")
    for M, N, K in [(32, 32, 32), (64, 64, 64), (128, 128, 128)]:
        all_pass &= run_test(M, N, K, BM=4, BN=4, BK=16)

    print()
    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)
