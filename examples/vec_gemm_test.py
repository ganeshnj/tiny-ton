"""Correctness test for the K4 Vec GEMM kernel.

K4 widens A's global load from scalar LDG.32 to vectorized LDG.128 (4 floats
per thread per k-step).  The numerical result must be identical to K3 — the
only difference is *how* the data reaches shared memory, not *what* data is
computed.
"""

import sys
import os

sys.path.insert(0, os.path.expanduser("~/tiny-ton/python"))
sys.path.insert(0, os.path.expanduser("~/tiny-ton/build/bindings"))

import numpy as np
import tiny_ton as tt


@tt.jit
def vec_gemm(A_ptr, B_ptr, C_ptr, N, K,
             BM: tt.constexpr, BN: tt.constexpr, BK: tt.constexpr):
    """K4 — vectorized (LDG.128) A-loads + XOR-swizzled shmem.

    Each thread loads 4 consecutive A-floats via tt.load4, then stores them
    into swizzled shmem slots.  B stays scalar (stride-N access).

    A is M×K, B is K×N, C is M×N.
    Grid  : (M // BM, N // BN)
    Block : BK threads
    Shmem : 8×BK floats  (4×BK for A vec4 + 4×BK for B scalar)
    K-loop step: 4×BK
    """
    bm  = tt.program_id(0)
    bn  = tt.program_id(1)
    tid = tt.arange(0, BK)

    for tm in range(BM):
        for tn in range(BN):
            a_row = A_ptr + (bm * BM + tm) * K
            b_col = B_ptr + (bn * BN + tn)
            c_out = C_ptr + (bm * BM + tm) * N + (bn * BN + tn)

            acc = 0.0
            for k0 in range(0, K, 4 * BK):
                a0, a1, a2, a3 = tt.load4(a_row + k0 + tid * 4)

                sw0 = (tid * 4)     ^ (((tid * 4)     >> 3) & 7)
                sw1 = (tid * 4 + 1) ^ (((tid * 4 + 1) >> 3) & 7)
                sw2 = (tid * 4 + 2) ^ (((tid * 4 + 2) >> 3) & 7)
                sw3 = (tid * 4 + 3) ^ (((tid * 4 + 3) >> 3) & 7)
                tt.shared_store(sw0, a0, buffer_size=8*BK)
                tt.shared_store(sw1, a1, buffer_size=8*BK)
                tt.shared_store(sw2, a2, buffer_size=8*BK)
                tt.shared_store(sw3, a3, buffer_size=8*BK)

                for j in range(4):
                    b_val = tt.load(b_col + (k0 + j * BK + tid) * N)
                    bidx = j * BK + tid
                    bsw = bidx ^ ((bidx >> 3) & 7)
                    tt.shared_store(4 * BK + bsw, b_val, buffer_size=8*BK)

                tt.sync()

                for j in range(4):
                    aidx = j * BK + tid
                    asw = aidx ^ ((aidx >> 3) & 7)
                    a_sh = tt.shared_load(asw, buffer_size=8*BK)
                    bidx = j * BK + tid
                    bsw = bidx ^ ((bidx >> 3) & 7)
                    b_sh = tt.shared_load(4 * BK + bsw, buffer_size=8*BK)
                    acc = acc + tt.reduce_sum(a_sh * b_sh)
                tt.sync()
            tt.store(c_out, acc)


def run_test(M, N, K, BM=1, BN=1, BK=16):
    assert K % (4 * BK) == 0, f"K={K} must be divisible by 4*BK={4*BK}"
    assert M % BM == 0, f"M={M} must be divisible by BM={BM}"
    assert N % BN == 0, f"N={N} must be divisible by BN={BN}"

    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    grid = (M // BM, N // BN)
    vec_gemm[grid](A, B, C, N, K, BM, BN, BK)

    ref = A @ B
    max_err = float(np.max(np.abs(C - ref)))
    ok = max_err < 1e-2
    status = "PASS" if ok else "FAIL"
    print(f"  M={M:4d} N={N:4d} K={K:4d}  BM={BM} BN={BN} BK={BK:2d}  "
          f"max_err={max_err:.2e}  {status}")
    return ok


if __name__ == "__main__":
    print("K4 Vec GEMM correctness tests")
    print("=" * 60)

    all_pass = True

    print("BM=BN=1, BK=16  (K must be divisible by 64):")
    for M, N, K in [(64, 64, 64), (64, 64, 128), (128, 128, 128)]:
        all_pass &= run_test(M, N, K, BM=1, BN=1, BK=16)

    print("\nBM=BN=2, BK=16:")
    for M, N, K in [(64, 64, 64), (128, 128, 128)]:
        all_pass &= run_test(M, N, K, BM=2, BN=2, BK=16)

    print("\nBM=BN=4, BK=16:")
    for M, N, K in [(64, 64, 64), (128, 128, 128), (256, 256, 256)]:
        all_pass &= run_test(M, N, K, BM=4, BN=4, BK=16)

    print()
    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)
