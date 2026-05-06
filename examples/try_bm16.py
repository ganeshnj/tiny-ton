"""Quick correctness + timing test for BM=BN=BK=16."""
import sys, time
sys.path.insert(0, 'python')
sys.path.insert(0, 'build/bindings')
import numpy as np
import tiny_ton as tt


@tt.jit
def shmem_gemm(A_ptr, B_ptr, C_ptr, N, K,
               BM: tt.constexpr, BN: tt.constexpr, BK: tt.constexpr):
    bm = tt.program_id(0)
    bn = tt.program_id(1)
    tid = tt.arange(0, BK)
    for tm in range(BM):
        for tn in range(BN):
            a_row = A_ptr + (bm * BM + tm) * K
            b_col = B_ptr + (bn * BN + tn)
            c_out = C_ptr + (bm * BM + tm) * N + (bn * BN + tn)
            acc = 0.0
            for k0 in range(0, K, BK):
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


BM, BN, BK = 16, 16, 16

print(f"BM={BM} BN={BN} BK={BK}")
print(f"{'Shape':>16s}  {'grid':>10s}  {'blocks':>6s}  {'compile ms':>10s}  "
      f"{'run ms':>8s}  {'max_err':>10s}  result")
print("-" * 80)

for M, N, K in [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]:
    if M % BM or N % BN or K % BK:
        continue
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    grid = (M // BM, N // BN)
    blocks = grid[0] * grid[1]

    # first call includes compile time
    t0 = time.time()
    shmem_gemm[grid](A, B, C, N, K, BM, BN, BK)
    compile_ms = 1000 * (time.time() - t0)

    # second call is cached — pure run time
    C2 = np.zeros((M, N), dtype=np.float32)
    t0 = time.time()
    shmem_gemm[grid](A, B, C2, N, K, BM, BN, BK)
    run_ms = 1000 * (time.time() - t0)

    ref = A @ B
    err = float(np.max(np.abs(C - ref)))
    ok = np.allclose(C, ref, atol=1e-3)
    shape = f"{M}x{N}x{K}"
    print(f"{shape:>16s}  {str(grid):>10s}  {blocks:6d}  {compile_ms:10.0f}  "
          f"{run_ms:8.1f}  {err:10.2e}  {'PASS' if ok else 'FAIL'}")
