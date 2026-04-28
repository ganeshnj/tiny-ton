"""Tiled GEMM kernels demonstrating compile-time for-loop unrolling.

Three kernels, each verified against NumPy:

  loop_sum_kernel     -- prove for-loop accumulation works (no matmul)
  tiled_dot_kernel    -- matvec C[row] = A[row,:] @ B[:] tiled over K
  tiled_matmul_kernel -- full GEMM C[M,N] = A[M,K] @ B[K,N] with nested loops

The JIT compiler unrolls ``for k in range(0, K, TILE_K)`` at compile time
because K and TILE_K are ``tt.constexpr`` parameters.  Each unrolled body
emits straight-line MLIR ops; the accumulator ``acc`` chains naturally across
iterations through SSA value bindings (no phi nodes needed).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "bindings"))

import numpy as np
import tiny_ton as tt


# ---------------------------------------------------------------------------
# 1. Loop sum — proves for-loop accumulation with masked loads
# ---------------------------------------------------------------------------

@tt.jit
def loop_sum_kernel(src_ptr, dst_ptr, N: tt.constexpr, TILE: tt.constexpr):
    """Sum N elements using a tiled for loop; store result in dst[0].

    Demonstrates: compile-time unrolling, masked loads, accumulation.
    Grid: (1,) — single block.
    """
    tid = tt.arange(0, TILE)
    acc = 0.0
    for t in range(0, N, TILE):
        mask = tid < (N - t)
        x = tt.load(src_ptr + t + tid, mask=mask, other=0.0)
        acc = acc + tt.reduce_sum(x)
    # Only thread 0 writes back.
    tt.store(dst_ptr + tid, acc, mask=tid < 1)


def test_loop_sum():
    print("--- loop_sum_kernel ---")
    for N, TILE in [(4, 4), (8, 4), (12, 4), (16, 8)]:
        src = np.arange(1, N + 1, dtype=np.float32)
        dst = np.zeros(TILE, dtype=np.float32)
        loop_sum_kernel[(1,)](src, dst, N, TILE)
        expected = float(src.sum())
        assert abs(dst[0] - expected) < 1e-4, (
            f"N={N} TILE={TILE}: got {dst[0]}, expected {expected}")
        print(f"  N={N:2d} TILE={TILE}: got {dst[0]:.1f}  expected {expected:.1f}  OK")


# ---------------------------------------------------------------------------
# 2. Tiled dot product / matvec — for-loop over K tiles
# ---------------------------------------------------------------------------

@tt.jit
def tiled_dot_kernel(A_ptr, B_ptr, C_ptr, K: tt.constexpr, TILE_K: tt.constexpr):
    """C[row] = dot(A[row, :], B[:]) — one block per row, tiled over K.

    Demonstrates: for-loop that accumulates a dot product across K tiles.
    Grid: (M,) — one block per output element.
    """
    row = tt.program_id(0)
    tid = tt.arange(0, TILE_K)
    acc = 0.0
    for k0 in range(0, K, TILE_K):
        a_tile = tt.load(A_ptr + row * K + k0 + tid)
        b_tile = tt.load(B_ptr + k0 + tid)
        acc = acc + tt.reduce_sum(a_tile * b_tile)
    tt.store(C_ptr + row, acc)


def test_tiled_dot():
    print("--- tiled_dot_kernel ---")
    for M, K, TILE_K in [(4, 8, 4), (8, 16, 4), (8, 16, 8)]:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K).astype(np.float32)
        C = np.zeros(M, dtype=np.float32)
        tiled_dot_kernel[(M,)](A, B, C, K, TILE_K)
        expected = A @ B
        max_err = float(np.abs(C - expected).max())
        assert max_err < 1e-4, (
            f"M={M} K={K} TILE_K={TILE_K}: max_err={max_err}")
        print(f"  M={M} K={K:2d} TILE_K={TILE_K}: max_err={max_err:.2e}  OK")


# ---------------------------------------------------------------------------
# 3. Full tiled GEMM — nested for-loops over output cols and K tiles
# ---------------------------------------------------------------------------

@tt.jit
def tiled_matmul_kernel(A_ptr, B_ptr, C_ptr,
                         K: tt.constexpr, N: tt.constexpr,
                         TILE_K: tt.constexpr):
    """C[row, :] = A[row, :] @ B — nested for-loops over cols and K tiles.

    A: [M, K] row-major
    B: [K, N] row-major
    C: [M, N] row-major

    Grid: (M,) — one block per output row.  The outer ``col`` loop and the
    inner ``k0`` loop are both compile-time unrolled (all bounds are
    tt.constexpr).  The accumulator ``acc`` chains across k0 iterations via
    SSA value bindings.
    """
    row = tt.program_id(0)
    tid = tt.arange(0, TILE_K)
    for col in range(N):
        acc = 0.0
        for k0 in range(0, K, TILE_K):
            # Load TILE_K elements of A row: A[row, k0 : k0+TILE_K]
            a_tile = tt.load(A_ptr + row * K + k0 + tid)
            # Load TILE_K elements of B column: B[k0:k0+TILE_K, col]
            # B stored row-major: B[k, col] = B_ptr[k*N + col]
            # B[k0+tid, col] = B_ptr[k0*N + col + tid*N]
            b_tile = tt.load(B_ptr + k0 * N + col + tid * N)
            acc = acc + tt.reduce_sum(a_tile * b_tile)
        tt.store(C_ptr + row * N + col, acc)


def test_tiled_matmul():
    print("--- tiled_matmul_kernel ---")
    for M, K, N, TILE_K in [(2, 4, 4, 4), (4, 8, 4, 4), (4, 8, 8, 4)]:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)
        tiled_matmul_kernel[(M,)](A, B, C, K, N, TILE_K)
        expected = A @ B
        max_err = float(np.abs(C - expected).max())
        assert max_err < 1e-4, (
            f"M={M} K={K} N={N} TILE_K={TILE_K}: max_err={max_err}")
        print(f"  M={M} K={K} N={N} TILE_K={TILE_K}: max_err={max_err:.2e}  OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_loop_sum()
    test_tiled_dot()
    test_tiled_matmul()
    print("\nAll tiled GEMM tests passed!")
