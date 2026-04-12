"""Test shared memory ops: tt.sync, tt.shared_store, tt.shared_load.

Verifies:
  - Basic shared memory round-trip: store then load
  - Tiled 2-row-per-block matvec using shared memory for x vector reuse
  - Correctness against NumPy for various matrix sizes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import tiny_ton as tt


def compare(name, got, expected, atol=1e-4):
    ok = np.allclose(got, expected, atol=atol)
    print(f'  {name}: {"PASS" if ok else "FAIL"}')
    if not ok:
        print(f'    got:      {got}')
        print(f'    expected: {expected}')
    return ok


# ---------------------------------------------------------------------------
# Kernel 1: shared memory round-trip (store → sync → load)
# ---------------------------------------------------------------------------

@tt.jit
def shmem_roundtrip(src, dst, N, BLOCK: tt.constexpr):
    tid  = tt.arange(0, BLOCK)
    mask = tid < N
    val  = tt.load(src + tid, mask=mask)
    tt.shared_store(tid, val)
    tt.sync()
    out  = tt.shared_load(tid)
    tt.store(dst + tid, out, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: single-row linear (baseline, no shared memory)
# ---------------------------------------------------------------------------

@tt.jit
def linear_kernel(W_ptr, x_ptr, y_ptr, in_features, BLOCK: tt.constexpr):
    pid = tt.program_id(0)
    tid = tt.arange(0, BLOCK)
    mask = tid < in_features
    w = tt.load(W_ptr + pid * in_features + tid, mask=mask)
    x = tt.load(x_ptr + tid, mask=mask)
    dot = tt.reduce_sum(w * x)
    tt.store(y_ptr + pid, dot)


# ---------------------------------------------------------------------------
# Kernel 3: tiled 2-row-per-block linear (shared memory for x reuse)
# ---------------------------------------------------------------------------

@tt.jit
def tiled_linear_kernel(W_ptr, x_ptr, y_ptr, in_features, BLOCK: tt.constexpr):
    pid  = tt.program_id(0)
    tid  = tt.arange(0, BLOCK)
    mask = tid < in_features

    x_val = tt.load(x_ptr + tid, mask=mask)
    tt.shared_store(tid, x_val)
    tt.sync()
    x_sh = tt.shared_load(tid)

    w0   = tt.load(W_ptr + (pid * 2)     * in_features + tid, mask=mask)
    w1   = tt.load(W_ptr + (pid * 2 + 1) * in_features + tid, mask=mask)
    dot0 = tt.reduce_sum(w0 * x_sh)
    dot1 = tt.reduce_sum(w1 * x_sh)
    tt.store(y_ptr + pid * 2,     dot0)
    tt.store(y_ptr + pid * 2 + 1, dot1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shmem_roundtrip():
    print('--- shared memory round-trip ---')
    all_ok = True
    for N in [4, 16, 27]:
        x = np.random.randn(N).astype(np.float32)
        out = np.zeros(N, dtype=np.float32)
        shmem_roundtrip[(1,)](x.copy(), out, N, N)
        ok = compare(f'N={N}', out, x)
        all_ok = all_ok and ok
    return all_ok


def test_single_row_linear():
    print('--- single-row linear (baseline) ---')
    all_ok = True
    for out_features, in_features in [(4, 4), (8, 16), (6, 27)]:
        W = np.random.randn(out_features, in_features).astype(np.float32)
        x = np.random.randn(in_features).astype(np.float32)
        expected = W @ x
        y = np.zeros(out_features, dtype=np.float32)
        BLOCK = max(in_features, 4)
        linear_kernel[(out_features,)](
            W.flatten().copy(), x.copy(), y, in_features, BLOCK)
        ok = compare(f'{out_features}x{in_features}', y, expected)
        all_ok = all_ok and ok
    return all_ok


def test_tiled_linear():
    print('--- tiled 2-row-per-block linear (shared memory) ---')
    all_ok = True
    for out_features, in_features in [(4, 4), (8, 16), (6, 27)]:
        W = np.random.randn(out_features, in_features).astype(np.float32)
        x = np.random.randn(in_features).astype(np.float32)
        expected = W @ x
        y = np.zeros(out_features, dtype=np.float32)
        BLOCK = max(in_features, 4)
        n_blocks = out_features // 2
        tiled_linear_kernel[(n_blocks,)](
            W.flatten().copy(), x.copy(), y, in_features, BLOCK)
        ok = compare(f'{out_features}x{in_features} tiled', y, expected)
        all_ok = all_ok and ok
    return all_ok


def test_tiled_vs_baseline():
    print('--- tiled vs baseline match ---')
    W = np.random.randn(8, 16).astype(np.float32)
    x = np.random.randn(16).astype(np.float32)

    y_baseline = np.zeros(8, dtype=np.float32)
    linear_kernel[(8,)](W.flatten().copy(), x.copy(), y_baseline, 16, 16)

    y_tiled = np.zeros(8, dtype=np.float32)
    tiled_linear_kernel[(4,)](W.flatten().copy(), x.copy(), y_tiled, 16, 16)

    return compare('8x16 baseline==tiled', y_tiled, y_baseline)


if __name__ == '__main__':
    np.random.seed(42)
    results = [
        test_shmem_roundtrip(),
        test_single_row_linear(),
        test_tiled_linear(),
        test_tiled_vs_baseline(),
    ]
    print()
    if all(results):
        print('All tests PASSED')
    else:
        print('SOME TESTS FAILED')
        sys.exit(1)
