"""Test tt.constexpr — compile-time block size parameter.

Verifies:
  - BLOCK: tt.constexpr sets block size from the call site
  - Different BLOCK values produce separate cache entries
  - Masked loads work correctly when BLOCK > N
  - Backward compatibility: kernels with literal tt.arange(0, 64) are unchanged
  - Works end-to-end with fused softmax at various block sizes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import tiny_ton as tt


def compare(name, got, expected, atol=1e-5):
    ok = np.allclose(got, expected, atol=atol)
    print(f'  {name}: {"PASS" if ok else "FAIL"}')
    if not ok:
        print(f'    got:      {got}')
        print(f'    expected: {expected}')
    return ok


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------

@tt.jit
def relu_constexpr(src, dst, N, BLOCK: tt.constexpr):
    tid  = tt.arange(0, BLOCK)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask)
    tt.store(dst + tid, tt.relu(x), mask=mask)


@tt.jit
def relu_literal(src, dst, N):
    tid  = tt.arange(0, 64)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask)
    tt.store(dst + tid, tt.relu(x), mask=mask)


@tt.jit
def fused_softmax_constexpr(src, dst, N, BLOCK: tt.constexpr):
    tid  = tt.arange(0, BLOCK)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask, other=-float('inf'))
    mx   = tt.reduce_max(x)
    e    = tt.exp(x - mx)
    s    = tt.reduce_sum(e)
    tt.store(dst + tid, e / s, mask=mask)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_relu_constexpr():
    print('--- relu with constexpr BLOCK ---')
    all_ok = True
    for BLOCK, N in [(4, 4), (16, 4), (27, 27), (64, 64)]:
        x = np.random.randn(N).astype(np.float32)
        out = np.zeros(N, dtype=np.float32)
        relu_constexpr[(1,)](x.copy(), out, N, BLOCK)
        ok = compare(f'BLOCK={BLOCK}, N={N}', out, np.maximum(x, 0))
        all_ok = all_ok and ok
    return all_ok


def test_cache_separation():
    print('--- cache entries are separate per BLOCK value ---')
    relu_constexpr._cache.clear()
    x = np.zeros(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    relu_constexpr[(1,)](x.copy(), out, 4, 4)
    relu_constexpr[(1,)](x.copy(), out, 4, 16)
    relu_constexpr[(1,)](x.copy(), out, 4, 16)  # cache hit
    n = len(relu_constexpr._cache)
    ok = n == 2
    print(f'  cache size after BLOCK=4,16,16: {n} {"PASS" if ok else "FAIL"}')
    return ok


def test_backward_compat():
    print('--- backward compat: literal tt.arange(0, 64) ---')
    x = np.array([-1.0, 2.0, 3.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    relu_literal[(1,)](x.copy(), out, 3)
    return compare('relu_literal', out, np.maximum(x, 0))


def test_fused_softmax_constexpr():
    print('--- fused softmax with constexpr BLOCK ---')
    all_ok = True
    for N in [4, 16, 27]:
        x = np.random.randn(N).astype(np.float32)
        out = np.zeros(N, dtype=np.float32)
        fused_softmax_constexpr[(1,)](x.copy(), out, N, N)
        e = np.exp(x - x.max())
        ok = compare(f'N={N}', out, e / e.sum(), atol=1e-5)
        all_ok = all_ok and ok
    return all_ok


if __name__ == '__main__':
    np.random.seed(0)
    results = [
        test_relu_constexpr(),
        test_cache_separation(),
        test_backward_compat(),
        test_fused_softmax_constexpr(),
    ]
    print()
    if all(results):
        print('All tests PASSED')
    else:
        print('SOME TESTS FAILED')
        sys.exit(1)
