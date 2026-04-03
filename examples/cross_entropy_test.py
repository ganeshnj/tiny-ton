"""Composed cross_entropy (7 kernel launches) -- verified against NumPy.

Run: PYTHONPATH="build/bindings:python" python3 examples/cross_entropy_test.py
"""

import numpy as np
import tiny_ton as tt


# --- softmax kernels (5 launches) -------------------------------------------

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


def softmax(x, out, N):
    grid = (max(1, (N + 63) // 64),)
    tmp_max = np.zeros(1, dtype=x.dtype)
    tmp_exp = np.zeros(N, dtype=x.dtype)
    tmp_sum = np.zeros(1, dtype=x.dtype)
    kern_reduce_max[(1,)](x, tmp_max, N)
    kern_sub_scalar[grid](x, tmp_max, tmp_exp, N)
    kern_exp[grid](tmp_exp, tmp_exp, N)
    kern_reduce_sum[(1,)](tmp_exp, tmp_sum, N)
    kern_div_scalar[grid](tmp_exp, tmp_sum, out, N)


# --- kernel 6: gather scalar ------------------------------------------------

@tt.jit
def kern_gather_scalar(src, index, dst):
    tid = tt.arange(0, 64)
    val = tt.load(src + index)
    tt.store(dst, val)


# --- kernel 7: negative log -------------------------------------------------

@tt.jit
def kern_neg_log(src, dst):
    tid = tt.arange(0, 64)
    val = tt.load(src)
    result = 0.0 - tt.log(val)
    tt.store(dst, result)


# --- orchestrator ------------------------------------------------------------

def cross_entropy(logits, target, N):
    """Host-side cross_entropy: 7 kernel launches."""
    probs = np.zeros(N, dtype=logits.dtype)
    softmax(logits, probs, N)

    p = np.zeros(1, dtype=logits.dtype)
    kern_gather_scalar[(1,)](probs, target, p)

    loss = np.zeros(1, dtype=logits.dtype)
    kern_neg_log[(1,)](p, loss)
    return loss[0]


def numpy_cross_entropy(logits, target):
    shifted = logits - np.max(logits)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    return -(shifted[target] - log_sum_exp)


def main():
    np.random.seed(42)
    all_ok = True

    N = 16
    logits = np.random.randn(N).astype(np.float32) * 2.0

    for target in [0, 5, 15]:
        expected = numpy_cross_entropy(logits, target)
        got = cross_entropy(logits.copy(), target, N)

        ok = np.allclose(got, expected, atol=1e-4)
        positive = got > 0.0
        finite = np.isfinite(got)
        passed = ok and positive and finite

        tag = "PASS" if passed else "FAIL"
        print(f"cross_entropy target={target:2d}: {tag}  "
              f"got={got:.6f}  expected={expected:.6f}")
        if not passed:
            all_ok = False

    assert all_ok, "cross_entropy test failed"
    print("All cross_entropy tests passed.")


if __name__ == "__main__":
    main()
