"""Dot product and matvec -- verified against NumPy.

Run: PYTHONPATH="build/bindings:python" python3 examples/dot_matvec_test.py
"""

import numpy as np
import tiny_ton as tt


@tt.jit
def dot_kernel(a_ptr, b_ptr, dst_ptr, N):
    tid = tt.arange(0, 64)
    mask = tid < N
    a = tt.load(a_ptr + tid, mask=mask)
    b = tt.load(b_ptr + tid, mask=mask)
    result = tt.reduce_sum(a * b)
    tt.store(dst_ptr, result)


@tt.jit
def matvec_kernel(W_ptr, x_ptr, y_ptr, in_features):
    pid = tt.program_id(0)
    tid = tt.arange(0, 64)
    mask = tid < in_features
    w = tt.load(W_ptr + pid * in_features + tid, mask=mask)
    x = tt.load(x_ptr + tid, mask=mask)
    dot = tt.reduce_sum(w * x)
    tt.store(y_ptr + pid, dot)


def main():
    np.random.seed(42)
    all_ok = True

    # --- dot product ---
    N = 16
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    expected_dot = np.dot(a, b)
    dst = np.zeros(1, dtype=np.float32)
    dot_kernel[(1,)](a.copy(), b.copy(), dst, N)

    ok = np.allclose(dst[0], expected_dot, atol=1e-4)
    print(f"dot f32: {'PASS' if ok else 'FAIL'}  got={dst[0]:.6f}  expected={expected_dot:.6f}")
    if not ok:
        all_ok = False

    # --- matvec ---
    out_features, in_features = 4, 16
    W = np.random.randn(out_features, in_features).astype(np.float32)
    x = np.random.randn(in_features).astype(np.float32)
    expected_y = W @ x
    W_flat = W.flatten()
    y = np.zeros(out_features, dtype=np.float32)
    matvec_kernel[(out_features,)](W_flat.copy(), x.copy(), y, in_features)

    ok_mv = np.allclose(y, expected_y, atol=1e-4)
    print(f"matvec f32: {'PASS' if ok_mv else 'FAIL'}")
    if not ok_mv:
        for i in range(out_features):
            print(f"  [{i}] got={y[i]:.6f}  expected={expected_y[i]:.6f}")
        all_ok = False

    assert all_ok, "dot/matvec test failed"
    print("All dot/matvec tests passed.")


if __name__ == "__main__":
    main()
