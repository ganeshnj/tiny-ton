"""Linear (matvec) at microgpt scale + chained layers -- verified against NumPy.

Run: PYTHONPATH="build/bindings:python" python3 examples/linear_test.py
"""

import numpy as np
import tiny_ton as tt


@tt.jit
def linear_kernel(W_ptr, x_ptr, y_ptr, in_features):
    pid = tt.program_id(0)
    tid = tt.arange(0, 64)
    mask = tid < in_features
    w = tt.load(W_ptr + pid * in_features + tid, mask=mask)
    x = tt.load(x_ptr + tid, mask=mask)
    dot = tt.reduce_sum(w * x)
    tt.store(y_ptr + pid, dot)


@tt.jit
def kern_relu(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    y = tt.relu(x)
    tt.store(dst + off, y, mask=mask)


def linear(W_flat, x, y, out_features, in_features):
    linear_kernel[(out_features,)](W_flat, x, y, in_features)


def main():
    np.random.seed(42)
    all_ok = True

    # --- single linear (attention-projection-like: 16 -> 8) ---
    in_f, out_f = 16, 8
    W = np.random.randn(out_f, in_f).astype(np.float32)
    x = np.random.randn(in_f).astype(np.float32)
    expected = W @ x

    y = np.zeros(out_f, dtype=np.float32)
    linear(W.flatten().copy(), x.copy(), y, out_f, in_f)

    ok = np.allclose(y, expected, atol=1e-4)
    print(f"linear (16->8): {'PASS' if ok else 'FAIL'}")
    if not ok:
        for i in range(out_f):
            print(f"  [{i}] got={y[i]:.6f}  expected={expected[i]:.6f}")
        all_ok = False

    # --- chained linears: x -> W1 -> relu -> W2 -> y (MLP-like) ---
    in_f, hidden, out_f = 16, 32, 16
    W1 = np.random.randn(hidden, in_f).astype(np.float32)
    W2 = np.random.randn(out_f, hidden).astype(np.float32)
    x = np.random.randn(in_f).astype(np.float32)

    expected_h = W1 @ x
    expected_h_relu = np.maximum(expected_h, 0.0)
    expected_y = W2 @ expected_h_relu

    h = np.zeros(hidden, dtype=np.float32)
    h_relu = np.zeros(hidden, dtype=np.float32)
    y2 = np.zeros(out_f, dtype=np.float32)

    linear(W1.flatten().copy(), x.copy(), h, hidden, in_f)
    kern_relu[(1,)](h, h_relu, hidden)
    linear(W2.flatten().copy(), h_relu, y2, out_f, hidden)

    ok2 = np.allclose(y2, expected_y, atol=1e-3)
    print(f"chained linear (16->32->relu->16): {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        for i in range(min(8, out_f)):
            print(f"  [{i}] got={y2[i]:.6f}  expected={expected_y[i]:.6f}")
        all_ok = False

    assert all_ok, "linear test failed"
    print("All linear tests passed.")


if __name__ == "__main__":
    main()
