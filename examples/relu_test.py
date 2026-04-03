"""Element-wise ReLU -- verified against NumPy.

Run: PYTHONPATH="build/bindings:python" python3 examples/relu_test.py
"""

import numpy as np
import tiny_ton as tt


@tt.jit
def kern_relu(src, dst, N):
    pid = tt.program_id(0)
    off = pid * 64 + tt.arange(0, 64)
    mask = off < N
    x = tt.load(src + off, mask=mask)
    y = tt.relu(x)
    tt.store(dst + off, y, mask=mask)


def main():
    np.random.seed(42)
    BLOCK = 64
    N = 256
    grid = (N // BLOCK,)

    x = np.array(
        np.random.uniform(-5.0, 5.0, N), dtype=np.float32
    )
    expected = np.maximum(x, 0.0)

    dst = np.zeros(N, dtype=np.float32)
    kern_relu[grid](x.copy(), dst, N)

    ok = np.allclose(dst, expected, atol=0.0)
    print(f"relu f32: {'PASS' if ok else 'FAIL'}")
    if not ok:
        for i in range(min(8, N)):
            print(f"  [{i}] got={dst[i]:.6f}  expected={expected[i]:.6f}")

    # --- f16 ---
    x16 = x.astype(np.float16)
    expected16 = np.maximum(x16, np.float16(0.0))
    dst16 = np.zeros(N, dtype=np.float16)
    kern_relu[grid](x16.copy(), dst16, N)

    ok16 = np.allclose(dst16, expected16, atol=0.0)
    print(f"relu f16: {'PASS' if ok16 else 'FAIL'}")
    if not ok16:
        for i in range(min(8, N)):
            print(f"  [{i}] got={dst16[i]:.4f}  expected={expected16[i]:.4f}")

    assert ok and ok16, "relu test failed"
    print("All relu tests passed.")


if __name__ == "__main__":
    main()
