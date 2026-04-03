"""Single-head attention (12 kernel launches) -- verified against NumPy.

Run: PYTHONPATH="build/bindings:python" python3 examples/attention_test.py
"""

import numpy as np
import tiny_ton as tt


# --- linear / matvec kernel (same kernel, different grid) -------------------

@tt.jit
def linear_kernel(W_ptr, x_ptr, y_ptr, in_features):
    pid = tt.program_id(0)
    tid = tt.arange(0, 64)
    mask = tid < in_features
    w = tt.load(W_ptr + pid * in_features + tid, mask=mask)
    x = tt.load(x_ptr + tid, mask=mask)
    dot = tt.reduce_sum(w * x)
    tt.store(y_ptr + pid, dot)


matvec_kernel = linear_kernel


# --- softmax kernels (reused from softmax_test) -----------------------------

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
    """Host-side softmax: 5 kernel launches."""
    grid = (max(1, (N + 63) // 64),)
    tmp_max = np.zeros(1, dtype=x.dtype)
    tmp_exp = np.zeros(N, dtype=x.dtype)
    tmp_sum = np.zeros(1, dtype=x.dtype)

    kern_reduce_max[(1,)](x, tmp_max, N)
    kern_sub_scalar[grid](x, tmp_max, tmp_exp, N)
    kern_exp[grid](tmp_exp, tmp_exp, N)
    kern_reduce_sum[(1,)](tmp_exp, tmp_sum, N)
    kern_div_scalar[grid](tmp_exp, tmp_sum, out, N)


# --- attention orchestrator (12 launches) -----------------------------------

def attention(x, Wq, Wk, Wv, Wo, K_cache, V_cache, n_embd):
    """Single-head scaled dot-product attention with Q/K/V/O projections."""
    q = np.zeros(n_embd, dtype=np.float32)
    k = np.zeros(n_embd, dtype=np.float32)
    v = np.zeros(n_embd, dtype=np.float32)

    linear_kernel[(n_embd,)](Wq, x, q, n_embd)
    linear_kernel[(n_embd,)](Wk, x, k, n_embd)
    linear_kernel[(n_embd,)](Wv, x, v, n_embd)

    K_cache.append(k.copy())
    V_cache.append(v.copy())
    K = np.ascontiguousarray(np.vstack(K_cache))
    V = np.vstack(V_cache)
    seq_len = len(K_cache)

    scores = np.zeros(seq_len, dtype=np.float32)
    matvec_kernel[(seq_len,)](K.flatten(), q, scores, n_embd)

    sqrt_d = np.array([np.sqrt(float(n_embd))], dtype=np.float32)
    scores_scaled = np.zeros(seq_len, dtype=np.float32)
    kern_div_scalar[(1,)](scores, sqrt_d, scores_scaled, seq_len)

    weights = np.zeros(seq_len, dtype=np.float32)
    softmax(scores_scaled, weights, seq_len)

    V_T = np.ascontiguousarray(V.T)
    attn_out = np.zeros(n_embd, dtype=np.float32)
    matvec_kernel[(n_embd,)](V_T.flatten(), weights, attn_out, seq_len)

    output = np.zeros(n_embd, dtype=np.float32)
    linear_kernel[(n_embd,)](Wo, attn_out, output, n_embd)
    return output


# --- NumPy reference --------------------------------------------------------

def attention_numpy(x, Wq, Wk, Wv, Wo, K_cache, V_cache, n_embd):
    """NumPy reference for single-head attention."""
    q = Wq @ x
    k = Wk @ x
    v = Wv @ x

    K_cache.append(k.copy())
    V_cache.append(v.copy())
    K = np.vstack(K_cache)
    V = np.vstack(V_cache)

    scores = K @ q / np.sqrt(float(n_embd))
    shifted = scores - np.max(scores)
    w = np.exp(shifted) / np.sum(np.exp(shifted))
    attn_out = V.T @ w
    return Wo @ attn_out


# --- test -------------------------------------------------------------------

def main():
    np.random.seed(42)
    n_embd = 16
    n_tokens = 4

    Wq = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    Wk = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    Wv = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    Wo = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1

    tokens = [np.random.randn(n_embd).astype(np.float32) for _ in range(n_tokens)]

    K_gpu, V_gpu = [], []
    K_ref, V_ref = [], []

    all_ok = True
    for t in range(n_tokens):
        x = tokens[t]
        gpu_out = attention(x.copy(),
                            Wq.flatten().copy(), Wk.flatten().copy(),
                            Wv.flatten().copy(), Wo.flatten().copy(),
                            K_gpu, V_gpu, n_embd)
        ref_out = attention_numpy(x.copy(), Wq, Wk, Wv, Wo,
                                  K_ref, V_ref, n_embd)

        ok = np.allclose(gpu_out, ref_out, atol=1e-3)
        print(f"attention pos={t} (seq_len={t+1}): {'PASS' if ok else 'FAIL'}")
        if not ok:
            for i in range(n_embd):
                print(f"  [{i}] got={gpu_out[i]:.6f}  expected={ref_out[i]:.6f}")
            all_ok = False

    assert all_ok, "attention test failed"
    print("All attention tests passed.")


if __name__ == "__main__":
    main()
