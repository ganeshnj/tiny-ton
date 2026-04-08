"""Fused attention (7 kernel launches) -- verified against NumPy and 12-launch version.

Fusions applied:
  - 5-kernel softmax → fused_scaled_softmax (includes score / sqrt(d))
  - kern_div_scalar folded into fused_scaled_softmax

Launch count: 3 (linear) + 1 (matvec scores) + 1 (fused scaled softmax)
              + 1 (matvec V^T@w) + 1 (linear Wo) = 7

Run: PYTHONPATH="build/bindings:python" python3 examples/fused_attention_test.py
"""

import numpy as np
import tiny_ton as tt


# --- linear / matvec kernel (reused from attention_test.py) ------------------

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


# --- fused scaled softmax (score/sqrt_d + softmax in one kernel) -------------

@tt.jit
def fused_scaled_softmax_kernel(src, dst, N, sqrt_d_ptr):
    tid  = tt.arange(0, 64)
    mask = tid < N
    x    = tt.load(src + tid, mask=mask, other=-float('inf'))
    sd   = tt.load(sqrt_d_ptr)
    x    = x / sd
    mx   = tt.reduce_max(x)
    e    = tt.exp(x - mx)
    s    = tt.reduce_sum(e)
    tt.store(dst + tid, e / s, mask=mask)


# --- fused attention orchestrator (7 launches) --------------------------------

def fused_attention(x, Wq, Wk, Wv, Wo, K_cache, V_cache, n_embd):
    """Single-head attention with fused scaled softmax. 7 kernel launches."""
    q = np.zeros(n_embd, dtype=np.float32)
    k = np.zeros(n_embd, dtype=np.float32)
    v = np.zeros(n_embd, dtype=np.float32)

    linear_kernel[(n_embd,)](Wq, x, q, n_embd)        # launch 1
    linear_kernel[(n_embd,)](Wk, x, k, n_embd)        # launch 2
    linear_kernel[(n_embd,)](Wv, x, v, n_embd)        # launch 3

    K_cache.append(k.copy())
    V_cache.append(v.copy())
    K = np.ascontiguousarray(np.vstack(K_cache))
    V = np.vstack(V_cache)
    seq_len = len(K_cache)

    scores = np.zeros(seq_len, dtype=np.float32)
    matvec_kernel[(seq_len,)](K.flatten(), q, scores, n_embd)  # launch 4

    sqrt_d = np.array([np.sqrt(float(n_embd))], dtype=np.float32)
    weights = np.zeros(seq_len, dtype=np.float32)
    fused_scaled_softmax_kernel[(1,)](scores, weights, seq_len, sqrt_d)  # launch 5

    V_T = np.ascontiguousarray(V.T)
    attn_out = np.zeros(n_embd, dtype=np.float32)
    matvec_kernel[(n_embd,)](V_T.flatten(), weights, attn_out, seq_len)  # launch 6

    output = np.zeros(n_embd, dtype=np.float32)
    linear_kernel[(n_embd,)](Wo, attn_out, output, n_embd)  # launch 7
    return output


# --- NumPy reference --------------------------------------------------------

def attention_numpy(x, Wq, Wk, Wv, Wo, K_cache, V_cache, n_embd):
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

    K_fused, V_fused = [], []
    K_ref, V_ref = [], []

    all_ok = True
    for t in range(n_tokens):
        x = tokens[t]
        fused_out = fused_attention(x.copy(),
                                    Wq.flatten().copy(), Wk.flatten().copy(),
                                    Wv.flatten().copy(), Wo.flatten().copy(),
                                    K_fused, V_fused, n_embd)
        ref_out = attention_numpy(x.copy(), Wq, Wk, Wv, Wo,
                                  K_ref, V_ref, n_embd)

        ok = np.allclose(fused_out, ref_out, atol=1e-3)
        print(f"fused_attention pos={t} (seq_len={t+1}): {'PASS' if ok else 'FAIL'}")
        if not ok:
            for i in range(n_embd):
                print(f"  [{i}] fused={fused_out[i]:.6f}  ref={ref_out[i]:.6f}")
            all_ok = False

    assert all_ok, "fused_attention test failed"
    print("All fused_attention tests passed.")
    print()
    print("Launch count comparison (per attention call):")
    print("  12-launch version: 3 linear + 1 matvec + 1 div + 5 softmax + 1 matvec + 1 linear")
    print("  7-launch version:  3 linear + 1 matvec + 1 fused_scaled_softmax + 1 matvec + 1 linear")


if __name__ == "__main__":
    main()
