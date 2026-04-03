# cross_entropy (Cross-Entropy Loss) — Design & Implementation

## What it does

Cross-entropy loss measures how well a probability distribution (from softmax) matches the true label. For a single token prediction:

```
loss = -log(softmax(logits)[target])
```

In microgpt:

```python
def cross_entropy(logits, target):
    probs = softmax(logits)
    return -math.log(probs[target])
```

This is the training loss function — it drives the model to increase the probability assigned to the correct next token.

## 7-kernel decomposition

Cross-entropy composes softmax (5 launches) with two additional scalar operations:

```
Steps 1-5:  probs = softmax(logits)       →  5 kernel launches (see softmax.md)
Step 6:     p = probs[target]             →  kern_gather_scalar  (1 block, scalar)
Step 7:     loss = -log(p)                →  kern_neg_log         (1 block, scalar)
```

## New kernels (2)

Both operate on single scalar values:

```python
@tt.jit
def kern_gather_scalar(src, index, dst):
    tid = tt.arange(0, 64)
    val = tt.load(src + index)
    tt.store(dst, val)

@tt.jit
def kern_neg_log(src, dst):
    tid = tt.arange(0, 64)
    val = tt.load(src)
    result = 0.0 - tt.log(val)
    tt.store(dst, result)
```

`kern_gather_scalar` picks a single element from a vector by index — `tt.load(src + index)` computes the address as pointer + offset. All threads load the same element (scalar broadcast).

`kern_neg_log` loads a scalar probability, computes `0.0 - log(p)`, and stores the result. Uses existing `tt.log`.

## Python orchestrator

```python
def cross_entropy(logits, target, N):
    probs = np.zeros(N, dtype=logits.dtype)
    softmax(logits, probs, N)

    p = np.zeros(1, dtype=logits.dtype)
    kern_gather_scalar[(1,)](probs, target, p)

    loss = np.zeros(1, dtype=logits.dtype)
    kern_neg_log[(1,)](p, loss)
    return loss[0]
```

## Constraint: N <= 64

Same as softmax — the reduction kernels require the full vector to fit in one block. microgpt's vocab is small enough for this.

## How it maps to microgpt

microgpt computes cross-entropy for each token position during training:

```python
logits = linear(x, state_dict['lm_head'])
loss = cross_entropy(logits, next_token)
```

With tiny-ton, `linear` is 1 kernel launch and `cross_entropy` is 7, totaling 8 launches per token.

## How it flows through the stack

```
Python orchestrator: 7 x kernel[grid](...)
    │
    ▼
JIT (for each kernel): existing builtins
    │  softmax (5 kernels) → reduce_max, sub, exp, reduce_sum, div
    │  gather_scalar        → tinyton.load (address math)
    │  neg_log              → tinyton.load + tinyton.log + tinyton.sub
    │
    ▼
GPU lowering (existing patterns)
    │  all ops already lowered
    │
    ▼
PTX: 7 separate kernel launches via cuLaunchKernel
```

## Files changed

| File | What |
|------|------|
| `docs/cross-entropy.md` | This design doc |
| `examples/cross_entropy_test.py` | Standalone test: 7-launch cross_entropy vs NumPy |

No C++ files changed. No new builtins in `jit.py`. All kernels are user-written.

## Testing strategy

`examples/cross_entropy_test.py` creates random f32 logits of length 16, picks several target indices, runs the 7-launch cross_entropy, and compares against the NumPy reference: `-log(softmax(logits)[target])`. Tolerance: `atol=1e-4` (float accumulation across 7 kernels). Also verifies loss is positive and finite.
