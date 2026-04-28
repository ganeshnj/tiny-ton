# Tiled GEMM via Compile-Time Loop Unrolling

## The problem

After adding shared memory (`tt.sync`, `tt.shared_store`, `tt.shared_load` — see
[15-shared-memory.md](15-shared-memory.md)), the 2-row-per-block matvec can reuse
the input vector `x` for two output rows, halving global reads.  The natural
next step is full tiled GEMM: tile **both** dimensions (output rows and the
reduction dimension K) so that an M×K × K×N matmul only reads each tile of
A and B from global memory once, reusing it across many output elements.

Full tiled GEMM requires a **loop** over K tiles:

```python
acc = 0.0
for k0 in range(0, K, TILE_K):          # iterate over K tiles
    load A[row, k0:k0+TILE_K]
    load B[k0:k0+TILE_K, col]
    acc += dot(a_tile, b_tile)
store C[row, col] = acc
```

The loop variable `k0` takes values `0, TILE_K, 2*TILE_K, …` and is used in
pointer arithmetic.  The accumulator `acc` carries a partial sum across
iterations.

## Approach: compile-time unrolling

The TinyTon IR is a **flat list of SSA ops in a single block** — there are no
back-edges, phi nodes, or structured loop regions.  Adding a true runtime loop
(e.g. `scf.for`) would require CFG-aware liveness analysis, phi insertion in
register allocation, and a backward-jump instruction in the simulator ISA —
substantial new infrastructure.

Instead, we **unroll the loop at JIT compile time** when all bounds (`start`,
`stop`, `step`) are known statically (i.e. they are Python literals or
`tt.constexpr` parameters).  The compiler walks the Python AST and emits the
loop body once for each concrete value of the loop variable — producing
straight-line IR that the existing stack handles without any changes.

### Why this is sufficient for tiled GEMM

In tiled GEMM, the tile size `TILE_K` and the reduction dimension `K` are
constant per kernel compilation (both annotated `tt.constexpr`).  The number
of tiles `K // TILE_K` is therefore fixed at compile time.  Unrolling
`K // TILE_K` iterations is fine for typical block sizes (e.g. `K=64`,
`TILE_K=16` → 4 iterations).

## Implementation: two changes to `jit.py`

All changes are Python-only.  No C++, MLIR dialect, or simulator modifications
were needed.

### 1. `visit_For` — loop unrolling

```python
def visit_For(self, node: ast.For):
    loop_var = node.target.id               # e.g. "k0"
    # Evaluate range(start, stop, step) at compile time
    raw = [self._eval_python_int(a) for a in node.iter.args]
    start, stop, step = ...                 # unpack 1-, 2-, or 3-arg range
    for val in range(start, stop, step):
        self.symbols[loop_var] = val        # Python int
        for stmt in node.body:
            self.visit(stmt)               # emit IR for this iteration
    self.symbols.pop(loop_var, None)
```

The loop variable is stored as a plain Python `int` in `symbols`.

### 2. `_eval(ast.Name)` — auto-promote scalars to IR constants

Before this change, `symbols[name]` was returned as-is.  For constexpr
parameters (already stored as Python ints) or loop variables this broke
pointer arithmetic like `A_ptr + k0 + tid` because `emit_add` expects MLIR
Values.  The fix:

```python
if isinstance(node, ast.Name):
    val = self.symbols[node.id]
    if isinstance(val, int):    return self.builder.emit_const(val)
    if isinstance(val, float):  return self.builder.emit_fconst(val)
    return val   # already an MLIR Value
```

This means any Python int or float in `symbols` — whether from a
`tt.constexpr` parameter or a loop variable — is transparently promoted to
an IR constant whenever it appears in an expression.

### `_eval_python_int` helper

Used to evaluate range bounds as Python integers at compile time.  Accepts
literals, `tt.constexpr` names, loop variable names, and simple arithmetic
combinations:

```python
def _eval_python_int(self, node) -> int:
    if ast.Constant:  return int(node.value)
    if ast.Name:      look up constexpr_values or symbols (must be int)
    if ast.BinOp:     recurse with +, -, *, //
```

## SSA accumulator pattern

A common pattern in tiled kernels is:

```python
acc = 0.0
for k0 in range(0, K, TILE_K):
    ...
    acc = acc + tt.reduce_sum(a_tile * b_tile)
```

After unrolling with `K=8`, `TILE_K=4` (2 iterations):

```
%acc0 = emit_fconst(0.0)
# ---- k0=0 ----
%a0   = emit_load(A_ptr + row*8 + 0 + tid)
%b0   = emit_load(B_ptr + 0 + tid)
%rs0  = emit_reduce_sum(emit_mul(%a0, %b0))
%acc1 = emit_add(%acc0, %rs0)
# ---- k0=4 ----
%a1   = emit_load(A_ptr + row*8 + 4 + tid)
%b1   = emit_load(B_ptr + 4 + tid)
%rs1  = emit_reduce_sum(emit_mul(%a1, %b1))
%acc2 = emit_add(%acc1, %rs1)
# ---- store ----
emit_store(C_ptr + row, %acc2)
```

Each `acc = acc + x` statement overwrites `symbols['acc']` with the new MLIR
Value, so the chain `%acc0 → %acc1 → %acc2` forms naturally without any phi
nodes.

## Kernels in `examples/tiled_gemm_test.py`

### `loop_sum_kernel`

Sums N elements using a tiled for loop.  Proves basic accumulation and masked
loads work across iterations.

```python
@tt.jit
def loop_sum_kernel(src_ptr, dst_ptr, N: tt.constexpr, TILE: tt.constexpr):
    tid = tt.arange(0, TILE)
    acc = 0.0
    for t in range(0, N, TILE):
        mask = tid < (N - t)
        x = tt.load(src_ptr + t + tid, mask=mask, other=0.0)
        acc = acc + tt.reduce_sum(x)
    tt.store(dst_ptr + tid, acc, mask=tid < 1)
```

### `tiled_dot_kernel`

Tiled matvec `C[row] = dot(A[row, :], B[:])`.  One block per output element,
loop over K tiles.

```python
@tt.jit
def tiled_dot_kernel(A_ptr, B_ptr, C_ptr, K: tt.constexpr, TILE_K: tt.constexpr):
    row = tt.program_id(0)
    tid = tt.arange(0, TILE_K)
    acc = 0.0
    for k0 in range(0, K, TILE_K):
        a_tile = tt.load(A_ptr + row * K + k0 + tid)
        b_tile = tt.load(B_ptr + k0 + tid)
        acc = acc + tt.reduce_sum(a_tile * b_tile)
    tt.store(C_ptr + row, acc)
```

### `tiled_matmul_kernel`

Full GEMM `C[M,N] = A[M,K] @ B[K,N]`.  Nested for loops: outer loop over
output columns (unrolled `N` times), inner loop over K tiles (unrolled
`K//TILE_K` times).  One block per output row, grid size `(M,)`.

```python
@tt.jit
def tiled_matmul_kernel(A_ptr, B_ptr, C_ptr,
                         K: tt.constexpr, N: tt.constexpr,
                         TILE_K: tt.constexpr):
    row = tt.program_id(0)
    tid = tt.arange(0, TILE_K)
    for col in range(N):
        acc = 0.0
        for k0 in range(0, K, TILE_K):
            a_tile = tt.load(A_ptr + row * K + k0 + tid)
            b_tile = tt.load(B_ptr + k0 * N + col + tid * N)   # strided B col
            acc = acc + tt.reduce_sum(a_tile * b_tile)
        tt.store(C_ptr + row * N + col, acc)
```

The nested loop is unrolled to `N * (K // TILE_K)` copies of the body at
compile time.  For the test parameters `N=8, K=8, TILE_K=4` this is 16 copies
— still compact IR.

## What this is NOT

**No shared memory reuse for B tiles.**  In `tiled_matmul_kernel`, each B
element `B[k, col]` is loaded from global memory by each block independently.
The canonical GEMM optimization loads a tile of B into shared memory once so
that multiple output rows (in different blocks or via different threads) can
reuse it.  That requires launching with a 2D grid *and* coordinating across
rows within a block — beyond the current single-block-per-row model.

**No runtime-variable loop bounds.**  The `for k0 in range(0, K, TILE_K)`
construct requires `K` and `TILE_K` to be `tt.constexpr`.  To support a
runtime `K` (passed as an ordinary integer argument), the IR would need a
real loop region.

## Future work: `scf.for` for runtime loops

For runtime-variable loop bounds, the correct MLIR representation is
`scf.for`:

```mlir
scf.for %k0 = %c0 to %K step %TILE_K iter_args(%acc = %c0) {
    ...
    scf.yield %new_acc
}
```

This requires:

1. A new `TinyTon_ForRangeOp` with a body region in the TableGen dialect.
2. CFG-aware liveness analysis in `RegisterAlloc.cpp` (phi nodes at loop
   header).
3. A backward branch instruction in the simulator ISA (currently only forward
   conditional skips exist via `BZ`).
4. Lowering in `TinyTonToGPU.cpp` from `ForRangeOp` → `scf.for` (the
   `MLIRSCFDialect` is already linked in `lib/Compiler/CMakeLists.txt`).

Compile-time unrolling covers the common case (fixed tile sizes) and is a
useful foundation regardless of whether runtime loops are added later.
