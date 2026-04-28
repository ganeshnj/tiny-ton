"""JIT decorator and Python AST -> C++ IR bridge."""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from typing import Any, Optional

import numpy as np


_BUILTINS = {"program_id", "arange", "load", "store",
             "exp", "log", "sqrt", "rsqrt", "abs", "max", "relu",
             "reduce_sum", "reduce_max",
             "sync", "shared_store", "shared_load"}

# Module aliases that should be treated as the tiny_ton namespace.
_MODULE_ALIASES = {"tt", "tiny_ton"}

# Map numpy dtypes to tiny-ton dtype strings.
_DTYPE_MAP = {
    np.dtype("int32"): "i32",
    np.dtype("float32"): "f32",
    np.dtype("float16"): "f16",
}


def _is_constexpr_annotation(ann: ast.expr) -> bool:
    """Return True if *ann* is ``tt.constexpr`` or bare ``constexpr``."""
    if isinstance(ann, ast.Attribute):
        return (isinstance(ann.value, ast.Name)
                and ann.value.id in _MODULE_ALIASES
                and ann.attr == "constexpr")
    if isinstance(ann, ast.Name):
        return ann.id == "constexpr"
    return False


class KernelVisitor(ast.NodeVisitor):
    """Walks a Python AST and emits TinyTon IR via the C++ IRBuilder."""

    def __init__(self, builder, param_names: list[str],
                 arg_is_pointer: Optional[list[bool]] = None,
                 arg_dtypes: Optional[list[str]] = None,
                 constexpr_values: Optional[dict[str, int]] = None):
        self.builder = builder
        self.symbols: dict[str, Any] = {}
        self.block_size: Optional[int] = None
        self._arg_dtypes = arg_dtypes or []
        self._constexpr_values = constexpr_values or {}
        self._kernel_dtype = "i32"
        for d in self._arg_dtypes:
            if d in ("f32", "f16"):
                self._kernel_dtype = d
                break

        ir_idx = 0
        for i, name in enumerate(param_names):
            if name in self._constexpr_values:
                # Constexpr params are compile-time ints, not IR args.
                self.symbols[name] = self._constexpr_values[name]
                continue
            is_ptr = arg_is_pointer[i] if arg_is_pointer else False
            dtype = arg_dtypes[i] if arg_dtypes else "i32"
            self.symbols[name] = builder.emit_arg(ir_idx, is_pointer=is_ptr,
                                                  dtype=dtype)
            ir_idx += 1

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for stmt in node.body:
            self.visit(stmt)
        self.builder.emit_ret()

    def visit_Assign(self, node: ast.Assign):
        value = self._eval(node.value)
        assert len(node.targets) == 1
        target = node.targets[0]
        assert isinstance(target, ast.Name)
        self.symbols[target.id] = value

    def visit_AnnAssign(self, node: ast.AnnAssign):
        assert node.value is not None, "annotation-only statements not supported"
        value = self._eval(node.value)
        assert isinstance(node.target, ast.Name)
        self.symbols[node.target.id] = value

    def visit_Expr(self, node: ast.Expr):
        self._eval(node.value)

    def visit_For(self, node: ast.For):
        """Unroll ``for x in range(...)`` at compile time.

        All range bounds must be Python literals or ``tt.constexpr``
        parameters (i.e. values known when the kernel is compiled).  The
        loop variable is stored as a Python int in ``symbols`` each
        iteration; ``_eval`` promotes it to an IR constant whenever it
        appears in an expression.
        """
        assert isinstance(node.target, ast.Name), (
            "for loop target must be a simple name")
        loop_var = node.target.id

        iter_node = node.iter
        assert (
            isinstance(iter_node, ast.Call)
            and isinstance(iter_node.func, ast.Name)
            and iter_node.func.id == "range"
        ), "tiny-ton for loops must use range(...) with constexpr bounds"

        raw = [self._eval_python_int(a) for a in iter_node.args]
        if len(raw) == 1:
            start, stop, step = 0, raw[0], 1
        elif len(raw) == 2:
            start, stop, step = raw[0], raw[1], 1
        else:
            start, stop, step = raw[0], raw[1], raw[2]

        for val in range(start, stop, step):
            self.symbols[loop_var] = val  # Python int → IR const in _eval
            for stmt in node.body:
                self.visit(stmt)

        self.symbols.pop(loop_var, None)

    def _eval_python_int(self, node: ast.expr) -> int:
        """Evaluate *node* as a Python integer at compile time.

        Accepts integer literals, constexpr parameter names, loop variable
        names (already stored as Python ints in ``symbols``), and simple
        arithmetic combinations thereof.  Used to evaluate ``range()``
        bounds in ``visit_For``.
        """
        if isinstance(node, ast.Constant):
            return int(node.value)
        if isinstance(node, ast.Name):
            if node.id in self._constexpr_values:
                return self._constexpr_values[node.id]
            if node.id in self.symbols and isinstance(self.symbols[node.id], int):
                return self.symbols[node.id]
            raise AssertionError(
                f"loop bound '{node.id}' must be a literal or a "
                f"tt.constexpr parameter")
        if isinstance(node, ast.BinOp):
            lhs = self._eval_python_int(node.left)
            rhs = self._eval_python_int(node.right)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.FloorDiv):
                return lhs // rhs
        raise NotImplementedError(
            f"loop bound must be a constexpr literal, got {ast.dump(node)}")

    # ------------------------------------------------------------------
    # Expression evaluator (returns a PyValue or None for void ops)
    # ------------------------------------------------------------------

    def _promote_scalar(self, val):
        """Promote a raw Python int/float to an IR constant Value.

        Called on BinOp operands as a safety net: constexpr params and loop
        variables are stored as Python ints in ``symbols`` and should already
        be promoted by ``_eval``, but if ``emit_const`` returns the raw value
        on some builds/platforms, this catches the fallthrough.
        """
        if isinstance(val, int):
            return self.builder.emit_const(val)
        if isinstance(val, float):
            if self._kernel_dtype == "f16":
                return self.builder.emit_hconst(val)
            return self.builder.emit_fconst(val)
        return val

    def _eval(self, node: ast.expr):
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, str):
                # Docstrings are Expr(Constant(str)) — ignore them.
                return None
            if isinstance(val, float):
                if self._kernel_dtype == "f16":
                    return self.builder.emit_hconst(val)
                return self.builder.emit_fconst(val)
            return self.builder.emit_const(int(val))

        if isinstance(node, ast.Name):
            assert node.id in self.symbols, f"undefined: {node.id}"
            val = self.symbols[node.id]
            # Auto-promote Python scalars (from constexpr params or loop
            # variables) to IR constants so they work in pointer arithmetic.
            if isinstance(val, int):
                return self.builder.emit_const(val)
            if isinstance(val, float):
                if self._kernel_dtype == "f16":
                    return self.builder.emit_hconst(val)
                return self.builder.emit_fconst(val)
            return val

        if isinstance(node, ast.BinOp):
            lhs = self._eval(node.left)
            rhs = self._eval(node.right)
            # Defensively promote Python scalars to IR constants in case a
            # constexpr param or loop variable slipped through as a raw int/float
            # (e.g. when emit_const returns the raw value on some builds).
            lhs = self._promote_scalar(lhs)
            rhs = self._promote_scalar(rhs)
            if isinstance(node.op, ast.Add):
                return self.builder.emit_add(lhs, rhs)
            if isinstance(node.op, ast.Sub):
                return self.builder.emit_sub(lhs, rhs)
            if isinstance(node.op, ast.Mult):
                return self.builder.emit_mul(lhs, rhs)
            if isinstance(node.op, ast.Div):
                return self.builder.emit_div(lhs, rhs)
            raise NotImplementedError(f"unsupported binop: {type(node.op)}")

        if isinstance(node, ast.Compare):
            assert len(node.ops) == 1 and len(node.comparators) == 1
            lhs = self._eval(node.left)
            rhs = self._eval(node.comparators[0])
            if isinstance(node.ops[0], ast.Lt):
                return self.builder.emit_cmp_lt(lhs, rhs)
            raise NotImplementedError(f"unsupported cmp: {type(node.ops[0])}")

        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            if isinstance(node.op, ast.USub):
                zero = self.builder.emit_fconst(0.0)
                return self.builder.emit_sub(zero, operand)
            raise NotImplementedError(f"unsupported unaryop: {type(node.op)}")

        if isinstance(node, ast.Call):
            return self._eval_call(node)

        raise NotImplementedError(f"unsupported AST node: {type(node)}")

    # ------------------------------------------------------------------
    # Call dispatch
    # ------------------------------------------------------------------

    def _eval_call(self, node: ast.Call):
        if (isinstance(node.func, ast.Name) and node.func.id == "float"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 'inf'):
            return self.builder.emit_fconst(float('inf'))

        builtin = self._resolve_builtin(node.func)
        if builtin is None:
            raise NotImplementedError(f"unsupported call: {ast.dump(node.func)}")

        if builtin == "program_id":
            axis = node.args[0].value if node.args else 0
            return self.builder.emit_program_id(int(axis))

        if builtin == "arange":
            start = node.args[0].value
            end_node = node.args[1]
            if isinstance(end_node, ast.Constant):
                end = end_node.value
            elif isinstance(end_node, ast.Name):
                # Support constexpr names: tt.arange(0, BLOCK)
                assert end_node.id in self._constexpr_values, (
                    f"arange end '{end_node.id}' must be a literal or a "
                    f"constexpr parameter")
                end = self._constexpr_values[end_node.id]
            else:
                raise NotImplementedError(
                    f"arange end must be a literal or constexpr name, "
                    f"got {ast.dump(end_node)}")
            self.block_size = int(end) - int(start)
            return self.builder.emit_thread_id(0)

        if builtin == "load":
            addr = self._eval(node.args[0])
            mask = self._get_kwarg(node, "mask")
            other = self._get_kwarg(node, "other")
            return self.builder.emit_load(addr, mask=mask, other=other,
                                          dtype=self._kernel_dtype)

        if builtin == "store":
            addr = self._eval(node.args[0])
            val = self._eval(node.args[1])
            mask = self._get_kwarg(node, "mask")
            self.builder.emit_store(addr, val, mask=mask)
            return None

        if builtin == "exp":
            return self.builder.emit_exp(self._eval(node.args[0]))
        if builtin == "log":
            return self.builder.emit_log(self._eval(node.args[0]))
        if builtin == "sqrt":
            return self.builder.emit_sqrt(self._eval(node.args[0]))
        if builtin == "rsqrt":
            return self.builder.emit_rsqrt(self._eval(node.args[0]))
        if builtin == "abs":
            return self.builder.emit_abs(self._eval(node.args[0]))
        if builtin == "max":
            a = self._eval(node.args[0])
            b = self._eval(node.args[1])
            return self.builder.emit_max(a, b)
        if builtin == "relu":
            x = self._eval(node.args[0])
            if self._kernel_dtype == "f16":
                zero = self.builder.emit_hconst(0.0)
            else:
                zero = self.builder.emit_fconst(0.0)
            return self.builder.emit_max(x, zero)

        if builtin == "reduce_sum":
            return self.builder.emit_reduce_sum(self._eval(node.args[0]))
        if builtin == "reduce_max":
            return self.builder.emit_reduce_max(self._eval(node.args[0]))

        if builtin == "sync":
            self.builder.emit_sync()
            return None
        if builtin == "shared_store":
            idx = self._eval(node.args[0])
            val = self._eval(node.args[1])
            assert self.block_size is not None, \
                "tt.arange must be called before tt.shared_store"
            self.builder.emit_shared_store(idx, val, self.block_size)
            return None
        if builtin == "shared_load":
            idx = self._eval(node.args[0])
            assert self.block_size is not None, \
                "tt.arange must be called before tt.shared_load"
            return self.builder.emit_shared_load(
                idx, self.block_size, self._kernel_dtype)

        raise NotImplementedError(f"unsupported builtin: {builtin}")

    @staticmethod
    def _resolve_builtin(func_node) -> Optional[str]:
        """Return the builtin name if func_node is tt.<builtin>, else None."""
        if isinstance(func_node, ast.Attribute):
            if isinstance(func_node.value, ast.Name):
                if func_node.value.id in _MODULE_ALIASES:
                    if func_node.attr in _BUILTINS:
                        return func_node.attr
        return None

    def _get_kwarg(self, node: ast.Call, name: str):
        for kw in node.keywords:
            if kw.arg == name:
                return self._eval(kw.value)
        return None


class JITFunction:
    """Wraps a Python function for JIT compilation to GPU code."""

    def __init__(self, fn):
        self.fn = fn
        self.source = textwrap.dedent(inspect.getsource(fn))
        self.tree = ast.parse(self.source)
        self._cache: dict[tuple, Any] = {}
        self._constexpr_params: set[str] = self._find_constexpr_params()

    def _find_constexpr_params(self) -> set[str]:
        """Return the set of parameter names annotated with tt.constexpr."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                return {
                    a.arg for a in node.args.args
                    if a.annotation is not None
                    and _is_constexpr_annotation(a.annotation)
                }
        return set()

    def __getitem__(self, grid: tuple):
        """Enable kernel[grid](...) launch syntax."""
        def launcher(*args, **kwargs):
            key = self._make_key(args)
            if key not in self._cache:
                self._cache[key] = self._compile(args)
            self._launch(self._cache[key], grid, args)
        return launcher

    def _make_key(self, args: tuple) -> tuple:
        func_def = self._get_func_def()
        param_names = [a.arg for a in func_def.args.args]
        key = []
        for name, a in zip(param_names, args):
            if name in self._constexpr_params:
                # Constexpr value is baked into compiled code — include in key.
                key.append(("constexpr", int(a)))
            else:
                key.append((type(a).__name__,
                             getattr(a, "shape", None),
                             getattr(a, "dtype", None)))
        return tuple(key)

    def _get_func_def(self) -> ast.FunctionDef:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                return node
        raise AssertionError("no function found in source")

    def _compile(self, args: tuple):
        import _tiny_ton_core as core

        func_def = self._get_func_def()
        param_names = [a.arg for a in func_def.args.args]

        # Separate constexpr params (compile-time) from runtime args.
        constexpr_values: dict[str, int] = {}
        runtime_args: list[Any] = []
        runtime_names: list[str] = []
        for name, a in zip(param_names, args):
            if name in self._constexpr_params:
                constexpr_values[name] = int(a)
            else:
                runtime_args.append(a)
                runtime_names.append(name)

        arg_is_pointer = [isinstance(a, np.ndarray) for a in runtime_args]
        arg_dtypes = []
        for a in runtime_args:
            if isinstance(a, np.ndarray) and a.dtype in _DTYPE_MAP:
                arg_dtypes.append(_DTYPE_MAP[a.dtype])
            else:
                arg_dtypes.append("i32")

        kernel_dtype = "i32"
        for d in arg_dtypes:
            if d in ("f32", "f16"):
                kernel_dtype = d
                break

        use_cuda = core.has_cuda()

        builder = core.IRBuilder()
        builder.begin_function(func_def.name)

        visitor = KernelVisitor(builder, param_names, arg_is_pointer,
                                arg_dtypes, constexpr_values)
        visitor.visit(func_def)

        block_size = visitor.block_size or 1

        if use_cuda:
            sm = os.environ.get("TTN_SM_VERSION", "sm_87")
            nvptx_result = builder.compile_to_nvptx(sm_version=sm, block_size=block_size)
            assert nvptx_result.success, \
                f"NVPTX compilation failed: {nvptx_result.error}"
            return {
                "backend": "cuda",
                "ptx": nvptx_result.ptx,
                "kernel_name": nvptx_result.kernel_name,
                "block_size": block_size,
                "kernel_dtype": kernel_dtype,
                "constexpr_params": self._constexpr_params,
                "param_names": param_names,
            }

        result = builder.compile()
        assert result.success, f"compilation failed: {result.error}"
        return {
            "backend": "simulator",
            "binary": result.get_binary(),
            "block_size": block_size,
            "kernel_dtype": kernel_dtype,
            "constexpr_params": self._constexpr_params,
            "param_names": param_names,
        }

    def _launch(self, compiled, grid: tuple, args: tuple):
        if compiled["backend"] == "cuda":
            self._launch_cuda(compiled, grid, args)
        else:
            self._launch_simulator(compiled, grid, args)

    def _runtime_args(self, compiled, args: tuple):
        """Return only the non-constexpr args (those passed to the kernel)."""
        constexpr_params = compiled["constexpr_params"]
        param_names = compiled["param_names"]
        return [a for name, a in zip(param_names, args)
                if name not in constexpr_params]

    def _launch_cuda(self, compiled, grid: tuple, args: tuple):
        import _tiny_ton_core as core

        rt = core.CUDARuntime()
        block_size = compiled["block_size"]
        num_blocks = grid[0] if grid else 1

        dev_ptrs = []
        array_mappings = []

        kernel_args = []
        for a in self._runtime_args(compiled, args):
            if isinstance(a, np.ndarray):
                flat = np.ascontiguousarray(a.flatten())
                nbytes = flat.nbytes
                dptr = rt.alloc(nbytes)
                rt.copy_to_device(dptr, flat)
                dev_ptrs.append(dptr)
                array_mappings.append((dptr, nbytes, a))
                kernel_args.append(dptr)
            else:
                kernel_args.append(int(a))

        try:
            rt.launch(compiled["ptx"], compiled["kernel_name"],
                      num_blocks, block_size, kernel_args)

            for dptr, nbytes, arr in array_mappings:
                flat = np.empty(arr.size, dtype=arr.dtype)
                rt.copy_from_device(flat, dptr, nbytes)
                arr.flat[:] = flat
        finally:
            for dptr in dev_ptrs:
                rt.free(dptr)

    def _launch_simulator(self, compiled, grid: tuple, args: tuple):
        import _tiny_ton_core as core

        binary = compiled["binary"]
        block_size = compiled["block_size"]
        num_blocks = grid[0] if grid else 1
        kernel_dtype = compiled.get("kernel_dtype", "i32")

        runtime_args = self._runtime_args(compiled, args)

        total_elements = 0
        for a in runtime_args:
            if isinstance(a, np.ndarray):
                total_elements += a.size

        mem_words = max(total_elements + 1024, 4096)
        sim = core.SimulatedGPU(mem_words)

        kernel_args = []
        next_addr = 0
        array_mappings = []

        for a in runtime_args:
            if isinstance(a, np.ndarray):
                flat = a.flatten()
                if flat.dtype == np.float32:
                    data = flat.view(np.int32).tolist()
                elif flat.dtype == np.float16:
                    # Each f16 value stored as its 16-bit pattern zero-extended
                    # to 32 bits (one value per memory word).
                    data = flat.view(np.uint16).astype(np.int32).tolist()
                else:
                    data = flat.astype(np.int32).tolist()
                sim.write_memory(next_addr, data)
                array_mappings.append((next_addr, len(data), a))
                kernel_args.append(next_addr)
                next_addr += len(data)
            else:
                kernel_args.append(int(a))

        sim.set_args(kernel_args)

        prog = [int(x) for x in binary]
        sim.load_program(prog)
        sim.run(num_blocks, block_size)

        for base_addr, count, arr in array_mappings:
            result = sim.read_memory(base_addr, count)
            if arr.dtype == np.float32:
                arr.flat[:] = np.array(result, dtype=np.int32).view(np.float32)
            elif arr.dtype == np.float16:
                # Extract low 16 bits and reinterpret as f16.
                raw = np.array(result, dtype=np.uint32).astype(np.uint16)
                arr.flat[:] = raw.view(np.float16)
            else:
                arr.flat[:] = result


def jit(fn):
    """Decorator that marks a function for JIT compilation to GPU code.

    Usage::

        @tt.jit
        def vector_add(a_ptr, b_ptr, c_ptr, N):
            ...

        vector_add[(grid,)](a, b, c, N)
    """
    return JITFunction(fn)
