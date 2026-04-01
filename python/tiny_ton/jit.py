"""JIT decorator and Python AST -> C++ IR bridge."""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from typing import Any, Optional

import numpy as np


_BUILTINS = {"program_id", "arange", "load", "store",
             "exp", "log", "sqrt", "rsqrt", "abs", "max",
             "reduce_sum", "reduce_max"}

# Module aliases that should be treated as the tiny_ton namespace.
_MODULE_ALIASES = {"tt", "tiny_ton"}

# Map numpy dtypes to tiny-ton dtype strings.
_DTYPE_MAP = {
    np.dtype("int32"): "i32",
    np.dtype("float32"): "f32",
    np.dtype("float16"): "f16",
}


class KernelVisitor(ast.NodeVisitor):
    """Walks a Python AST and emits TinyTon IR via the C++ IRBuilder."""

    def __init__(self, builder, param_names: list[str],
                 arg_is_pointer: Optional[list[bool]] = None,
                 arg_dtypes: Optional[list[str]] = None):
        self.builder = builder
        self.symbols: dict[str, Any] = {}
        self.block_size: Optional[int] = None
        self._arg_dtypes = arg_dtypes or []
        self._kernel_dtype = "i32"
        for d in self._arg_dtypes:
            if d in ("f32", "f16"):
                self._kernel_dtype = d
                break

        for i, name in enumerate(param_names):
            is_ptr = arg_is_pointer[i] if arg_is_pointer else False
            dtype = arg_dtypes[i] if arg_dtypes else "i32"
            self.symbols[name] = builder.emit_arg(i, is_pointer=is_ptr,
                                                  dtype=dtype)

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

    # ------------------------------------------------------------------
    # Expression evaluator (returns a PyValue or None for void ops)
    # ------------------------------------------------------------------

    def _eval(self, node: ast.expr):
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, float):
                if self._kernel_dtype == "f16":
                    return self.builder.emit_hconst(val)
                return self.builder.emit_fconst(val)
            return self.builder.emit_const(int(val))

        if isinstance(node, ast.Name):
            assert node.id in self.symbols, f"undefined: {node.id}"
            return self.symbols[node.id]

        if isinstance(node, ast.BinOp):
            lhs = self._eval(node.left)
            rhs = self._eval(node.right)
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

        if isinstance(node, ast.Call):
            return self._eval_call(node)

        raise NotImplementedError(f"unsupported AST node: {type(node)}")

    # ------------------------------------------------------------------
    # Call dispatch
    # ------------------------------------------------------------------

    def _eval_call(self, node: ast.Call):
        builtin = self._resolve_builtin(node.func)
        if builtin is None:
            raise NotImplementedError(f"unsupported call: {ast.dump(node.func)}")

        if builtin == "program_id":
            axis = node.args[0].value if node.args else 0
            return self.builder.emit_program_id(int(axis))

        if builtin == "arange":
            start = node.args[0].value
            end = node.args[1].value
            self.block_size = int(end) - int(start)
            return self.builder.emit_thread_id(0)

        if builtin == "load":
            addr = self._eval(node.args[0])
            mask = self._get_kwarg(node, "mask")
            return self.builder.emit_load(addr, mask=mask,
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

        if builtin == "reduce_sum":
            return self.builder.emit_reduce_sum(self._eval(node.args[0]))
        if builtin == "reduce_max":
            return self.builder.emit_reduce_max(self._eval(node.args[0]))

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

    def __getitem__(self, grid: tuple):
        """Enable kernel[grid](...) launch syntax."""
        def launcher(*args, **kwargs):
            key = self._make_key(args)
            if key not in self._cache:
                self._cache[key] = self._compile(args)
            self._launch(self._cache[key], grid, args)
        return launcher

    def _make_key(self, args: tuple) -> tuple:
        return tuple(
            (type(a).__name__, getattr(a, "shape", None), getattr(a, "dtype", None))
            for a in args
        )

    def _compile(self, args: tuple):
        import _tiny_ton_core as core

        func_def = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break
        assert func_def is not None, "no function found in source"

        param_names = [a.arg for a in func_def.args.args]
        arg_is_pointer = [isinstance(a, np.ndarray) for a in args]
        arg_dtypes = []
        for a in args:
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
                                arg_dtypes)
        visitor.visit(func_def)

        block_size = visitor.block_size or 1

        if use_cuda:
            sm = os.environ.get("TTN_SM_VERSION", "sm_87")
            nvptx_result = builder.compile_to_nvptx(sm_version=sm)
            assert nvptx_result.success, \
                f"NVPTX compilation failed: {nvptx_result.error}"
            return {
                "backend": "cuda",
                "ptx": nvptx_result.ptx,
                "kernel_name": nvptx_result.kernel_name,
                "block_size": block_size,
                "kernel_dtype": kernel_dtype,
            }

        result = builder.compile()
        assert result.success, f"compilation failed: {result.error}"
        return {
            "backend": "simulator",
            "binary": result.get_binary(),
            "block_size": block_size,
            "kernel_dtype": kernel_dtype,
        }

    def _launch(self, compiled, grid: tuple, args: tuple):
        if compiled["backend"] == "cuda":
            self._launch_cuda(compiled, grid, args)
        else:
            self._launch_simulator(compiled, grid, args)

    def _launch_cuda(self, compiled, grid: tuple, args: tuple):
        import _tiny_ton_core as core

        rt = core.CUDARuntime()
        block_size = compiled["block_size"]
        num_blocks = grid[0] if grid else 1

        dev_ptrs = []
        array_mappings = []

        kernel_args = []
        for a in args:
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

        total_elements = 0
        for a in args:
            if isinstance(a, np.ndarray):
                total_elements += a.size

        mem_words = max(total_elements + 1024, 4096)
        sim = core.SimulatedGPU(mem_words)

        kernel_args = []
        next_addr = 0
        array_mappings = []

        for a in args:
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
