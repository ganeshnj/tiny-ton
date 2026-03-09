"""JIT decorator and Python AST → C++ IR bridge."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any


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

    def _compile(self, args: tuple) -> Any:
        """Walk Python AST and emit IR via C++ builder."""
        # TODO: implement — use _tiny_ton_core.IRBuilder
        raise NotImplementedError("JIT compilation not implemented yet")

    def _launch(self, compiled: Any, grid: tuple, args: tuple) -> None:
        """Execute a compiled kernel with the given grid and arguments."""
        # TODO: implement — call _tiny_ton_core.Runtime.launch
        raise NotImplementedError("Kernel launch not implemented yet")


class KernelVisitor(ast.NodeVisitor):
    """Walks a Python AST and emits TinyTon IR via the C++ IRBuilder."""

    def __init__(self, builder, args):
        self.builder = builder
        self.symbols: dict[str, Any] = {}
        self.args = args

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # TODO: implement
        pass

    def visit_Assign(self, node: ast.Assign):
        # TODO: implement
        pass

    def visit_Expr(self, node: ast.Expr):
        # TODO: implement
        pass

    def visit_For(self, node: ast.For):
        # TODO: implement
        pass

    def visit_If(self, node: ast.If):
        # TODO: implement
        pass

    def visit_Call(self, node: ast.Call):
        # TODO: implement
        pass

    def visit_BinOp(self, node: ast.BinOp):
        # TODO: implement
        pass

    def visit_Compare(self, node: ast.Compare):
        # TODO: implement
        pass


def jit(fn):
    """Decorator that marks a function for JIT compilation to GPU code.

    Usage::

        @tt.jit
        def vector_add(a_ptr, b_ptr, c_ptr, N):
            ...

        vector_add[(grid,)](a, b, c, N)
    """
    return JITFunction(fn)
