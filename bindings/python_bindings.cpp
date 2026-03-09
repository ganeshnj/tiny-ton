#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/Compiler/Pipeline.h"
#include "tiny-ton/Runtime/Runtime.h"

namespace py = pybind11;

PYBIND11_MODULE(_tiny_ton_core, m) {
  m.doc() = "tiny-ton C++ core: IR builder, compiler, and runtime";

  // --- IRBuilder -----------------------------------------------------------
  py::class_<tinyton::Value>(m, "Value");

  py::class_<tinyton::IRBuilder>(m, "IRBuilder")
      .def(py::init<>())
      .def("begin_function", &tinyton::IRBuilder::beginFunction)
      .def("get_arg", &tinyton::IRBuilder::getArg,
           py::return_value_policy::reference)
      .def("emit_const", &tinyton::IRBuilder::emitConst,
           py::return_value_policy::reference)
      .def("emit_add", &tinyton::IRBuilder::emitAdd,
           py::return_value_policy::reference)
      .def("emit_sub", &tinyton::IRBuilder::emitSub,
           py::return_value_policy::reference)
      .def("emit_mul", &tinyton::IRBuilder::emitMul,
           py::return_value_policy::reference)
      .def("emit_div", &tinyton::IRBuilder::emitDiv,
           py::return_value_policy::reference)
      .def("emit_load", &tinyton::IRBuilder::emitLoad,
           py::return_value_policy::reference)
      .def("emit_store", &tinyton::IRBuilder::emitStore)
      .def("emit_program_id", &tinyton::IRBuilder::emitProgramId,
           py::return_value_policy::reference)
      .def("emit_ret", &tinyton::IRBuilder::emitRet);

  // --- Compiler ------------------------------------------------------------
  py::class_<tinyton::CompileResult>(m, "CompileResult")
      .def_readonly("success", &tinyton::CompileResult::success)
      .def_readonly("output", &tinyton::CompileResult::output)
      .def_readonly("error", &tinyton::CompileResult::error);

  m.def("compile", &tinyton::compile, py::arg("source"), py::arg("opts"));

  // --- Runtime -------------------------------------------------------------
  py::class_<tinyton::CompiledKernel>(m, "CompiledKernel")
      .def(py::init<>())
      .def_readwrite("binary", &tinyton::CompiledKernel::binary);

  py::class_<tinyton::LaunchParams>(m, "LaunchParams")
      .def(py::init<>())
      .def_readwrite("grid_x", &tinyton::LaunchParams::gridX)
      .def_readwrite("grid_y", &tinyton::LaunchParams::gridY)
      .def_readwrite("grid_z", &tinyton::LaunchParams::gridZ);

  py::class_<tinyton::Runtime>(m, "Runtime")
      .def(py::init<>());
}
