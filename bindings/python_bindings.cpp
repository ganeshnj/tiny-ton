#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/Compiler/Pipeline.h"
#include "tiny-ton/Runtime/Simulator.h"

namespace py = pybind11;

namespace {

struct PyValue {
  mlir::Value val;
};

} // namespace

PYBIND11_MODULE(_tiny_ton_core, m) {
  m.doc() = "tiny-ton C++ core: IR builder, compiler, and runtime";

  py::class_<PyValue>(m, "Value");

  py::class_<tinyton::IRBuilder>(m, "IRBuilder")
      .def(py::init<>())
      .def("begin_function",
           [](tinyton::IRBuilder &self, const std::string &name) {
             self.beginFunction(name);
           })
      .def("emit_const",
           [](tinyton::IRBuilder &self, int64_t val) {
             return PyValue{self.emitConst(val)};
           })
      .def("emit_arg",
           [](tinyton::IRBuilder &self, int64_t index) {
             return PyValue{self.emitArg(index)};
           })
      .def("emit_program_id",
           [](tinyton::IRBuilder &self, int64_t axis) {
             return PyValue{self.emitProgramId(axis)};
           })
      .def("emit_thread_id",
           [](tinyton::IRBuilder &self, int64_t axis) {
             return PyValue{self.emitThreadId(axis)};
           })
      .def("emit_add",
           [](tinyton::IRBuilder &self, PyValue lhs, PyValue rhs) {
             return PyValue{self.emitAdd(lhs.val, rhs.val)};
           })
      .def("emit_sub",
           [](tinyton::IRBuilder &self, PyValue lhs, PyValue rhs) {
             return PyValue{self.emitSub(lhs.val, rhs.val)};
           })
      .def("emit_mul",
           [](tinyton::IRBuilder &self, PyValue lhs, PyValue rhs) {
             return PyValue{self.emitMul(lhs.val, rhs.val)};
           })
      .def("emit_cmp_lt",
           [](tinyton::IRBuilder &self, PyValue lhs, PyValue rhs) {
             return PyValue{self.emitCmpLt(lhs.val, rhs.val)};
           })
      .def("emit_load",
           [](tinyton::IRBuilder &self, PyValue addr, py::object mask) {
             mlir::Value maskVal;
             if (!mask.is_none())
               maskVal = mask.cast<PyValue>().val;
             return PyValue{self.emitLoad(addr.val, maskVal)};
           },
           py::arg("addr"), py::arg("mask") = py::none())
      .def("emit_store",
           [](tinyton::IRBuilder &self, PyValue addr, PyValue val,
              py::object mask) {
             mlir::Value maskVal;
             if (!mask.is_none())
               maskVal = mask.cast<PyValue>().val;
             self.emitStore(addr.val, val.val, maskVal);
           },
           py::arg("addr"), py::arg("val"), py::arg("mask") = py::none())
      .def("emit_branch_zero",
           [](tinyton::IRBuilder &self, PyValue cond, int64_t skip) {
             self.emitBranchZero(cond.val, skip);
           })
      .def("emit_ret", [](tinyton::IRBuilder &self) { self.emitRet(); })
      .def("compile",
           [](tinyton::IRBuilder &self) {
             tinyton::CompileOptions opts;
             opts.emitMode = tinyton::CompileOptions::EmitMode::Asm;
             return tinyton::compileModule(self.getModule(), opts);
           });

  py::class_<tinyton::CompileResult>(m, "CompileResult")
      .def_readonly("success", &tinyton::CompileResult::success)
      .def_readonly("output", &tinyton::CompileResult::output)
      .def_readonly("error", &tinyton::CompileResult::error)
      .def("get_binary", [](const tinyton::CompileResult &self) {
        std::vector<int> binary;
        binary.reserve(self.instructions.size());
        for (auto &inst : self.instructions) {
          binary.push_back(inst.encoding);
        }
        return binary;
      });

  py::class_<tinyton::SimulatedGPU>(m, "SimulatedGPU")
      .def(py::init<int>(), py::arg("mem_words") = 4096)
      .def("load_program", &tinyton::SimulatedGPU::loadProgram)
      .def("set_args", &tinyton::SimulatedGPU::setArgs)
      .def("write_memory", &tinyton::SimulatedGPU::writeMemory)
      .def("read_memory", &tinyton::SimulatedGPU::readMemory)
      .def("run", &tinyton::SimulatedGPU::run);
}
