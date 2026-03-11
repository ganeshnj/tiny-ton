#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/IR/ElementType.h"
#include "tiny-ton/Compiler/Pipeline.h"
#include "tiny-ton/Runtime/Simulator.h"

#include "llvm/Support/raw_ostream.h"

#ifdef TTN_ENABLE_CUDA
#include "tiny-ton/Runtime/CUDARuntime.h"
#endif

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
      .def("emit_fconst",
           [](tinyton::IRBuilder &self, double val) {
             return PyValue{self.emitFConst(val)};
           })
      .def("emit_hconst",
           [](tinyton::IRBuilder &self, double val) {
             return PyValue{self.emitHConst(val)};
           })
      .def("emit_arg",
           [](tinyton::IRBuilder &self, int64_t index, bool isPointer,
              const std::string &dtype) {
             auto et = tinyton::elementTypeFromString(dtype);
             return PyValue{self.emitArg(index, isPointer, et)};
           },
           py::arg("index"), py::arg("is_pointer") = false,
           py::arg("dtype") = "i32")
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
      .def("emit_div",
           [](tinyton::IRBuilder &self, PyValue lhs, PyValue rhs) {
             return PyValue{self.emitDiv(lhs.val, rhs.val)};
           })
      .def("emit_cmp_lt",
           [](tinyton::IRBuilder &self, PyValue lhs, PyValue rhs) {
             return PyValue{self.emitCmpLt(lhs.val, rhs.val)};
           })
      .def("emit_load",
           [](tinyton::IRBuilder &self, PyValue addr, py::object mask,
              const std::string &dtype) {
             mlir::Value maskVal;
             if (!mask.is_none())
               maskVal = mask.cast<PyValue>().val;
             auto et = tinyton::elementTypeFromString(dtype);
             return PyValue{self.emitLoad(addr.val, maskVal, et)};
           },
           py::arg("addr"), py::arg("mask") = py::none(),
           py::arg("dtype") = "i32")
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
      .def("dump_mlir",
           [](tinyton::IRBuilder &self) {
             std::string out;
             llvm::raw_string_ostream os(out);
             self.getModule().print(os);
             return os.str();
           })
      .def("compile",
           [](tinyton::IRBuilder &self) {
             tinyton::CompileOptions opts;
             opts.emitMode = tinyton::CompileOptions::EmitMode::Asm;
             return tinyton::compileModule(self.getModule(), opts);
           })
      .def("compile_to_nvptx",
           [](tinyton::IRBuilder &self, const std::string &sm) {
             return tinyton::compileToNVPTX(self.getModule(), sm);
           },
           py::arg("sm_version") = "sm_87");

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

  py::class_<tinyton::NVPTXCompileResult>(m, "NVPTXCompileResult")
      .def_readonly("success", &tinyton::NVPTXCompileResult::success)
      .def_readonly("ptx", &tinyton::NVPTXCompileResult::ptx)
      .def_readonly("kernel_name", &tinyton::NVPTXCompileResult::kernelName)
      .def_readonly("error", &tinyton::NVPTXCompileResult::error);

  py::class_<tinyton::SimulatedGPU>(m, "SimulatedGPU")
      .def(py::init<int>(), py::arg("mem_words") = 4096)
      .def("load_program", &tinyton::SimulatedGPU::loadProgram)
      .def("set_args", &tinyton::SimulatedGPU::setArgs)
      .def("write_memory", &tinyton::SimulatedGPU::writeMemory)
      .def("read_memory", &tinyton::SimulatedGPU::readMemory)
      .def("run", &tinyton::SimulatedGPU::run);

#ifdef TTN_ENABLE_CUDA
  m.def("has_cuda", [] { return tinyton::CUDARuntime::isAvailable(); });

  py::class_<tinyton::CUDARuntime>(m, "CUDARuntime")
      .def(py::init<>())
      .def("alloc",
           [](tinyton::CUDARuntime &self, size_t bytes) {
             return reinterpret_cast<uintptr_t>(self.alloc(bytes));
           })
      .def("free",
           [](tinyton::CUDARuntime &self, uintptr_t ptr) {
             self.free(reinterpret_cast<void *>(ptr));
           })
      .def("copy_to_device",
           [](tinyton::CUDARuntime &self, uintptr_t dst, py::buffer src) {
             auto info = src.request();
             self.copyToDevice(reinterpret_cast<void *>(dst), info.ptr,
                               static_cast<size_t>(info.size * info.itemsize));
           })
      .def("copy_from_device",
           [](tinyton::CUDARuntime &self, py::buffer dst, uintptr_t src,
              size_t bytes) {
             auto info = dst.request(true);
             self.copyFromDevice(info.ptr, reinterpret_cast<void *>(src),
                                 bytes);
           })
      .def("launch",
           [](tinyton::CUDARuntime &self, const std::string &ptx,
              const std::string &kernelName, int gridX, int blockX,
              const std::vector<uintptr_t> &args) {
             std::vector<void *> voidArgs;
             voidArgs.reserve(args.size());
             for (auto a : args)
               voidArgs.push_back(reinterpret_cast<void *>(a));
             self.launch(ptx, kernelName, gridX, blockX, voidArgs);
           });
#else
  m.def("has_cuda", [] { return false; });
#endif
}
