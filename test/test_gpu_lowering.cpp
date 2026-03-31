#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/IR/ElementType.h"
#include "tiny-ton/Conversion/TinyTonToGPU.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdio>
#include <string>

static std::string lowerAndDump(tinyton::IRBuilder &builder) {
  auto module = builder.getModule();
  auto result = tinyton::lowerToGPU(module);
  assert(result.success && "lowerToGPU failed");

  std::string dump;
  llvm::raw_string_ostream os(dump);
  module.print(os);
  return dump;
}

static bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

struct GPULoweringTest {
  const char *name;
  bool (*fn)();
};

static bool testArithI32() {
  // tinyton.add (i32) -> arith.addi
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitConst(1);
  auto c = b.emitConst(2);
  b.emitAdd(a, c);
  b.emitRet();

  auto dump = lowerAndDump(b);
  if (!contains(dump, "arith.addi")) {
    std::printf("    missing arith.addi in:\n%s\n", dump.c_str());
    return false;
  }
  return true;
}

static bool testArithF32() {
  // tinyton.add (f32) -> arith.addf
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitFConst(1.0);
  auto c = b.emitFConst(2.0);
  b.emitAdd(a, c);
  b.emitRet();

  auto dump = lowerAndDump(b);
  if (!contains(dump, "arith.addf")) {
    std::printf("    missing arith.addf in:\n%s\n", dump.c_str());
    return false;
  }
  return true;
}

static bool testPointerArith() {
  // tinyton.add on a pointer arg -> llvm.getelementptr
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto ptr = b.emitArg(0, true, tinyton::ElementType::F32);
  auto off = b.emitConst(4);
  b.emitAdd(ptr, off);
  b.emitRet();

  auto dump = lowerAndDump(b);
  if (!contains(dump, "llvm.getelementptr")) {
    std::printf("    missing llvm.getelementptr in:\n%s\n", dump.c_str());
    return false;
  }
  return true;
}

static bool testProgramId() {
  // tinyton.program_id -> gpu.block_id
  tinyton::IRBuilder b;
  b.beginFunction("test");
  b.emitProgramId(0);
  b.emitRet();

  auto dump = lowerAndDump(b);
  if (!contains(dump, "gpu.block_id")) {
    std::printf("    missing gpu.block_id in:\n%s\n", dump.c_str());
    return false;
  }
  return true;
}

static bool testThreadId() {
  // tinyton.thread_id -> gpu.thread_id
  tinyton::IRBuilder b;
  b.beginFunction("test");
  b.emitThreadId(0);
  b.emitRet();

  auto dump = lowerAndDump(b);
  if (!contains(dump, "gpu.thread_id")) {
    std::printf("    missing gpu.thread_id in:\n%s\n", dump.c_str());
    return false;
  }
  return true;
}

static bool testMathExp() {
  // tinyton.exp -> math.exp
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(1.0);
  b.emitExp(x);
  b.emitRet();

  auto dump = lowerAndDump(b);
  if (!contains(dump, "math.exp")) {
    std::printf("    missing math.exp in:\n%s\n", dump.c_str());
    return false;
  }
  return true;
}

static bool testMaskedLoad() {
  // tinyton.load with mask -> cf.cond_br + llvm.load
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto ptr = b.emitArg(0, true, tinyton::ElementType::I32);
  auto mask = b.emitConst(1);
  b.emitLoad(ptr, mask);
  b.emitRet();

  auto dump = lowerAndDump(b);
  bool hasCond = contains(dump, "cf.cond_br");
  bool hasLoad = contains(dump, "llvm.load");
  if (!hasCond || !hasLoad) {
    std::printf("    missing cf.cond_br (%d) or llvm.load (%d) in:\n%s\n",
                hasCond, hasLoad, dump.c_str());
    return false;
  }
  return true;
}

static bool testReduceSum() {
  // tinyton.reduce_sum -> gpu.all_reduce
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(1.0);
  b.emitReduceSum(x);
  b.emitRet();

  auto dump = lowerAndDump(b);
  if (!contains(dump, "gpu.all_reduce")) {
    std::printf("    missing gpu.all_reduce in:\n%s\n", dump.c_str());
    return false;
  }
  if (!contains(dump, "add")) {
    std::printf("    missing 'add' op attr in:\n%s\n", dump.c_str());
    return false;
  }
  return true;
}

static bool testReduceMax() {
  // tinyton.reduce_max -> gpu.all_reduce
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(1.0);
  b.emitReduceMax(x);
  b.emitRet();

  auto dump = lowerAndDump(b);
  if (!contains(dump, "gpu.all_reduce")) {
    std::printf("    missing gpu.all_reduce in:\n%s\n", dump.c_str());
    return false;
  }
  if (!contains(dump, "maxnumf")) {
    std::printf("    missing 'maxnumf' op attr in:\n%s\n", dump.c_str());
    return false;
  }
  return true;
}

static bool testGPUModuleStructure() {
  // Output should have gpu.module containing gpu.func with gpu.kernel attr
  tinyton::IRBuilder b;
  b.beginFunction("test");
  b.emitConst(1);
  b.emitRet();

  auto dump = lowerAndDump(b);
  bool hasModule = contains(dump, "gpu.module");
  bool hasFunc = contains(dump, "gpu.func");
  bool hasKernel = contains(dump, "kernel");
  if (!hasModule || !hasFunc || !hasKernel) {
    std::printf("    missing structure: gpu.module=%d gpu.func=%d kernel=%d\n",
                hasModule, hasFunc, hasKernel);
    return false;
  }
  return true;
}

static bool testLoweringSuccess() {
  // lowerToGPU returns success == true
  tinyton::IRBuilder b;
  b.beginFunction("test");
  b.emitConst(1);
  b.emitRet();

  auto module = b.getModule();
  auto result = tinyton::lowerToGPU(module);
  if (!result.success) {
    std::printf("    lowerToGPU returned success=false: %s\n",
                result.error.c_str());
    return false;
  }
  return true;
}

int main() {
  GPULoweringTest tests[] = {
    {"arith_i32_addi", testArithI32},
    {"arith_f32_addf", testArithF32},
    {"pointer_arith_gep", testPointerArith},
    {"program_id_block_id", testProgramId},
    {"thread_id_thread_id", testThreadId},
    {"math_exp", testMathExp},
    {"masked_load", testMaskedLoad},
    {"reduce_sum_all_reduce", testReduceSum},
    {"reduce_max_all_reduce", testReduceMax},
    {"gpu_module_structure", testGPUModuleStructure},
    {"lowering_success", testLoweringSuccess},
  };

  int passed = 0, failed = 0;
  for (auto &t : tests) {
    bool ok = t.fn();
    std::printf("  %s: %s\n", t.name, ok ? "PASS" : "FAIL");
    if (ok) ++passed; else ++failed;
  }
  std::printf("\n%d/%d passed\n", passed, passed + failed);
  return failed ? 1 : 0;
}
