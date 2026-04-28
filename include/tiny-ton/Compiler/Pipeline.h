#pragma once

#include "tiny-ton/Compiler/CodeGen.h"
#include "tiny-ton/Compiler/RegisterAlloc.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>
#include <vector>

namespace tinyton {

struct CompileOptions {
  enum class EmitMode { MLIR, Asm, Hex, Bin };
  EmitMode emitMode = EmitMode::Asm;
};

struct CompileResult {
  bool success = false;
  std::string output;
  std::string error;
  std::vector<Instruction> instructions;
};

CompileResult compileModule(mlir::ModuleOp module, const CompileOptions &opts);

struct NVPTXCompileResult {
  bool success = false;
  std::string ptx;
  std::string kernelName;
  std::string error;
};

NVPTXCompileResult compileToNVPTX(mlir::ModuleOp module,
                                  llvm::StringRef smVersion = "sm_87",
                                  int blockSize = 32);

} // namespace tinyton
