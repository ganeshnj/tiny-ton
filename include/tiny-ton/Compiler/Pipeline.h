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

} // namespace tinyton
