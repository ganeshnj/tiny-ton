#pragma once

#include <string>

namespace tinyton {

struct CompileOptions {
  enum class EmitMode { MLIR, Asm, Hex, Bin, Trace };
  EmitMode emitMode = EmitMode::Asm;
};

struct CompileResult {
  bool success = false;
  std::string output;
  std::string error;
};

CompileResult compile(const std::string &source, const CompileOptions &opts);

} // namespace tinyton
