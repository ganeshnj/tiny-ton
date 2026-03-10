#pragma once

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace tinyton {

struct GPULoweringResult {
  bool success = false;
  std::string error;
  std::string kernelName;
};

GPULoweringResult lowerToGPU(mlir::ModuleOp module);

} // namespace tinyton
