#pragma once

#include "tiny-ton/Compiler/RegisterAlloc.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tinyton {

struct Instruction {
  uint16_t encoding;
  std::string assembly;
};

std::vector<Instruction> emit(mlir::ModuleOp module, const RegisterMap &regMap);

} // namespace tinyton
