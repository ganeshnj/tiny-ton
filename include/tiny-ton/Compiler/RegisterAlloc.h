#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace tinyton {

using RegisterMap = llvm::DenseMap<mlir::Value, int>;

RegisterMap allocateRegisters(mlir::ModuleOp module);

} // namespace tinyton
