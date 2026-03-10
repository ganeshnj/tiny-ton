#include "tiny-ton/Compiler/RegisterAlloc.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonOps.h"

#include "llvm/ADT/SmallVector.h"

namespace tinyton {

RegisterMap allocateRegisters(mlir::ModuleOp module) {
  RegisterMap regMap;

  // Find last use of each value
  llvm::DenseMap<mlir::Value, mlir::Operation *> lastUse;
  for (auto &op : module.getBody()->getOperations()) {
    for (mlir::Value operand : op.getOperands()) {
      lastUse[operand] = &op;
    }
  }

  // Linear scan: assign registers R0-R12
  constexpr int kMaxRegs = 13;
  llvm::SmallVector<bool, 13> inUse(kMaxRegs, false);

  auto allocReg = [&]() -> int {
    for (int i = 0; i < kMaxRegs; ++i) {
      if (!inUse[i]) {
        inUse[i] = true;
        return i;
      }
    }
    llvm::report_fatal_error("register allocator: out of registers");
    return -1;
  };

  for (auto &op : module.getBody()->getOperations()) {
    // Allocate destination register for ops that produce a result
    if (op.getNumResults() == 1) {
      mlir::Value result = op.getResult(0);
      int reg = allocReg();
      regMap[result] = reg;
    }

    // Free registers whose last use is this operation
    for (mlir::Value operand : op.getOperands()) {
      if (lastUse[operand] == &op) {
        auto it = regMap.find(operand);
        if (it != regMap.end()) {
          inUse[it->second] = false;
        }
      }
    }
  }

  return regMap;
}

} // namespace tinyton
