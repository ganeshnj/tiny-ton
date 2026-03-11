#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include <cstdint>
#include <memory>
#include <string>

namespace tinyton {

class IRBuilder {
public:
  IRBuilder();
  ~IRBuilder();

  void beginFunction(const std::string &name);

  mlir::Value emitConst(int64_t val);
  mlir::Value emitFConst(double val);
  mlir::Value emitArg(int64_t index, bool isPointer = false,
                      bool isFloat = false);
  mlir::Value emitProgramId(int64_t axis);
  mlir::Value emitThreadId(int64_t axis);

  mlir::Value emitAdd(mlir::Value lhs, mlir::Value rhs);
  mlir::Value emitSub(mlir::Value lhs, mlir::Value rhs);
  mlir::Value emitMul(mlir::Value lhs, mlir::Value rhs);
  mlir::Value emitDiv(mlir::Value lhs, mlir::Value rhs);
  mlir::Value emitCmpLt(mlir::Value lhs, mlir::Value rhs);

  mlir::Value emitLoad(mlir::Value addr, mlir::Value mask = {},
                       bool isFloat = false);
  void emitStore(mlir::Value addr, mlir::Value val, mlir::Value mask = {});

  void emitBranchZero(mlir::Value cond, int64_t skip);
  void emitRet();

  mlir::ModuleOp getModule();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tinyton
