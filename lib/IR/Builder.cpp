#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonDialect.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace tinyton {

struct IRBuilder::Impl {
  mlir::MLIRContext context;
  mlir::ModuleOp module;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::Block *block = nullptr;

  Impl() {
    context.getOrLoadDialect<tinyton::TinyTonDialect>();
    module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    builder = std::make_unique<mlir::OpBuilder>(&context);
  }
};

IRBuilder::IRBuilder() : impl_(std::make_unique<Impl>()) {}
IRBuilder::~IRBuilder() = default;

void IRBuilder::beginFunction(const std::string &name) {
  impl_->block = impl_->module.getBody();
  impl_->builder->setInsertionPointToEnd(impl_->block);
}

mlir::Value IRBuilder::emitConst(int64_t val) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto i32Ty = impl_->builder->getI32Type();
  auto attr = impl_->builder->getI32IntegerAttr(val);
  return impl_->builder->create<tinyton::ConstOp>(loc, i32Ty, attr)
      .getResult();
}

mlir::Value IRBuilder::emitFConst(double val) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto f32Ty = impl_->builder->getF32Type();
  auto attr = impl_->builder->getF32FloatAttr(static_cast<float>(val));
  return impl_->builder->create<tinyton::FConstOp>(loc, f32Ty, attr)
      .getResult();
}

mlir::Value IRBuilder::emitArg(int64_t index, bool isPointer, bool isFloat) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  // Pointer args are always i32 (addresses). Only non-pointer scalar args
  // use f32 result type. The is_float flag on pointer args indicates the
  // pointed-to element type, used by the GPU lowering for GEP/load/store.
  mlir::Type resTy = (!isPointer && isFloat)
                         ? (mlir::Type)impl_->builder->getF32Type()
                         : (mlir::Type)impl_->builder->getI32Type();
  auto indexAttr = impl_->builder->getI32IntegerAttr(index);
  auto ptrAttr = impl_->builder->getBoolAttr(isPointer);
  auto floatAttr = impl_->builder->getBoolAttr(isFloat);
  return impl_->builder
      ->create<tinyton::ArgOp>(loc, resTy, indexAttr, ptrAttr, floatAttr)
      .getResult();
}

mlir::Value IRBuilder::emitProgramId(int64_t axis) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto i32Ty = impl_->builder->getI32Type();
  auto attr = impl_->builder->getI32IntegerAttr(axis);
  return impl_->builder->create<tinyton::ProgramIdOp>(loc, i32Ty, attr)
      .getResult();
}

mlir::Value IRBuilder::emitThreadId(int64_t axis) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto i32Ty = impl_->builder->getI32Type();
  auto attr = impl_->builder->getI32IntegerAttr(axis);
  return impl_->builder->create<tinyton::ThreadIdOp>(loc, i32Ty, attr)
      .getResult();
}

mlir::Value IRBuilder::emitAdd(mlir::Value lhs, mlir::Value rhs) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = lhs.getType();
  return impl_->builder->create<tinyton::AddOp>(loc, ty, lhs, rhs)
      .getResult();
}

mlir::Value IRBuilder::emitSub(mlir::Value lhs, mlir::Value rhs) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = lhs.getType();
  return impl_->builder->create<tinyton::SubOp>(loc, ty, lhs, rhs)
      .getResult();
}

mlir::Value IRBuilder::emitMul(mlir::Value lhs, mlir::Value rhs) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = lhs.getType();
  return impl_->builder->create<tinyton::MulOp>(loc, ty, lhs, rhs)
      .getResult();
}

mlir::Value IRBuilder::emitDiv(mlir::Value lhs, mlir::Value rhs) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = lhs.getType();
  return impl_->builder->create<tinyton::DivOp>(loc, ty, lhs, rhs)
      .getResult();
}

mlir::Value IRBuilder::emitCmpLt(mlir::Value lhs, mlir::Value rhs) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto i32Ty = impl_->builder->getI32Type();
  return impl_->builder->create<tinyton::CmpLtOp>(loc, i32Ty, lhs, rhs)
      .getResult();
}

mlir::Value IRBuilder::emitLoad(mlir::Value addr, mlir::Value mask,
                                bool isFloat) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  mlir::Type resTy = isFloat ? (mlir::Type)impl_->builder->getF32Type()
                             : (mlir::Type)impl_->builder->getI32Type();
  return impl_->builder->create<tinyton::LoadOp>(loc, resTy, addr, mask)
      .getResult();
}

void IRBuilder::emitStore(mlir::Value addr, mlir::Value val, mlir::Value mask) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  impl_->builder->create<tinyton::StoreOp>(loc, addr, val, mask);
}

void IRBuilder::emitBranchZero(mlir::Value cond, int64_t skip) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto attr = impl_->builder->getI32IntegerAttr(skip);
  impl_->builder->create<tinyton::BranchZeroOp>(loc, cond, attr);
}

void IRBuilder::emitRet() {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  impl_->builder->create<tinyton::RetOp>(loc);
}

mlir::ModuleOp IRBuilder::getModule() { return impl_->module; }

} // namespace tinyton
