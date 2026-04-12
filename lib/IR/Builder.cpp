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

mlir::Value IRBuilder::emitHConst(double val) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto f16Ty = impl_->builder->getF16Type();
  // Store as f32 attr; the op result type is f16.
  auto attr = impl_->builder->getF32FloatAttr(static_cast<float>(val));
  return impl_->builder->create<tinyton::HConstOp>(loc, f16Ty, attr)
      .getResult();
}

mlir::Value IRBuilder::emitArg(int64_t index, bool isPointer,
                               ElementType elemType) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  // Pointer args are always i32 (addresses). Only non-pointer scalar args
  // use the actual element type as result. The elem_type attr on pointer args
  // records the pointed-to element type for GPU lowering.
  mlir::Type resTy;
  if (isPointer)
    resTy = impl_->builder->getI32Type();
  else
    resTy = elementTypeToMLIR(elemType, &impl_->context);

  auto indexAttr = impl_->builder->getI32IntegerAttr(index);
  auto ptrAttr = impl_->builder->getBoolAttr(isPointer);
  auto etAttr =
      impl_->builder->getI32IntegerAttr(static_cast<int64_t>(elemType));
  return impl_->builder
      ->create<tinyton::ArgOp>(loc, resTy, indexAttr, ptrAttr, etAttr)
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

mlir::Value IRBuilder::emitExp(mlir::Value operand) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = operand.getType();
  return impl_->builder->create<tinyton::ExpOp>(loc, ty, operand).getResult();
}

mlir::Value IRBuilder::emitLog(mlir::Value operand) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = operand.getType();
  return impl_->builder->create<tinyton::LogOp>(loc, ty, operand).getResult();
}

mlir::Value IRBuilder::emitSqrt(mlir::Value operand) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = operand.getType();
  return impl_->builder->create<tinyton::SqrtOp>(loc, ty, operand).getResult();
}

mlir::Value IRBuilder::emitRsqrt(mlir::Value operand) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = operand.getType();
  return impl_->builder->create<tinyton::RsqrtOp>(loc, ty, operand)
      .getResult();
}

mlir::Value IRBuilder::emitAbs(mlir::Value operand) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = operand.getType();
  return impl_->builder->create<tinyton::AbsOp>(loc, ty, operand).getResult();
}

mlir::Value IRBuilder::emitMax(mlir::Value lhs, mlir::Value rhs) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = lhs.getType();
  return impl_->builder->create<tinyton::MaxOp>(loc, ty, lhs, rhs).getResult();
}

mlir::Value IRBuilder::emitReduceSum(mlir::Value operand) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = operand.getType();
  return impl_->builder->create<tinyton::ReduceSumOp>(loc, ty, operand)
      .getResult();
}

mlir::Value IRBuilder::emitReduceMax(mlir::Value operand) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto ty = operand.getType();
  return impl_->builder->create<tinyton::ReduceMaxOp>(loc, ty, operand)
      .getResult();
}

void IRBuilder::emitSync() {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  impl_->builder->create<tinyton::SyncOp>(loc);
}

void IRBuilder::emitSharedStore(mlir::Value idx, mlir::Value val,
                                int64_t bufferSize) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto sizeAttr = impl_->builder->getI64IntegerAttr(bufferSize);
  impl_->builder->create<tinyton::SharedStoreOp>(loc, idx, val, sizeAttr);
}

mlir::Value IRBuilder::emitSharedLoad(mlir::Value idx, int64_t bufferSize,
                                      ElementType elemType) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  auto sizeAttr = impl_->builder->getI64IntegerAttr(bufferSize);
  mlir::Type resTy = elementTypeToMLIR(elemType, &impl_->context);
  return impl_->builder
      ->create<tinyton::SharedLoadOp>(loc, resTy, idx, sizeAttr)
      .getResult();
}

mlir::Value IRBuilder::emitLoad(mlir::Value addr, mlir::Value mask,
                                mlir::Value other, ElementType elemType) {
  auto loc = mlir::UnknownLoc::get(&impl_->context);
  mlir::Type resTy = elementTypeToMLIR(elemType, &impl_->context);
  return impl_->builder->create<tinyton::LoadOp>(loc, resTy, addr, mask, other)
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
