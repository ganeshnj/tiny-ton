#include "tiny-ton/Conversion/TinyTonToGPU.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonDialect.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace tinyton {

namespace {

enum class ArgKind {
  ScalarI32,
  ScalarF32,
  PtrToI32,
  PtrToF32,
};

struct ArgInfo {
  int64_t index;
  ArgKind kind;

  bool isPointer() const {
    return kind == ArgKind::PtrToI32 || kind == ArgKind::PtrToF32;
  }

  mlir::Type elementType(mlir::Type i32Ty, mlir::Type f32Ty) const {
    return (kind == ArgKind::ScalarF32 || kind == ArgKind::PtrToF32) ? f32Ty
                                                                     : i32Ty;
  }
};

ArgKind classifyArg(bool isPointer, bool isFloat) {
  if (isPointer)
    return isFloat ? ArgKind::PtrToF32 : ArgKind::PtrToI32;
  return isFloat ? ArgKind::ScalarF32 : ArgKind::ScalarI32;
}

} // namespace

GPULoweringResult lowerToGPU(mlir::ModuleOp srcModule) {
  GPULoweringResult result;
  auto *ctx = srcModule.getContext();
  auto loc = mlir::UnknownLoc::get(ctx);

  ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
  ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  ctx->getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  auto i32Ty = mlir::IntegerType::get(ctx, 32);
  auto f32Ty = mlir::Float32Type::get(ctx);
  auto i1Ty = mlir::IntegerType::get(ctx, 1);
  auto globalPtrTy = mlir::LLVM::LLVMPointerType::get(ctx, 1);

  llvm::SmallVector<mlir::Operation *> srcOps;
  for (auto &op : srcModule.getBody()->getOperations())
    srcOps.push_back(&op);

  llvm::SmallVector<ArgInfo> args;
  for (auto *op : srcOps) {
    if (auto argOp = llvm::dyn_cast<tinyton::ArgOp>(op)) {
      args.push_back(
          {argOp.getIndex(),
           classifyArg(argOp.getIsPointer(), argOp.getIsFloat())});
    }
  }

  llvm::SmallVector<mlir::Type> funcArgTypes;
  funcArgTypes.resize(args.size());
  for (auto &ai : args) {
    if (ai.isPointer())
      funcArgTypes[ai.index] = globalPtrTy;
    else
      funcArgTypes[ai.index] = ai.elementType(i32Ty, f32Ty);
  }

  std::string kernelName = "tinyton_kernel";
  result.kernelName = kernelName;

  auto newModule = mlir::ModuleOp::create(loc);
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(newModule.getBody());

  auto gpuModule =
      builder.create<mlir::gpu::GPUModuleOp>(loc, "tinyton_kernels");

  builder.setInsertionPointToStart(gpuModule.getBody());

  auto funcType = builder.getFunctionType(funcArgTypes, {});
  auto gpuFunc =
      builder.create<mlir::gpu::GPUFuncOp>(loc, kernelName, funcType);
  gpuFunc->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                   builder.getUnitAttr());

  auto *entryBlock = &gpuFunc.getBody().front();
  builder.setInsertionPointToStart(entryBlock);

  mlir::IRMapping valueMap;

  // Track pointer values and their element types (i32 or f32).
  llvm::DenseMap<mlir::Value, mlir::Type> pointerElementTypes;

  for (auto *op : srcOps) {
    if (auto argOp = llvm::dyn_cast<tinyton::ArgOp>(op)) {
      auto kind = classifyArg(argOp.getIsPointer(), argOp.getIsFloat());
      auto blockArg = entryBlock->getArgument(argOp.getIndex());
      valueMap.map(argOp.getResult(), blockArg);
      if (kind == ArgKind::PtrToI32 || kind == ArgKind::PtrToF32) {
        mlir::Type elemTy = (kind == ArgKind::PtrToF32) ? (mlir::Type)f32Ty
                                                         : i32Ty;
        pointerElementTypes[blockArg] = elemTy;
      }

    } else if (auto constOp = llvm::dyn_cast<tinyton::ConstOp>(op)) {
      auto val = builder.create<mlir::arith::ConstantIntOp>(
          loc, constOp.getValue(), i32Ty);
      valueMap.map(constOp.getResult(), val);

    } else if (auto fconstOp = llvm::dyn_cast<tinyton::FConstOp>(op)) {
      auto val = builder.create<mlir::arith::ConstantFloatOp>(
          loc, fconstOp.getValue(), f32Ty);
      valueMap.map(fconstOp.getResult(), val);

    } else if (auto pidOp = llvm::dyn_cast<tinyton::ProgramIdOp>(op)) {
      auto dim = static_cast<mlir::gpu::Dimension>(pidOp.getAxis());
      auto bid = builder.create<mlir::gpu::BlockIdOp>(loc, dim);
      auto cast = builder.create<mlir::arith::IndexCastOp>(loc, i32Ty, bid);
      valueMap.map(pidOp.getResult(), cast);

    } else if (auto tidOp = llvm::dyn_cast<tinyton::ThreadIdOp>(op)) {
      auto dim = static_cast<mlir::gpu::Dimension>(tidOp.getAxis());
      auto tid = builder.create<mlir::gpu::ThreadIdOp>(loc, dim);
      auto cast = builder.create<mlir::arith::IndexCastOp>(loc, i32Ty, tid);
      valueMap.map(tidOp.getResult(), cast);

    } else if (auto addOp = llvm::dyn_cast<tinyton::AddOp>(op)) {
      auto lhs = valueMap.lookup(addOp.getLhs());
      auto rhs = valueMap.lookup(addOp.getRhs());

      auto it = pointerElementTypes.find(lhs);
      auto it2 = pointerElementTypes.find(rhs);
      bool lhsIsPtr = it != pointerElementTypes.end();
      bool rhsIsPtr = it2 != pointerElementTypes.end();

      if (lhsIsPtr || rhsIsPtr) {
        auto base = lhsIsPtr ? lhs : rhs;
        auto offset = lhsIsPtr ? rhs : lhs;
        mlir::Type elemTy =
            lhsIsPtr ? it->second : it2->second;
        auto gep = builder.create<mlir::LLVM::GEPOp>(
            loc, globalPtrTy, elemTy, base, mlir::ValueRange{offset});
        valueMap.map(addOp.getResult(), gep.getResult());
        pointerElementTypes[gep.getResult()] = elemTy;
      } else if (lhs.getType().isF32()) {
        auto add = builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
        valueMap.map(addOp.getResult(), add);
      } else {
        auto add = builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        valueMap.map(addOp.getResult(), add);
      }

    } else if (auto subOp = llvm::dyn_cast<tinyton::SubOp>(op)) {
      auto lhs = valueMap.lookup(subOp.getLhs());
      auto rhs = valueMap.lookup(subOp.getRhs());
      if (lhs.getType().isF32()) {
        auto sub = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
        valueMap.map(subOp.getResult(), sub);
      } else {
        auto sub = builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
        valueMap.map(subOp.getResult(), sub);
      }

    } else if (auto mulOp = llvm::dyn_cast<tinyton::MulOp>(op)) {
      auto lhs = valueMap.lookup(mulOp.getLhs());
      auto rhs = valueMap.lookup(mulOp.getRhs());
      if (lhs.getType().isF32()) {
        auto mul = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
        valueMap.map(mulOp.getResult(), mul);
      } else {
        auto mul = builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
        valueMap.map(mulOp.getResult(), mul);
      }

    } else if (auto divOp = llvm::dyn_cast<tinyton::DivOp>(op)) {
      auto lhs = valueMap.lookup(divOp.getLhs());
      auto rhs = valueMap.lookup(divOp.getRhs());
      if (lhs.getType().isF32()) {
        auto div = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
        valueMap.map(divOp.getResult(), div);
      } else {
        auto div = builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
        valueMap.map(divOp.getResult(), div);
      }

    } else if (auto cmpOp = llvm::dyn_cast<tinyton::CmpLtOp>(op)) {
      auto lhs = valueMap.lookup(cmpOp.getLhs());
      auto rhs = valueMap.lookup(cmpOp.getRhs());
      if (lhs.getType().isF32()) {
        auto cmp = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
        auto ext = builder.create<mlir::arith::ExtUIOp>(loc, i32Ty, cmp);
        valueMap.map(cmpOp.getResult(), ext);
      } else {
        auto cmp = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
        auto ext = builder.create<mlir::arith::ExtUIOp>(loc, i32Ty, cmp);
        valueMap.map(cmpOp.getResult(), ext);
      }

    } else if (auto loadOp = llvm::dyn_cast<tinyton::LoadOp>(op)) {
      auto addr = valueMap.lookup(loadOp.getAddr());

      // Determine the load element type from the pointer's tracked element type
      // or from the load op's result type.
      mlir::Type elemTy = i32Ty;
      auto ptIt = pointerElementTypes.find(addr);
      if (ptIt != pointerElementTypes.end())
        elemTy = ptIt->second;
      else if (loadOp.getResult().getType().isF32())
        elemTy = f32Ty;

      if (loadOp.getMask()) {
        auto mask = valueMap.lookup(loadOp.getMask());
        auto maskBit = builder.create<mlir::arith::TruncIOp>(loc, i1Ty, mask);

        auto &region = gpuFunc.getBody();

        auto *thenBlock = new mlir::Block();
        auto *mergeBlock = new mlir::Block();
        mergeBlock->addArgument(elemTy, loc);

        region.push_back(thenBlock);
        region.push_back(mergeBlock);

        mlir::Value zero;
        if (elemTy.isF32())
          zero = builder.create<mlir::arith::ConstantFloatOp>(
              loc, llvm::APFloat(0.0f), f32Ty);
        else
          zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i32Ty);

        builder.create<mlir::cf::CondBranchOp>(loc, maskBit, thenBlock,
                                               mergeBlock,
                                               mlir::ValueRange{zero});

        builder.setInsertionPointToStart(thenBlock);
        auto loaded = builder.create<mlir::LLVM::LoadOp>(loc, elemTy, addr);
        builder.create<mlir::cf::BranchOp>(loc, mergeBlock,
                                           mlir::ValueRange{loaded});

        builder.setInsertionPointToStart(mergeBlock);
        valueMap.map(loadOp.getResult(), mergeBlock->getArgument(0));
      } else {
        auto loaded = builder.create<mlir::LLVM::LoadOp>(loc, elemTy, addr);
        valueMap.map(loadOp.getResult(), loaded);
      }

    } else if (auto storeOp = llvm::dyn_cast<tinyton::StoreOp>(op)) {
      auto addr = valueMap.lookup(storeOp.getAddr());
      auto val = valueMap.lookup(storeOp.getValue());

      if (storeOp.getMask()) {
        auto mask = valueMap.lookup(storeOp.getMask());
        auto maskBit = builder.create<mlir::arith::TruncIOp>(loc, i1Ty, mask);

        auto &region = gpuFunc.getBody();

        auto *thenBlock = new mlir::Block();
        auto *mergeBlock = new mlir::Block();

        region.push_back(thenBlock);
        region.push_back(mergeBlock);

        builder.create<mlir::cf::CondBranchOp>(loc, maskBit, thenBlock,
                                               mergeBlock);

        builder.setInsertionPointToStart(thenBlock);
        builder.create<mlir::LLVM::StoreOp>(loc, val, addr);
        builder.create<mlir::cf::BranchOp>(loc, mergeBlock);

        builder.setInsertionPointToStart(mergeBlock);
      } else {
        builder.create<mlir::LLVM::StoreOp>(loc, val, addr);
      }

    } else if (llvm::isa<tinyton::RetOp>(op)) {
      builder.create<mlir::gpu::ReturnOp>(loc);

    } else if (llvm::isa<tinyton::BranchZeroOp>(op)) {
      // BranchZero is a simulator-only concept; skip for GPU path
    }
  }

  srcModule.getBody()->clear();
  srcModule.getBody()->getOperations().splice(
      srcModule.getBody()->begin(), newModule.getBody()->getOperations());
  newModule.erase();

  result.success = true;
  return result;
}

} // namespace tinyton
