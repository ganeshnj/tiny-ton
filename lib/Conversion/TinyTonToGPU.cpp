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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace tinyton {

namespace {

struct ArgInfo {
  int64_t index;
  bool isPointer;
};

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
  auto i1Ty = mlir::IntegerType::get(ctx, 1);
  auto globalPtrTy = mlir::LLVM::LLVMPointerType::get(ctx, 1);

  llvm::SmallVector<mlir::Operation *> srcOps;
  for (auto &op : srcModule.getBody()->getOperations())
    srcOps.push_back(&op);

  llvm::SmallVector<ArgInfo> args;
  for (auto *op : srcOps) {
    if (auto argOp = llvm::dyn_cast<tinyton::ArgOp>(op)) {
      int64_t idx = argOp.getIndex();
      bool isPtr = argOp.getIsPointer();
      args.push_back({idx, isPtr});
    }
  }

  llvm::SmallVector<mlir::Type> funcArgTypes;
  funcArgTypes.resize(args.size());
  for (auto &ai : args)
    funcArgTypes[ai.index] = ai.isPointer ? (mlir::Type)globalPtrTy : i32Ty;

  std::string kernelName = "tinyton_kernel";
  result.kernelName = kernelName;

  auto newModule = mlir::ModuleOp::create(loc);
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(newModule.getBody());

  auto gpuModule = builder.create<mlir::gpu::GPUModuleOp>(loc, "tinyton_kernels");

  builder.setInsertionPointToStart(gpuModule.getBody());

  auto funcType = builder.getFunctionType(funcArgTypes, {});
  auto gpuFunc = builder.create<mlir::gpu::GPUFuncOp>(loc, kernelName, funcType);
  gpuFunc->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                   builder.getUnitAttr());

  auto *entryBlock = &gpuFunc.getBody().front();
  builder.setInsertionPointToStart(entryBlock);

  mlir::IRMapping valueMap;
  llvm::DenseSet<mlir::Value> pointerValues;

  for (auto *op : srcOps) {
    if (auto argOp = llvm::dyn_cast<tinyton::ArgOp>(op)) {
      int64_t idx = argOp.getIndex();
      auto blockArg = entryBlock->getArgument(idx);
      valueMap.map(argOp.getResult(), blockArg);
      if (argOp.getIsPointer())
        pointerValues.insert(blockArg);

    } else if (auto constOp = llvm::dyn_cast<tinyton::ConstOp>(op)) {
      auto val = builder.create<mlir::arith::ConstantIntOp>(
          loc, constOp.getValue(), i32Ty);
      valueMap.map(constOp.getResult(), val);

    } else if (auto pidOp = llvm::dyn_cast<tinyton::ProgramIdOp>(op)) {
      auto dim = static_cast<mlir::gpu::Dimension>(pidOp.getAxis());
      auto bid = builder.create<mlir::gpu::BlockIdOp>(loc, dim);
      auto cast =
          builder.create<mlir::arith::IndexCastOp>(loc, i32Ty, bid);
      valueMap.map(pidOp.getResult(), cast);

    } else if (auto tidOp = llvm::dyn_cast<tinyton::ThreadIdOp>(op)) {
      auto dim = static_cast<mlir::gpu::Dimension>(tidOp.getAxis());
      auto tid = builder.create<mlir::gpu::ThreadIdOp>(loc, dim);
      auto cast =
          builder.create<mlir::arith::IndexCastOp>(loc, i32Ty, tid);
      valueMap.map(tidOp.getResult(), cast);

    } else if (auto addOp = llvm::dyn_cast<tinyton::AddOp>(op)) {
      auto lhs = valueMap.lookup(addOp.getLhs());
      auto rhs = valueMap.lookup(addOp.getRhs());

      bool lhsIsPtr = pointerValues.count(lhs);
      bool rhsIsPtr = pointerValues.count(rhs);

      if (lhsIsPtr || rhsIsPtr) {
        auto base = lhsIsPtr ? lhs : rhs;
        auto offset = lhsIsPtr ? rhs : lhs;
        auto gep = builder.create<mlir::LLVM::GEPOp>(
            loc, globalPtrTy, i32Ty, base, mlir::ValueRange{offset});
        valueMap.map(addOp.getResult(), gep.getResult());
        pointerValues.insert(gep.getResult());
      } else {
        auto add = builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        valueMap.map(addOp.getResult(), add);
      }

    } else if (auto subOp = llvm::dyn_cast<tinyton::SubOp>(op)) {
      auto lhs = valueMap.lookup(subOp.getLhs());
      auto rhs = valueMap.lookup(subOp.getRhs());
      auto sub = builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
      valueMap.map(subOp.getResult(), sub);

    } else if (auto mulOp = llvm::dyn_cast<tinyton::MulOp>(op)) {
      auto lhs = valueMap.lookup(mulOp.getLhs());
      auto rhs = valueMap.lookup(mulOp.getRhs());
      auto mul = builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
      valueMap.map(mulOp.getResult(), mul);

    } else if (auto cmpOp = llvm::dyn_cast<tinyton::CmpLtOp>(op)) {
      auto lhs = valueMap.lookup(cmpOp.getLhs());
      auto rhs = valueMap.lookup(cmpOp.getRhs());
      auto cmp = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
      auto ext = builder.create<mlir::arith::ExtUIOp>(loc, i32Ty, cmp);
      valueMap.map(cmpOp.getResult(), ext);

    } else if (auto loadOp = llvm::dyn_cast<tinyton::LoadOp>(op)) {
      auto addr = valueMap.lookup(loadOp.getAddr());

      if (loadOp.getMask()) {
        auto mask = valueMap.lookup(loadOp.getMask());
        auto maskBit = builder.create<mlir::arith::TruncIOp>(loc, i1Ty, mask);

        auto &region = gpuFunc.getBody();

        auto *thenBlock = new mlir::Block();
        auto *mergeBlock = new mlir::Block();
        mergeBlock->addArgument(i32Ty, loc);

        region.push_back(thenBlock);
        region.push_back(mergeBlock);

        auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i32Ty);
        builder.create<mlir::cf::CondBranchOp>(loc, maskBit, thenBlock,
                                               mergeBlock,
                                               mlir::ValueRange{zero});

        builder.setInsertionPointToStart(thenBlock);
        auto loaded = builder.create<mlir::LLVM::LoadOp>(loc, i32Ty, addr);
        builder.create<mlir::cf::BranchOp>(loc, mergeBlock,
                                           mlir::ValueRange{loaded});

        builder.setInsertionPointToStart(mergeBlock);
        valueMap.map(loadOp.getResult(), mergeBlock->getArgument(0));
      } else {
        auto loaded =
            builder.create<mlir::LLVM::LoadOp>(loc, i32Ty, addr);
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
      srcModule.getBody()->begin(),
      newModule.getBody()->getOperations());
  newModule.erase();

  result.success = true;
  return result;
}

} // namespace tinyton
