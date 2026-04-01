#include "tiny-ton/Conversion/TinyTonToGPU.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonDialect.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonOps.h"
#include "tiny-ton/IR/ElementType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
  ScalarF16,
  PtrToI32,
  PtrToF32,
  PtrToF16,
};

struct ArgInfo {
  int64_t index;
  ArgKind kind;

  bool isPointer() const {
    return kind == ArgKind::PtrToI32 || kind == ArgKind::PtrToF32 ||
           kind == ArgKind::PtrToF16;
  }

  mlir::Type elementType(mlir::MLIRContext *ctx) const {
    switch (kind) {
    case ArgKind::ScalarF32:
    case ArgKind::PtrToF32:
      return mlir::Float32Type::get(ctx);
    case ArgKind::ScalarF16:
    case ArgKind::PtrToF16:
      return mlir::Float16Type::get(ctx);
    default:
      return mlir::IntegerType::get(ctx, 32);
    }
  }
};

ArgKind classifyArg(bool isPointer, int64_t elemTypeInt) {
  auto et = static_cast<ElementType>(elemTypeInt);
  if (isPointer) {
    switch (et) {
    case ElementType::F32:
      return ArgKind::PtrToF32;
    case ElementType::F16:
      return ArgKind::PtrToF16;
    default:
      return ArgKind::PtrToI32;
    }
  }
  switch (et) {
  case ElementType::F32:
    return ArgKind::ScalarF32;
  case ElementType::F16:
    return ArgKind::ScalarF16;
  default:
    return ArgKind::ScalarI32;
  }
}

bool isFloatType(mlir::Type ty) {
  return llvm::isa<mlir::FloatType>(ty);
}

mlir::Value makeFloatZero(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Type floatTy) {
  const auto &sem = llvm::cast<mlir::FloatType>(floatTy).getFloatSemantics();
  return builder.create<mlir::arith::ConstantFloatOp>(
      loc, llvm::APFloat::getZero(sem), llvm::cast<mlir::FloatType>(floatTy));
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
  ctx->getOrLoadDialect<mlir::math::MathDialect>();
  ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();

  auto i32Ty = mlir::IntegerType::get(ctx, 32);
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
           classifyArg(argOp.getIsPointer(), argOp.getElemType())});
    }
  }

  llvm::SmallVector<mlir::Type> funcArgTypes;
  funcArgTypes.resize(args.size());
  for (auto &ai : args) {
    if (ai.isPointer())
      funcArgTypes[ai.index] = globalPtrTy;
    else
      funcArgTypes[ai.index] = ai.elementType(ctx);
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

  bool hasReduceOps = false;
  for (auto *op : srcOps) {
    if (llvm::isa<tinyton::ReduceSumOp, tinyton::ReduceMaxOp>(op)) {
      hasReduceOps = true;
      break;
    }
  }

  mlir::BlockArgument shmemArg;
  if (hasReduceOps) {
    auto f32Ty = mlir::Float32Type::get(ctx);
    auto addrSpace = mlir::gpu::AddressSpaceAttr::get(
        ctx, mlir::gpu::AddressSpace::Workgroup);
    auto shmemTy = mlir::MemRefType::get(
        {32}, f32Ty, mlir::MemRefLayoutAttrInterface{}, addrSpace);
    shmemArg = gpuFunc.addWorkgroupAttribution(shmemTy, loc);
  }

  auto *entryBlock = &gpuFunc.getBody().front();
  builder.setInsertionPointToStart(entryBlock);

  mlir::IRMapping valueMap;
  llvm::DenseMap<mlir::Value, mlir::Type> pointerElementTypes;

  for (auto *op : srcOps) {
    if (auto argOp = llvm::dyn_cast<tinyton::ArgOp>(op)) {
      auto kind = classifyArg(argOp.getIsPointer(), argOp.getElemType());
      auto blockArg = entryBlock->getArgument(argOp.getIndex());
      valueMap.map(argOp.getResult(), blockArg);
      if (kind == ArgKind::PtrToI32 || kind == ArgKind::PtrToF32 ||
          kind == ArgKind::PtrToF16) {
        ArgInfo info{argOp.getIndex(), kind};
        pointerElementTypes[blockArg] = info.elementType(ctx);
      }

    } else if (auto constOp = llvm::dyn_cast<tinyton::ConstOp>(op)) {
      auto val = builder.create<mlir::arith::ConstantIntOp>(
          loc, constOp.getValue(), i32Ty);
      valueMap.map(constOp.getResult(), val);

    } else if (auto fconstOp = llvm::dyn_cast<tinyton::FConstOp>(op)) {
      auto f32Ty = mlir::Float32Type::get(ctx);
      auto val = builder.create<mlir::arith::ConstantFloatOp>(
          loc, fconstOp.getValue(), f32Ty);
      valueMap.map(fconstOp.getResult(), val);

    } else if (auto hconstOp = llvm::dyn_cast<tinyton::HConstOp>(op)) {
      auto f16Ty = mlir::Float16Type::get(ctx);
      llvm::APFloat f32Val = hconstOp.getValue();
      bool losesInfo;
      f32Val.convert(llvm::APFloat::IEEEhalf(),
                     llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      auto val = builder.create<mlir::arith::ConstantFloatOp>(
          loc, f32Val, f16Ty);
      valueMap.map(hconstOp.getResult(), val);

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
        mlir::Type elemTy = lhsIsPtr ? it->second : it2->second;
        auto gep = builder.create<mlir::LLVM::GEPOp>(
            loc, globalPtrTy, elemTy, base, mlir::ValueRange{offset});
        valueMap.map(addOp.getResult(), gep.getResult());
        pointerElementTypes[gep.getResult()] = elemTy;
      } else if (isFloatType(lhs.getType())) {
        auto add = builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
        valueMap.map(addOp.getResult(), add);
      } else {
        auto add = builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        valueMap.map(addOp.getResult(), add);
      }

    } else if (auto subOp = llvm::dyn_cast<tinyton::SubOp>(op)) {
      auto lhs = valueMap.lookup(subOp.getLhs());
      auto rhs = valueMap.lookup(subOp.getRhs());
      if (isFloatType(lhs.getType())) {
        auto sub = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
        valueMap.map(subOp.getResult(), sub);
      } else {
        auto sub = builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
        valueMap.map(subOp.getResult(), sub);
      }

    } else if (auto mulOp = llvm::dyn_cast<tinyton::MulOp>(op)) {
      auto lhs = valueMap.lookup(mulOp.getLhs());
      auto rhs = valueMap.lookup(mulOp.getRhs());
      if (isFloatType(lhs.getType())) {
        auto mul = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
        valueMap.map(mulOp.getResult(), mul);
      } else {
        auto mul = builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
        valueMap.map(mulOp.getResult(), mul);
      }

    } else if (auto divOp = llvm::dyn_cast<tinyton::DivOp>(op)) {
      auto lhs = valueMap.lookup(divOp.getLhs());
      auto rhs = valueMap.lookup(divOp.getRhs());
      if (isFloatType(lhs.getType())) {
        auto div = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
        valueMap.map(divOp.getResult(), div);
      } else {
        auto div = builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
        valueMap.map(divOp.getResult(), div);
      }

    } else if (auto cmpOp = llvm::dyn_cast<tinyton::CmpLtOp>(op)) {
      auto lhs = valueMap.lookup(cmpOp.getLhs());
      auto rhs = valueMap.lookup(cmpOp.getRhs());
      if (isFloatType(lhs.getType())) {
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

    } else if (auto expOp = llvm::dyn_cast<tinyton::ExpOp>(op)) {
      auto operand = valueMap.lookup(expOp.getOperand());
      auto ty = operand.getType();
      mlir::Value res;
      if (ty.isF16()) {
        auto f32Ty = mlir::Float32Type::get(ctx);
        auto ext = builder.create<mlir::arith::ExtFOp>(loc, f32Ty, operand);
        auto computed = builder.create<mlir::math::ExpOp>(loc, ext);
        res = builder.create<mlir::arith::TruncFOp>(loc, ty, computed);
      } else {
        res = builder.create<mlir::math::ExpOp>(loc, operand);
      }
      valueMap.map(expOp.getResult(), res);

    } else if (auto logOp = llvm::dyn_cast<tinyton::LogOp>(op)) {
      auto operand = valueMap.lookup(logOp.getOperand());
      auto ty = operand.getType();
      mlir::Value res;
      if (ty.isF16()) {
        auto f32Ty = mlir::Float32Type::get(ctx);
        auto ext = builder.create<mlir::arith::ExtFOp>(loc, f32Ty, operand);
        auto computed = builder.create<mlir::math::LogOp>(loc, ext);
        res = builder.create<mlir::arith::TruncFOp>(loc, ty, computed);
      } else {
        res = builder.create<mlir::math::LogOp>(loc, operand);
      }
      valueMap.map(logOp.getResult(), res);

    } else if (auto sqrtOp = llvm::dyn_cast<tinyton::SqrtOp>(op)) {
      auto operand = valueMap.lookup(sqrtOp.getOperand());
      auto ty = operand.getType();
      mlir::Value res;
      if (ty.isF16()) {
        auto f32Ty = mlir::Float32Type::get(ctx);
        auto ext = builder.create<mlir::arith::ExtFOp>(loc, f32Ty, operand);
        auto computed = builder.create<mlir::math::SqrtOp>(loc, ext);
        res = builder.create<mlir::arith::TruncFOp>(loc, ty, computed);
      } else {
        res = builder.create<mlir::math::SqrtOp>(loc, operand);
      }
      valueMap.map(sqrtOp.getResult(), res);

    } else if (auto rsqrtOp = llvm::dyn_cast<tinyton::RsqrtOp>(op)) {
      auto operand = valueMap.lookup(rsqrtOp.getOperand());
      auto ty = operand.getType();
      auto f32Ty = mlir::Float32Type::get(ctx);
      mlir::Value src = operand;
      if (ty.isF16())
        src = builder.create<mlir::arith::ExtFOp>(loc, f32Ty, operand);
      auto sqrtVal = builder.create<mlir::math::SqrtOp>(loc, src);
      auto one = builder.create<mlir::arith::ConstantFloatOp>(
          loc, llvm::APFloat(1.0f), f32Ty);
      auto divVal =
          builder.create<mlir::arith::DivFOp>(loc, one, sqrtVal);
      mlir::Value res = divVal;
      if (ty.isF16())
        res = builder.create<mlir::arith::TruncFOp>(loc, ty, divVal);
      valueMap.map(rsqrtOp.getResult(), res);

    } else if (auto absOp = llvm::dyn_cast<tinyton::AbsOp>(op)) {
      auto operand = valueMap.lookup(absOp.getOperand());
      mlir::Value res;
      if (isFloatType(operand.getType())) {
        auto ty = operand.getType();
        if (ty.isF16()) {
          auto f32Ty = mlir::Float32Type::get(ctx);
          auto ext = builder.create<mlir::arith::ExtFOp>(loc, f32Ty, operand);
          auto computed = builder.create<mlir::math::AbsFOp>(loc, ext);
          res = builder.create<mlir::arith::TruncFOp>(loc, ty, computed);
        } else {
          res = builder.create<mlir::math::AbsFOp>(loc, operand);
        }
      } else {
        auto zero =
            builder.create<mlir::arith::ConstantIntOp>(loc, 0, i32Ty);
        auto neg = builder.create<mlir::arith::SubIOp>(loc, zero, operand);
        auto cmp = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, operand, zero);
        res = builder.create<mlir::arith::SelectOp>(loc, cmp, neg, operand);
      }
      valueMap.map(absOp.getResult(), res);

    } else if (auto maxOp = llvm::dyn_cast<tinyton::MaxOp>(op)) {
      auto lhs = valueMap.lookup(maxOp.getLhs());
      auto rhs = valueMap.lookup(maxOp.getRhs());
      mlir::Value res;
      if (isFloatType(lhs.getType()))
        res = builder.create<mlir::arith::MaxNumFOp>(loc, lhs, rhs);
      else
        res = builder.create<mlir::arith::MaxSIOp>(loc, lhs, rhs);
      valueMap.map(maxOp.getResult(), res);

    } else if (auto reduceSumOp = llvm::dyn_cast<tinyton::ReduceSumOp>(op)) {
      auto operand = valueMap.lookup(reduceSumOp.getOperand());
      auto ty = operand.getType();
      auto f32Ty = mlir::Float32Type::get(ctx);
      mlir::Value val = operand;
      if (ty.isF16())
        val = builder.create<mlir::arith::ExtFOp>(loc, f32Ty, operand);

      bool isFloat = isFloatType(val.getType());
      auto combineSum = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::AddFOp>(loc, a, b)
                   : (mlir::Value)builder.create<mlir::arith::AddIOp>(loc, a,
                                                                      b);
      };
      mlir::Value identity =
          isFloat ? makeFloatZero(builder, loc, val.getType())
                  : (mlir::Value)builder.create<mlir::arith::ConstantIntOp>(
                        loc, 0, i32Ty);

      constexpr int kWarpSize = 32;
      for (int offset : {1, 2, 4, 8, 16}) {
        auto shuf = builder.create<mlir::gpu::ShuffleOp>(
            loc, val, offset, kWarpSize, mlir::gpu::ShuffleMode::XOR);
        val = combineSum(val, shuf.getShuffleResult());
      }

      auto tidIdx = builder.create<mlir::gpu::ThreadIdOp>(
          loc, mlir::gpu::Dimension::x);
      auto tid =
          builder.create<mlir::arith::IndexCastOp>(loc, i32Ty, tidIdx);
      auto c32 = builder.create<mlir::arith::ConstantIntOp>(loc, 32, i32Ty);
      auto warpId = builder.create<mlir::arith::DivUIOp>(loc, tid, c32);
      auto laneId = builder.create<mlir::arith::RemUIOp>(loc, tid, c32);

      mlir::Value storeVal =
          isFloat ? val
                  : (mlir::Value)builder.create<mlir::arith::BitcastOp>(
                        loc, f32Ty, val);
      auto warpIdIdx = builder.create<mlir::arith::IndexCastOp>(
          loc, builder.getIndexType(), warpId);
      builder.create<mlir::memref::StoreOp>(loc, storeVal, shmemArg,
                                            mlir::ValueRange{warpIdIdx});

      builder.create<mlir::gpu::BarrierOp>(loc);

      auto blockDimIdx = builder.create<mlir::gpu::BlockDimOp>(
          loc, mlir::gpu::Dimension::x);
      auto blockDim =
          builder.create<mlir::arith::IndexCastOp>(loc, i32Ty, blockDimIdx);
      auto numWarps = builder.create<mlir::arith::DivUIOp>(loc, blockDim, c32);
      auto inBounds = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, laneId, numWarps);
      auto c0 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i32Ty);
      auto safeLane = builder.create<mlir::arith::SelectOp>(loc, inBounds,
                                                            laneId, c0);
      auto safeLaneIdx = builder.create<mlir::arith::IndexCastOp>(
          loc, builder.getIndexType(), safeLane);
      auto loadedF32 = builder.create<mlir::memref::LoadOp>(
          loc, shmemArg, mlir::ValueRange{safeLaneIdx});
      mlir::Value loaded =
          isFloat ? (mlir::Value)loadedF32
                  : (mlir::Value)builder.create<mlir::arith::BitcastOp>(
                        loc, i32Ty, loadedF32);
      val = builder.create<mlir::arith::SelectOp>(loc, inBounds, loaded,
                                                  identity);

      for (int offset : {1, 2, 4, 8, 16}) {
        auto shuf = builder.create<mlir::gpu::ShuffleOp>(
            loc, val, offset, kWarpSize, mlir::gpu::ShuffleMode::XOR);
        val = combineSum(val, shuf.getShuffleResult());
      }

      if (ty.isF16())
        val = builder.create<mlir::arith::TruncFOp>(loc, ty, val);
      valueMap.map(reduceSumOp.getResult(), val);

    } else if (auto reduceMaxOp = llvm::dyn_cast<tinyton::ReduceMaxOp>(op)) {
      auto operand = valueMap.lookup(reduceMaxOp.getOperand());
      auto ty = operand.getType();
      auto f32Ty = mlir::Float32Type::get(ctx);
      mlir::Value val = operand;
      if (ty.isF16())
        val = builder.create<mlir::arith::ExtFOp>(loc, f32Ty, operand);

      bool isFloat = isFloatType(val.getType());
      auto combineMax = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        return isFloat ? (mlir::Value)builder.create<mlir::arith::MaxNumFOp>(
                             loc, a, b)
                       : (mlir::Value)builder.create<mlir::arith::MaxSIOp>(
                             loc, a, b);
      };
      mlir::Value identity;
      if (isFloat) {
        const auto &sem =
            llvm::cast<mlir::FloatType>(val.getType()).getFloatSemantics();
        identity = builder.create<mlir::arith::ConstantFloatOp>(
            loc, llvm::APFloat::getInf(sem, true),
            llvm::cast<mlir::FloatType>(val.getType()));
      } else {
        identity = builder.create<mlir::arith::ConstantIntOp>(
            loc, std::numeric_limits<int32_t>::min(), i32Ty);
      }

      constexpr int kWarpSize = 32;
      for (int offset : {1, 2, 4, 8, 16}) {
        auto shuf = builder.create<mlir::gpu::ShuffleOp>(
            loc, val, offset, kWarpSize, mlir::gpu::ShuffleMode::XOR);
        val = combineMax(val, shuf.getShuffleResult());
      }

      auto tidIdx = builder.create<mlir::gpu::ThreadIdOp>(
          loc, mlir::gpu::Dimension::x);
      auto tid =
          builder.create<mlir::arith::IndexCastOp>(loc, i32Ty, tidIdx);
      auto c32 = builder.create<mlir::arith::ConstantIntOp>(loc, 32, i32Ty);
      auto warpId = builder.create<mlir::arith::DivUIOp>(loc, tid, c32);
      auto laneId = builder.create<mlir::arith::RemUIOp>(loc, tid, c32);

      mlir::Value storeVal =
          isFloat ? val
                  : (mlir::Value)builder.create<mlir::arith::BitcastOp>(
                        loc, f32Ty, val);
      auto warpIdIdx = builder.create<mlir::arith::IndexCastOp>(
          loc, builder.getIndexType(), warpId);
      builder.create<mlir::memref::StoreOp>(loc, storeVal, shmemArg,
                                            mlir::ValueRange{warpIdIdx});

      builder.create<mlir::gpu::BarrierOp>(loc);

      auto blockDimIdx = builder.create<mlir::gpu::BlockDimOp>(
          loc, mlir::gpu::Dimension::x);
      auto blockDim =
          builder.create<mlir::arith::IndexCastOp>(loc, i32Ty, blockDimIdx);
      auto numWarps = builder.create<mlir::arith::DivUIOp>(loc, blockDim, c32);
      auto inBounds = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, laneId, numWarps);
      auto c0 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i32Ty);
      auto safeLane = builder.create<mlir::arith::SelectOp>(loc, inBounds,
                                                            laneId, c0);
      auto safeLaneIdx = builder.create<mlir::arith::IndexCastOp>(
          loc, builder.getIndexType(), safeLane);
      auto loadedF32 = builder.create<mlir::memref::LoadOp>(
          loc, shmemArg, mlir::ValueRange{safeLaneIdx});
      mlir::Value loaded =
          isFloat ? (mlir::Value)loadedF32
                  : (mlir::Value)builder.create<mlir::arith::BitcastOp>(
                        loc, i32Ty, loadedF32);
      val = builder.create<mlir::arith::SelectOp>(loc, inBounds, loaded,
                                                  identity);

      for (int offset : {1, 2, 4, 8, 16}) {
        auto shuf = builder.create<mlir::gpu::ShuffleOp>(
            loc, val, offset, kWarpSize, mlir::gpu::ShuffleMode::XOR);
        val = combineMax(val, shuf.getShuffleResult());
      }

      if (ty.isF16())
        val = builder.create<mlir::arith::TruncFOp>(loc, ty, val);
      valueMap.map(reduceMaxOp.getResult(), val);

    } else if (auto loadOp = llvm::dyn_cast<tinyton::LoadOp>(op)) {
      auto addr = valueMap.lookup(loadOp.getAddr());

      mlir::Type elemTy = i32Ty;
      auto ptIt = pointerElementTypes.find(addr);
      if (ptIt != pointerElementTypes.end())
        elemTy = ptIt->second;
      else if (isFloatType(loadOp.getResult().getType()))
        elemTy = loadOp.getResult().getType();

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
        if (isFloatType(elemTy))
          zero = makeFloatZero(builder, loc, elemTy);
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
