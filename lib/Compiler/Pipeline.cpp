#include "tiny-ton/Compiler/Pipeline.h"
#include "tiny-ton/Compiler/CodeGen.h"
#include "tiny-ton/Compiler/RegisterAlloc.h"
#include "tiny-ton/Conversion/TinyTonToGPU.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <mutex>

namespace {

struct CombinedGPULoweringPass
    : public mlir::OperationPass<mlir::gpu::GPUModuleOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CombinedGPULoweringPass)

  CombinedGPULoweringPass()
      : mlir::OperationPass<mlir::gpu::GPUModuleOp>(
            mlir::TypeID::get<CombinedGPULoweringPass>()) {}

  llvm::StringRef getName() const override { return "CombinedGPULoweringPass"; }
  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<CombinedGPULoweringPass>();
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    auto *ctx = gpuModule.getContext();

    mlir::LLVMTypeConverter converter(ctx);
    mlir::ConversionTarget target(*ctx);
    mlir::RewritePatternSet patterns(ctx);

    mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
    mlir::populateMathToLLVMConversionPatterns(converter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    mlir::populateFuncToLLVMFuncOpConversionPattern(converter, patterns);
    mlir::populateGpuToNVVMConversionPatterns(converter, patterns);

    mlir::configureGpuToNVVMConversionLegality(target);
    target.addIllegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalDialect<mlir::math::MathDialect>();
    target.addIllegalDialect<mlir::cf::ControlFlowDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    if (mlir::failed(
            mlir::applyPartialConversion(gpuModule, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace tinyton {

CompileResult compileModule(mlir::ModuleOp module, const CompileOptions &opts) {
  CompileResult result;

  RegisterMap regMap = allocateRegisters(module);
  result.instructions = emit(module, regMap);

  std::string out;
  llvm::raw_string_ostream os(out);

  switch (opts.emitMode) {
  case CompileOptions::EmitMode::MLIR: {
    module.print(os);
    break;
  }
  case CompileOptions::EmitMode::Asm: {
    for (size_t i = 0; i < result.instructions.size(); ++i) {
      auto &inst = result.instructions[i];
      os << llvm::formatv("{0}: 0x{1:X-4}  {2}\n", i, inst.encoding,
                          inst.assembly);
    }
    break;
  }
  case CompileOptions::EmitMode::Hex: {
    for (auto &inst : result.instructions) {
      os << llvm::formatv("0x{0:X-4}\n", inst.encoding);
    }
    break;
  }
  case CompileOptions::EmitMode::Bin: {
    for (auto &inst : result.instructions) {
      char buf[2];
      buf[0] = static_cast<char>((inst.encoding >> 8) & 0xFF);
      buf[1] = static_cast<char>(inst.encoding & 0xFF);
      os.write(buf, 2);
    }
    break;
  }
  }

  result.output = os.str();
  result.success = true;
  return result;
}

static void initNVPTXOnce() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

NVPTXCompileResult compileToNVPTX(mlir::ModuleOp srcModule,
                                  llvm::StringRef smVersion) {
  NVPTXCompileResult result;
  initNVPTXOnce();

  auto *ctx = srcModule.getContext();

  auto clonedModule = srcModule.clone();

  auto gpuResult = lowerToGPU(clonedModule);
  if (!gpuResult.success) {
    result.error = "GPU lowering failed: " + gpuResult.error;
    clonedModule.erase();
    return result;
  }
  result.kernelName = gpuResult.kernelName;

  mlir::PassManager pm(ctx);
  pm.enableVerifier(true);

  auto &gpuPM = pm.nest<mlir::gpu::GPUModuleOp>();
  gpuPM.addPass(std::make_unique<CombinedGPULoweringPass>());
  gpuPM.addPass(mlir::createReconcileUnrealizedCastsPass());

  std::string diagMsg;
  mlir::ScopedDiagnosticHandler diagHandler(
      ctx, [&](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        llvm::raw_string_ostream os(diagMsg);
        diag.print(os);
        os << "\n";
        for (auto &note : diag.getNotes()) {
          os << "  note: ";
          note.print(os);
          os << "\n";
        }
        return mlir::success();
      });

  if (mlir::failed(pm.run(clonedModule))) {
    result.error = "[combined-pass-v2] MLIR pass pipeline failed";
    if (!diagMsg.empty())
      result.error += ": " + diagMsg;
    clonedModule.erase();
    return result;
  }

  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  ctx->appendDialectRegistry(registry);

  mlir::gpu::GPUModuleOp gpuModuleOp = nullptr;
  clonedModule.walk([&](mlir::gpu::GPUModuleOp op) { gpuModuleOp = op; });
  if (!gpuModuleOp) {
    result.error = "No gpu.module found after lowering";
    clonedModule.erase();
    return result;
  }

  auto extractedModule = mlir::ModuleOp::create(
      mlir::UnknownLoc::get(ctx));
  {
    mlir::OpBuilder b(ctx);
    b.setInsertionPointToStart(extractedModule.getBody());
    for (auto &op : llvm::make_early_inc_range(gpuModuleOp.getBody()->getOperations())) {
      if (llvm::isa<mlir::gpu::ModuleEndOp>(&op))
        continue;
      op.moveBefore(extractedModule.getBody(), extractedModule.getBody()->end());
    }
  }

  llvm::LLVMContext llvmCtx;
  auto llvmModule = mlir::translateModuleToLLVMIR(
      extractedModule.getOperation(), llvmCtx, "tinyton_gpu");

  extractedModule.erase();

  clonedModule.erase();

  if (!llvmModule) {
    result.error = "MLIR to LLVM IR translation failed";
    return result;
  }

  llvmModule->setTargetTriple("nvptx64-nvidia-cuda");

  std::string targetError;
  auto *target =
      llvm::TargetRegistry::lookupTarget("nvptx64-nvidia-cuda", targetError);
  if (!target) {
    result.error = "NVPTX target not found: " + targetError;
    return result;
  }

  auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine("nvptx64-nvidia-cuda", smVersion, "+ptx75",
                                  llvm::TargetOptions(), std::nullopt));
  if (!targetMachine) {
    result.error = "Failed to create target machine";
    return result;
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());

  llvm::SmallString<0> ptxBuf;
  llvm::raw_svector_ostream ptxStream(ptxBuf);
  llvm::legacy::PassManager llvmPM;

  if (targetMachine->addPassesToEmitFile(llvmPM, ptxStream, nullptr,
                                         llvm::CodeGenFileType::AssemblyFile)) {
    result.error = "Target machine cannot emit assembly";
    return result;
  }

  llvmPM.run(*llvmModule);

  result.ptx = std::string(ptxBuf.begin(), ptxBuf.end());
  result.success = true;
  return result;
}

} // namespace tinyton
