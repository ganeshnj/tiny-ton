#include "tiny-ton/Compiler/Pipeline.h"
#include "tiny-ton/Compiler/CodeGen.h"
#include "tiny-ton/Compiler/RegisterAlloc.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

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

} // namespace tinyton
