#include "tiny-ton/Compiler/Pipeline.h"
#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/Runtime/Simulator.h"

#include <iostream>
#include <string>
#include <vector>

static void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " --test-add [--emit mlir|asm]\n";
}

static int runTestAdd(const std::string &emitMode) {
  tinyton::IRBuilder builder;
  builder.beginFunction("test_add");

  auto v3 = builder.emitConst(3);
  auto v5 = builder.emitConst(5);
  auto sum = builder.emitAdd(v3, v5);
  auto addr = builder.emitConst(0);
  builder.emitStore(addr, sum);
  builder.emitRet();

  auto module = builder.getModule();

  tinyton::CompileOptions opts;
  if (emitMode == "mlir")
    opts.emitMode = tinyton::CompileOptions::EmitMode::MLIR;
  else
    opts.emitMode = tinyton::CompileOptions::EmitMode::Asm;

  auto result = tinyton::compileModule(module, opts);
  if (!result.success) {
    std::cerr << "error: " << result.error << "\n";
    return 1;
  }

  std::cout << result.output;

  std::vector<uint16_t> binary;
  for (auto &inst : result.instructions) {
    binary.push_back(inst.encoding);
  }

  tinyton::SimulatedGPU gpu;
  gpu.loadProgram(binary);
  gpu.run(1, 1);

  auto mem = gpu.readMemory(0, 1);
  std::cerr << "simulator: mem[0] = " << mem[0] << " (expected 8)\n";

  return (mem[0] == 8) ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string emitMode = "asm";
  bool testAdd = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--emit" && i + 1 < argc) {
      emitMode = argv[++i];
    } else if (arg == "--test-add") {
      testAdd = true;
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    }
  }

  if (testAdd) {
    return runTestAdd(emitMode);
  }

  printUsage(argv[0]);
  return 1;
}
