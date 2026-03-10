#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/Compiler/Pipeline.h"
#include "tiny-ton/Runtime/Simulator.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  // Build IR: const 3, const 5, add, const 0 (addr), store, ret
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
  opts.emitMode = tinyton::CompileOptions::EmitMode::Asm;
  auto result = tinyton::compileModule(module, opts);

  assert(result.success && "compilation failed");
  std::printf("=== Assembly ===\n%s\n", result.output.c_str());

  std::vector<uint16_t> binary;
  for (auto &inst : result.instructions) {
    binary.push_back(inst.encoding);
  }

  std::printf("=== Binary (%zu instructions) ===\n", binary.size());
  for (size_t i = 0; i < binary.size(); ++i) {
    std::printf("  [%zu] 0x%04X\n", i, binary[i]);
  }

  tinyton::SimulatedGPU gpu;
  gpu.loadProgram(binary);
  gpu.run(1, 1);

  auto mem = gpu.readMemory(0, 1);
  std::printf("\n=== Result ===\n");
  std::printf("  mem[0] = %d (expected 8)\n", mem[0]);

  if (mem[0] == 8) {
    std::printf("\nPASS: 3 + 5 = 8\n");
    return 0;
  } else {
    std::fprintf(stderr, "\nFAIL: expected mem[0] == 8, got %d\n", mem[0]);
    return 1;
  }
}
