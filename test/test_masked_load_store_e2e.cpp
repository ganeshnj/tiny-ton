#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/Compiler/Pipeline.h"
#include "tiny-ton/Runtime/Simulator.h"

#include <cassert>
#include <cstdio>
#include <vector>

static std::vector<uint16_t> compile(tinyton::IRBuilder &builder) {
  auto module = builder.getModule();
  tinyton::CompileOptions opts;
  opts.emitMode = tinyton::CompileOptions::EmitMode::Asm;
  auto result = tinyton::compileModule(module, opts);
  assert(result.success && "compilation failed");
  std::vector<uint16_t> binary;
  for (auto &inst : result.instructions)
    binary.push_back(inst.encoding);
  return binary;
}

static bool testUnmaskedLoadStoreMultiThread() {
  // Kernel: each thread loads mem[arg0 + tid], adds 1, stores to
  // mem[arg0 + tid].
  // Input:  [10, 20, 30, 40]
  // Output: [11, 21, 31, 41]
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto base = b.emitArg(0);
  auto tid = b.emitThreadId(0);
  auto addr = b.emitAdd(base, tid);
  auto val = b.emitLoad(addr);
  auto one = b.emitConst(1);
  auto result = b.emitAdd(val, one);
  b.emitStore(addr, result);
  b.emitRet();

  auto binary = compile(b);
  tinyton::SimulatedGPU gpu;
  gpu.loadProgram(binary);
  gpu.writeMemory(0, {10, 20, 30, 40});
  gpu.setArgs({0});
  gpu.run(1, 4);

  auto mem = gpu.readMemory(0, 4);
  return mem[0] == 11 && mem[1] == 21 && mem[2] == 31 && mem[3] == 41;
}

static bool testMaskedLoad() {
  // Kernel: each thread loads mem[tid] with mask (tid < N), stores to
  // mem[out_base + tid]. N=2 so only threads 0,1 load real values; threads
  // 2,3 get zero.
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto inBase = b.emitArg(0);
  auto outBase = b.emitArg(1);
  auto N = b.emitArg(2);
  auto tid = b.emitThreadId(0);
  auto inAddr = b.emitAdd(inBase, tid);
  auto outAddr = b.emitAdd(outBase, tid);
  auto mask = b.emitCmpLt(tid, N);
  auto val = b.emitLoad(inAddr, mask);
  b.emitStore(outAddr, val);
  b.emitRet();

  auto binary = compile(b);
  tinyton::SimulatedGPU gpu(1024);
  gpu.loadProgram(binary);
  gpu.writeMemory(0, {100, 200, 300, 400});
  gpu.setArgs({0, 10, 2});
  gpu.run(1, 4);

  auto mem = gpu.readMemory(10, 4);
  return mem[0] == 100 && mem[1] == 200 && mem[2] == 0 && mem[3] == 0;
}

static bool testMaskedStore() {
  // Kernel: stores value 99 to mem[tid] only if tid < 2.
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto N = b.emitArg(0);
  auto tid = b.emitThreadId(0);
  auto mask = b.emitCmpLt(tid, N);
  auto val = b.emitConst(99);
  b.emitStore(tid, val, mask);
  b.emitRet();

  auto binary = compile(b);
  tinyton::SimulatedGPU gpu;
  gpu.loadProgram(binary);
  gpu.writeMemory(0, {0, 0, 0, 0});
  gpu.setArgs({2});
  gpu.run(1, 4);

  auto mem = gpu.readMemory(0, 4);
  return mem[0] == 99 && mem[1] == 99 && mem[2] == 0 && mem[3] == 0;
}

static bool testProgramId() {
  // Kernel: stores blockId to mem[blockId]. Run 4 blocks, 1 thread each.
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto pid = b.emitProgramId(0);
  b.emitStore(pid, pid);
  b.emitRet();

  auto binary = compile(b);
  tinyton::SimulatedGPU gpu;
  gpu.loadProgram(binary);
  gpu.run(4, 1);

  auto mem = gpu.readMemory(0, 4);
  return mem[0] == 0 && mem[1] == 1 && mem[2] == 2 && mem[3] == 3;
}

int main() {
  struct { const char *name; bool (*fn)(); } tests[] = {
    {"unmasked_load_store_multi_thread", testUnmaskedLoadStoreMultiThread},
    {"masked_load", testMaskedLoad},
    {"masked_store", testMaskedStore},
    {"program_id", testProgramId},
  };

  int passed = 0, failed = 0;
  for (auto &t : tests) {
    bool ok = t.fn();
    std::printf("  %s: %s\n", t.name, ok ? "PASS" : "FAIL");
    if (ok) ++passed; else ++failed;
  }
  std::printf("\n%d/%d passed\n", passed, passed + failed);
  return failed ? 1 : 0;
}
