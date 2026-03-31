#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/IR/ElementType.h"
#include "tiny-ton/Compiler/Pipeline.h"
#include "tiny-ton/Runtime/Simulator.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static float regToFloat(int32_t r) {
  float f;
  std::memcpy(&f, &r, 4);
  return f;
}

static int32_t floatToReg(float f) {
  int32_t r;
  std::memcpy(&r, &f, 4);
  return r;
}

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

static bool testReduceSumF32() {
  // 4 threads, each loads one f32 value, reduce_sum, thread 0 stores result.
  // Input: [1.0, 2.0, 3.0, 4.0], expected sum = 10.0
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto inBase = b.emitArg(0, true, tinyton::ElementType::F32);
  auto tid = b.emitThreadId(0);
  auto addr = b.emitAdd(inBase, tid);
  auto val = b.emitLoad(addr, {}, tinyton::ElementType::F32);
  auto total = b.emitReduceSum(val);
  auto outAddr = b.emitArg(1, true, tinyton::ElementType::F32);
  b.emitStore(outAddr, total);
  b.emitRet();

  auto binary = compile(b);
  tinyton::SimulatedGPU gpu(1024);
  gpu.loadProgram(binary);
  gpu.writeMemory(0, {floatToReg(1.0f), floatToReg(2.0f),
                      floatToReg(3.0f), floatToReg(4.0f)});
  gpu.setArgs({0, 100});
  gpu.run(1, 4);

  float result = regToFloat(gpu.readMemory(100, 1)[0]);
  bool ok = std::fabs(result - 10.0f) < 1e-5f;
  if (!ok)
    std::printf("    got %f, expected 10.0\n", result);
  return ok;
}

static bool testReduceMaxF32() {
  // 4 threads, each loads one f32 value, reduce_max, all threads store result.
  // Input: [1.0, 4.0, 2.0, 3.0], expected max = 4.0
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto inBase = b.emitArg(0, true, tinyton::ElementType::F32);
  auto tid = b.emitThreadId(0);
  auto addr = b.emitAdd(inBase, tid);
  auto val = b.emitLoad(addr, {}, tinyton::ElementType::F32);
  auto mx = b.emitReduceMax(val);
  auto outAddr = b.emitArg(1, true, tinyton::ElementType::F32);
  b.emitStore(outAddr, mx);
  b.emitRet();

  auto binary = compile(b);
  tinyton::SimulatedGPU gpu(1024);
  gpu.loadProgram(binary);
  gpu.writeMemory(0, {floatToReg(1.0f), floatToReg(4.0f),
                      floatToReg(2.0f), floatToReg(3.0f)});
  gpu.setArgs({0, 100});
  gpu.run(1, 4);

  float result = regToFloat(gpu.readMemory(100, 1)[0]);
  bool ok = std::fabs(result - 4.0f) < 1e-5f;
  if (!ok)
    std::printf("    got %f, expected 4.0\n", result);
  return ok;
}

static bool testReduceSumI32() {
  // 4 threads with integer values, reduce_sum.
  // Values: thread0=10, thread1=20, thread2=30, thread3=40 -> sum=100
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto inBase = b.emitArg(0);
  auto tid = b.emitThreadId(0);
  auto addr = b.emitAdd(inBase, tid);
  auto val = b.emitLoad(addr);
  auto total = b.emitReduceSum(val);
  auto outAddr = b.emitArg(1);
  b.emitStore(outAddr, total);
  b.emitRet();

  auto binary = compile(b);
  tinyton::SimulatedGPU gpu(1024);
  gpu.loadProgram(binary);
  gpu.writeMemory(0, {10, 20, 30, 40});
  gpu.setArgs({0, 100});
  gpu.run(1, 4);

  int32_t result = gpu.readMemory(100, 1)[0];
  bool ok = (result == 100);
  if (!ok)
    std::printf("    got %d, expected 100\n", result);
  return ok;
}

static bool testReduceMaxI32() {
  // 4 threads with integer values, reduce_max.
  // Values: 10, 40, 20, 30 -> max = 40
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto inBase = b.emitArg(0);
  auto tid = b.emitThreadId(0);
  auto addr = b.emitAdd(inBase, tid);
  auto val = b.emitLoad(addr);
  auto mx = b.emitReduceMax(val);
  auto outAddr = b.emitArg(1);
  b.emitStore(outAddr, mx);
  b.emitRet();

  auto binary = compile(b);
  tinyton::SimulatedGPU gpu(1024);
  gpu.loadProgram(binary);
  gpu.writeMemory(0, {10, 40, 20, 30});
  gpu.setArgs({0, 100});
  gpu.run(1, 4);

  int32_t result = gpu.readMemory(100, 1)[0];
  bool ok = (result == 40);
  if (!ok)
    std::printf("    got %d, expected 40\n", result);
  return ok;
}

int main() {
  struct { const char *name; bool (*fn)(); } tests[] = {
    {"reduce_sum_f32", testReduceSumF32},
    {"reduce_max_f32", testReduceMaxF32},
    {"reduce_sum_i32", testReduceSumI32},
    {"reduce_max_i32", testReduceMaxI32},
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
