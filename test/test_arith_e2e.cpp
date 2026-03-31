#include "tiny-ton/IR/Builder.h"
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

static int32_t runAndRead(const std::vector<uint16_t> &binary) {
  tinyton::SimulatedGPU gpu;
  gpu.loadProgram(binary);
  gpu.run(1, 1);
  return gpu.readMemory(0, 1)[0];
}

static bool testI32Add() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitConst(7);
  auto c = b.emitConst(3);
  auto r = b.emitAdd(a, c);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return runAndRead(compile(b)) == 10;
}

static bool testI32Sub() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitConst(10);
  auto c = b.emitConst(4);
  auto r = b.emitSub(a, c);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return runAndRead(compile(b)) == 6;
}

static bool testI32Mul() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitConst(6);
  auto c = b.emitConst(7);
  auto r = b.emitMul(a, c);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return runAndRead(compile(b)) == 42;
}

static bool testI32CmpLt() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitConst(3);
  auto c = b.emitConst(5);
  auto r = b.emitCmpLt(a, c);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return runAndRead(compile(b)) == 1;
}

static bool testI32CmpLtFalse() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitConst(5);
  auto c = b.emitConst(3);
  auto r = b.emitCmpLt(a, c);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return runAndRead(compile(b)) == 0;
}

static bool testF32Add() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitFConst(2.5);
  auto c = b.emitFConst(3.5);
  auto r = b.emitAdd(a, c);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  float result = regToFloat(runAndRead(compile(b)));
  return std::fabs(result - 6.0f) < 1e-5f;
}

static bool testF32Sub() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitFConst(10.0);
  auto c = b.emitFConst(3.5);
  auto r = b.emitSub(a, c);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  float result = regToFloat(runAndRead(compile(b)));
  return std::fabs(result - 6.5f) < 1e-5f;
}

static bool testF32Mul() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitFConst(3.0);
  auto c = b.emitFConst(4.0);
  auto r = b.emitMul(a, c);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  float result = regToFloat(runAndRead(compile(b)));
  return std::fabs(result - 12.0f) < 1e-5f;
}

static bool testF32Div() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitFConst(10.0);
  auto c = b.emitFConst(4.0);
  auto r = b.emitDiv(a, c);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  float result = regToFloat(runAndRead(compile(b)));
  return std::fabs(result - 2.5f) < 1e-5f;
}

int main() {
  struct { const char *name; bool (*fn)(); } tests[] = {
    {"i32_add", testI32Add},
    {"i32_sub", testI32Sub},
    {"i32_mul", testI32Mul},
    {"i32_cmp_lt_true", testI32CmpLt},
    {"i32_cmp_lt_false", testI32CmpLtFalse},
    {"f32_add", testF32Add},
    {"f32_sub", testF32Sub},
    {"f32_mul", testF32Mul},
    {"f32_div", testF32Div},
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
