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

static float runAndReadFloat(const std::vector<uint16_t> &binary) {
  tinyton::SimulatedGPU gpu;
  gpu.loadProgram(binary);
  gpu.run(1, 1);
  return regToFloat(gpu.readMemory(0, 1)[0]);
}

static bool approx(float a, float b, float tol = 1e-5f) {
  return std::fabs(a - b) < tol;
}

static bool testExp() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(1.0);
  auto r = b.emitExp(x);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return approx(runAndReadFloat(compile(b)), std::exp(1.0f));
}

static bool testLog() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(2.718281828);
  auto r = b.emitLog(x);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return approx(runAndReadFloat(compile(b)), std::log(2.718281828f));
}

static bool testSqrt() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(9.0);
  auto r = b.emitSqrt(x);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return approx(runAndReadFloat(compile(b)), 3.0f);
}

static bool testRsqrt() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(4.0);
  auto r = b.emitRsqrt(x);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return approx(runAndReadFloat(compile(b)), 0.5f);
}

static bool testAbs() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(-5.0);
  auto r = b.emitAbs(x);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return approx(runAndReadFloat(compile(b)), 5.0f);
}

static bool testMax() {
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(3.0);
  auto y = b.emitFConst(7.0);
  auto r = b.emitMax(x, y);
  auto addr = b.emitConst(0);
  b.emitStore(addr, r);
  b.emitRet();
  return approx(runAndReadFloat(compile(b)), 7.0f);
}

int main() {
  struct { const char *name; bool (*fn)(); } tests[] = {
    {"f32_exp",   testExp},
    {"f32_log",   testLog},
    {"f32_sqrt",  testSqrt},
    {"f32_rsqrt", testRsqrt},
    {"f32_abs",   testAbs},
    {"f32_max",   testMax},
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
