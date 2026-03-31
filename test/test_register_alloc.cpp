#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/Compiler/RegisterAlloc.h"

#include <cassert>
#include <cstdio>
#include <set>

static bool testDistinctRegisters() {
  // 3 live values at the same time -> 3 distinct registers.
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitConst(1);
  auto c = b.emitConst(2);
  auto d = b.emitConst(3);
  auto sum1 = b.emitAdd(a, c);
  auto sum2 = b.emitAdd(sum1, d);
  auto addr = b.emitConst(0);
  b.emitStore(addr, sum2);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());

  int ra = regMap[a], rc = regMap[c], rd = regMap[d];
  std::set<int> regs = {ra, rc, rd};
  if (regs.size() != 3) {
    std::printf("    expected 3 distinct regs, got %zu\n", regs.size());
    return false;
  }
  return true;
}

static bool testRegisterReuse() {
  // Value `a` is consumed by the first add, so its register should be
  // reusable by a later value.
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitConst(1);
  auto c = b.emitConst(2);
  auto sum1 = b.emitAdd(a, c);  // last use of a
  auto d = b.emitConst(3);       // can reuse a's register
  auto sum2 = b.emitAdd(sum1, d);
  auto addr = b.emitConst(0);
  b.emitStore(addr, sum2);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());

  bool allAssigned = regMap.count(a) && regMap.count(c) &&
                     regMap.count(sum1) && regMap.count(d) &&
                     regMap.count(sum2) && regMap.count(addr);
  if (!allAssigned) {
    std::printf("    not all values assigned registers\n");
    return false;
  }
  return true;
}

static bool testHighPressure() {
  // Build a chain of 20 additions. Each step emits a new const and adds it
  // to the accumulator. The accumulator + new const are the only live values
  // at any add, so registers freed after each add should be reused.
  tinyton::IRBuilder b;
  b.beginFunction("test");

  auto acc = b.emitConst(0);
  for (int i = 1; i <= 20; ++i) {
    auto v = b.emitConst(i);
    acc = b.emitAdd(acc, v);
  }

  auto addr = b.emitConst(0);
  b.emitStore(addr, acc);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());

  for (auto &kv : regMap) {
    if (kv.second < 0 || kv.second > 12) {
      std::printf("    register %d out of range [0,12]\n", kv.second);
      return false;
    }
  }
  return true;
}

int main() {
  struct { const char *name; bool (*fn)(); } tests[] = {
    {"distinct_registers", testDistinctRegisters},
    {"register_reuse", testRegisterReuse},
    {"high_pressure", testHighPressure},
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
