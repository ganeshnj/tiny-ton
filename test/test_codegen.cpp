#include "tiny-ton/IR/Builder.h"
#include "tiny-ton/Compiler/RegisterAlloc.h"
#include "tiny-ton/Compiler/CodeGen.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

struct CodegenTest {
  const char *name;
  bool (*fn)();
};

static bool testConstEncoding() {
  // CONST R0, #5 -> opcode 0x9, rd=0, imm=5 -> 0x9005
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto v = b.emitConst(5);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());
  auto insts = tinyton::emit(b.getModule(), regMap);

  int rd = regMap[v];
  uint16_t expected = (0x9 << 12) | (rd << 8) | 5;
  if (insts[0].encoding != expected) {
    std::printf("    CONST: got 0x%04X, expected 0x%04X\n",
                insts[0].encoding, expected);
    return false;
  }
  return true;
}

static bool testAddEncoding() {
  // ADD Rd, Rs, Rt -> opcode 0x3, encodeRRR
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitConst(1);
  auto c = b.emitConst(2);
  auto sum = b.emitAdd(a, c);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());
  auto insts = tinyton::emit(b.getModule(), regMap);

  int rd = regMap[sum], rs = regMap[a], rt = regMap[c];
  uint16_t expected = (0x3 << 12) | (rd << 8) | (rs << 4) | rt;
  // The ADD instruction is the 3rd one (after two CONSTs)
  if (insts[2].encoding != expected) {
    std::printf("    ADD: got 0x%04X, expected 0x%04X\n",
                insts[2].encoding, expected);
    return false;
  }
  return true;
}

static bool testFaddEncoding() {
  // FADD Rd, Rs, Rt -> opcode 0x0, encodeRRR
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto a = b.emitFConst(1.0);
  auto c = b.emitFConst(2.0);
  auto sum = b.emitAdd(a, c);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());
  auto insts = tinyton::emit(b.getModule(), regMap);

  int rd = regMap[sum], rs = regMap[a], rt = regMap[c];
  uint16_t expected = (0x0 << 12) | (rd << 8) | (rs << 4) | rt;
  // FCONST is 3 words each, so FADD is at index 6
  if (insts[6].encoding != expected) {
    std::printf("    FADD: got 0x%04X, expected 0x%04X\n",
                insts[6].encoding, expected);
    return false;
  }
  return true;
}

static bool testLdrEncoding() {
  // LDR Rd, [Rs] -> opcode 0x7, encodeRRR(0x7, rd, rs, 0)
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto addr = b.emitConst(0);
  auto val = b.emitLoad(addr);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());
  auto insts = tinyton::emit(b.getModule(), regMap);

  int rd = regMap[val], rs = regMap[addr];
  uint16_t expected = (0x7 << 12) | (rd << 8) | (rs << 4) | 0;
  // LDR is at index 1 (after CONST)
  if (insts[1].encoding != expected) {
    std::printf("    LDR: got 0x%04X, expected 0x%04X\n",
                insts[1].encoding, expected);
    return false;
  }
  return true;
}

static bool testStrEncoding() {
  // STR [Rs], Rt -> opcode 0x8, encodeRRR(0x8, 0, rs, rt)
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto addr = b.emitConst(0);
  auto val = b.emitConst(42);
  b.emitStore(addr, val);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());
  auto insts = tinyton::emit(b.getModule(), regMap);

  int rs = regMap[addr], rt = regMap[val];
  uint16_t expected = (0x8 << 12) | (0 << 8) | (rs << 4) | rt;
  // STR is at index 2 (after two CONSTs)
  if (insts[2].encoding != expected) {
    std::printf("    STR: got 0x%04X, expected 0x%04X\n",
                insts[2].encoding, expected);
    return false;
  }
  return true;
}

static bool testRetEncoding() {
  // RET -> 0xF000
  tinyton::IRBuilder b;
  b.beginFunction("test");
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());
  auto insts = tinyton::emit(b.getModule(), regMap);

  if (insts[0].encoding != 0xF000) {
    std::printf("    RET: got 0x%04X, expected 0xF000\n", insts[0].encoding);
    return false;
  }
  return true;
}

static bool testFconstEncoding() {
  // FCONST is 3-word: opcode word + hi16 + lo16
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto v = b.emitFConst(1.0);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());
  auto insts = tinyton::emit(b.getModule(), regMap);

  assert(insts.size() >= 4 && "expected at least 4 instructions");

  int rd = regMap[v];
  uint16_t expectedOpWord = (0xE << 12) | (rd << 8) | 0x00;
  if (insts[0].encoding != expectedOpWord) {
    std::printf("    FCONST op word: got 0x%04X, expected 0x%04X\n",
                insts[0].encoding, expectedOpWord);
    return false;
  }

  float fval = 1.0f;
  uint32_t bits;
  std::memcpy(&bits, &fval, 4);
  uint16_t hi = (bits >> 16) & 0xFFFF;
  uint16_t lo = bits & 0xFFFF;
  if (insts[1].encoding != hi || insts[2].encoding != lo) {
    std::printf("    FCONST data: got 0x%04X 0x%04X, expected 0x%04X 0x%04X\n",
                insts[1].encoding, insts[2].encoding, hi, lo);
    return false;
  }
  return true;
}

static bool testExpEncoding() {
  // FEXP: extended opcode 0xE, sub-op 0x09, typeFlag=0 (f32)
  tinyton::IRBuilder b;
  b.beginFunction("test");
  auto x = b.emitFConst(1.0);
  auto r = b.emitExp(x);
  b.emitRet();

  auto regMap = tinyton::allocateRegisters(b.getModule());
  auto insts = tinyton::emit(b.getModule(), regMap);

  int rd = regMap[r], rs = regMap[x];
  // op word: encodeRI(0xE, rd, 0x09 << 4 | 0)
  uint16_t expected = (0xE << 12) | (rd << 8) | (0x09 << 4);
  // EXP is at index 3 (after 3-word FCONST)
  if (insts[3].encoding != expected) {
    std::printf("    EXP op word: got 0x%04X, expected 0x%04X\n",
                insts[3].encoding, expected);
    return false;
  }
  // operands word: rs << 4
  uint16_t expectedOps = (rs << 4);
  if (insts[4].encoding != expectedOps) {
    std::printf("    EXP operands: got 0x%04X, expected 0x%04X\n",
                insts[4].encoding, expectedOps);
    return false;
  }
  return true;
}

int main() {
  CodegenTest tests[] = {
    {"const_encoding", testConstEncoding},
    {"add_encoding", testAddEncoding},
    {"fadd_encoding", testFaddEncoding},
    {"ldr_encoding", testLdrEncoding},
    {"str_encoding", testStrEncoding},
    {"ret_encoding", testRetEncoding},
    {"fconst_encoding", testFconstEncoding},
    {"exp_encoding", testExpEncoding},
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
