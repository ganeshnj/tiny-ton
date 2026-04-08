#include "tiny-ton/Compiler/CodeGen.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonOps.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

namespace tinyton {

static uint16_t encodeRRR(uint8_t opcode, uint8_t rd, uint8_t rs, uint8_t rt) {
  return (opcode << 12) | (rd << 8) | (rs << 4) | rt;
}

static uint16_t encodeRI(uint8_t opcode, uint8_t rd, uint8_t imm) {
  return (opcode << 12) | (rd << 8) | imm;
}

static uint32_t floatBits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return bits;
}

static uint16_t halfBits(float f) {
  // Convert f32 -> f16 bit pattern via manual IEEE 754 conversion.
  uint32_t fb = floatBits(f);
  uint32_t sign = (fb >> 16) & 0x8000;
  int32_t exp = ((fb >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (fb >> 13) & 0x3FF;

  if (((fb >> 23) & 0xFF) == 0xFF) {
    // Inf / NaN
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 : 0));
  }
  if (exp <= 0) {
    // Underflow -> zero
    return static_cast<uint16_t>(sign);
  }
  if (exp >= 0x1F) {
    // Overflow -> Inf
    return static_cast<uint16_t>(sign | 0x7C00);
  }
  return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

static bool isFloatResult(mlir::Value v) {
  return llvm::isa<mlir::FloatType>(v.getType());
}

static bool isF16Result(mlir::Value v) { return v.getType().isF16(); }

std::vector<Instruction> emit(mlir::ModuleOp module,
                              const RegisterMap &regMap) {
  std::vector<Instruction> instructions;

  auto getReg = [&](mlir::Value v) -> uint8_t {
    auto it = regMap.find(v);
    assert(it != regMap.end() && "value not in register map");
    return static_cast<uint8_t>(it->second);
  };

  for (auto &op : module.getBody()->getOperations()) {

    // --- Constants / Args / Thread ID ---

    if (auto constOp = llvm::dyn_cast<tinyton::ConstOp>(&op)) {
      uint8_t rd = getReg(constOp.getResult());
      uint8_t imm = static_cast<uint8_t>(constOp.getValue() & 0xFF);
      uint16_t enc = encodeRI(0x9, rd, imm);
      std::string asm_str =
          llvm::formatv("CONST R{0}, #{1}", (int)rd, (int)imm);
      instructions.push_back({enc, asm_str});

    } else if (auto fconstOp = llvm::dyn_cast<tinyton::FConstOp>(&op)) {
      uint8_t rd = getReg(fconstOp.getResult());
      float fval = fconstOp.getValue().convertToFloat();
      uint32_t bits = floatBits(fval);
      uint16_t hi = static_cast<uint16_t>((bits >> 16) & 0xFFFF);
      uint16_t lo = static_cast<uint16_t>(bits & 0xFFFF);

      // 3-word encoding: opcode word + hi16 + lo16
      uint16_t enc = encodeRI(0xE, rd, 0x00);
      std::string asm_str =
          llvm::formatv("FCONST R{0}, #{1}", (int)rd, fval);
      instructions.push_back({enc, asm_str});
      instructions.push_back({hi, llvm::formatv("  .hi 0x{0:X-4}", hi)});
      instructions.push_back({lo, llvm::formatv("  .lo 0x{0:X-4}", lo)});

    } else if (auto hconstOp = llvm::dyn_cast<tinyton::HConstOp>(&op)) {
      uint8_t rd = getReg(hconstOp.getResult());
      float fval = hconstOp.getValue().convertToFloat();
      uint16_t hbits = halfBits(fval);

      // 2-word encoding: opcode word (sub-op 0x03) + 16-bit f16 value
      uint16_t enc = encodeRI(0xE, rd, 0x03 << 4);
      std::string asm_str =
          llvm::formatv("HCONST R{0}, #{1}", (int)rd, fval);
      instructions.push_back({enc, asm_str});
      instructions.push_back(
          {hbits, llvm::formatv("  .f16 0x{0:X-4}", hbits)});

    } else if (auto argOp = llvm::dyn_cast<tinyton::ArgOp>(&op)) {
      uint8_t rd = getReg(argOp.getResult());
      uint8_t imm = static_cast<uint8_t>(argOp.getIndex() & 0xFF);
      uint16_t enc = encodeRI(0xA, rd, imm);
      std::string asm_str =
          llvm::formatv("ARG R{0}, #{1}", (int)rd, (int)imm);
      instructions.push_back({enc, asm_str});

    } else if (auto pidOp = llvm::dyn_cast<tinyton::ProgramIdOp>(&op)) {
      uint8_t rd = getReg(pidOp.getResult());
      uint8_t imm = static_cast<uint8_t>(pidOp.getAxis() & 0xFF);
      uint16_t enc = encodeRI(0xB, rd, imm);
      std::string asm_str =
          llvm::formatv("PID R{0}, #{1}", (int)rd, (int)imm);
      instructions.push_back({enc, asm_str});

    } else if (auto tidOp = llvm::dyn_cast<tinyton::ThreadIdOp>(&op)) {
      uint8_t rd = getReg(tidOp.getResult());
      uint8_t imm = static_cast<uint8_t>(tidOp.getAxis() & 0xFF);
      uint16_t enc = encodeRI(0xC, rd, imm);
      std::string asm_str =
          llvm::formatv("TID R{0}, #{1}", (int)rd, (int)imm);
      instructions.push_back({enc, asm_str});

    // --- Arithmetic (integer and float) ---

    } else if (auto addOp = llvm::dyn_cast<tinyton::AddOp>(&op)) {
      uint8_t rd = getReg(addOp.getResult());
      uint8_t rs = getReg(addOp.getLhs());
      uint8_t rt = getReg(addOp.getRhs());

      if (isF16Result(addOp.getResult())) {
        // HADD: extended opcode 0xE, sub-op 0x04
        uint16_t enc = encodeRI(0xE, rd, 0x04 << 4);
        uint16_t operands = (rs << 4) | rt;
        instructions.push_back(
            {enc, llvm::formatv("HADD R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
        instructions.push_back(
            {operands,
             llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});
      } else if (isFloatResult(addOp.getResult())) {
        uint16_t enc = encodeRRR(0x0, rd, rs, rt);
        instructions.push_back(
            {enc, llvm::formatv("FADD R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
      } else {
        uint16_t enc = encodeRRR(0x3, rd, rs, rt);
        instructions.push_back(
            {enc, llvm::formatv("ADD R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
      }

    } else if (auto subOp = llvm::dyn_cast<tinyton::SubOp>(&op)) {
      uint8_t rd = getReg(subOp.getResult());
      uint8_t rs = getReg(subOp.getLhs());
      uint8_t rt = getReg(subOp.getRhs());

      if (isF16Result(subOp.getResult())) {
        uint16_t enc = encodeRI(0xE, rd, 0x05 << 4);
        uint16_t operands = (rs << 4) | rt;
        instructions.push_back(
            {enc, llvm::formatv("HSUB R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
        instructions.push_back(
            {operands,
             llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});
      } else if (isFloatResult(subOp.getResult())) {
        uint16_t enc = encodeRRR(0x1, rd, rs, rt);
        instructions.push_back(
            {enc, llvm::formatv("FSUB R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
      } else {
        uint16_t enc = encodeRRR(0x4, rd, rs, rt);
        instructions.push_back(
            {enc, llvm::formatv("SUB R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
      }

    } else if (auto mulOp = llvm::dyn_cast<tinyton::MulOp>(&op)) {
      uint8_t rd = getReg(mulOp.getResult());
      uint8_t rs = getReg(mulOp.getLhs());
      uint8_t rt = getReg(mulOp.getRhs());

      if (isF16Result(mulOp.getResult())) {
        uint16_t enc = encodeRI(0xE, rd, 0x06 << 4);
        uint16_t operands = (rs << 4) | rt;
        instructions.push_back(
            {enc, llvm::formatv("HMUL R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
        instructions.push_back(
            {operands,
             llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});
      } else if (isFloatResult(mulOp.getResult())) {
        uint16_t enc = encodeRRR(0x2, rd, rs, rt);
        instructions.push_back(
            {enc, llvm::formatv("FMUL R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
      } else {
        uint16_t enc = encodeRRR(0x5, rd, rs, rt);
        instructions.push_back(
            {enc, llvm::formatv("MUL R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
      }

    } else if (auto divOp = llvm::dyn_cast<tinyton::DivOp>(&op)) {
      uint8_t rd = getReg(divOp.getResult());
      uint8_t rs = getReg(divOp.getLhs());
      uint8_t rt = getReg(divOp.getRhs());

      if (isF16Result(divOp.getResult())) {
        // HDIV: extended opcode 0xE, sub-op 0x07
        uint16_t enc = encodeRI(0xE, rd, 0x07 << 4);
        uint16_t operands = (rs << 4) | rt;
        instructions.push_back(
            {enc, llvm::formatv("HDIV R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
        instructions.push_back(
            {operands,
             llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});
      } else {
        // FDIV / DIV: sub-op 0x01
        std::string mnemonic = isFloatResult(divOp.getResult()) ? "FDIV" : "DIV";
        uint16_t enc = encodeRI(0xE, rd, (0x01 << 4) | 0x0);
        uint16_t operands = (rs << 4) | rt;
        instructions.push_back(
            {enc, llvm::formatv("{0} R{1}, R{2}, R{3}", mnemonic, (int)rd,
                                (int)rs, (int)rt)});
        instructions.push_back(
            {operands,
             llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});
      }

    // --- Comparison ---

    } else if (auto cmpOp = llvm::dyn_cast<tinyton::CmpLtOp>(&op)) {
      uint8_t rd = getReg(cmpOp.getResult());
      uint8_t rs = getReg(cmpOp.getLhs());
      uint8_t rt = getReg(cmpOp.getRhs());

      if (isF16Result(cmpOp.getLhs())) {
        // HCMP_LT: extended opcode 0xE, sub-op 0x08
        uint16_t enc = encodeRI(0xE, rd, 0x08 << 4);
        uint16_t operands = (rs << 4) | rt;
        instructions.push_back(
            {enc, llvm::formatv("HCMP_LT R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
        instructions.push_back(
            {operands,
             llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});
      } else if (isFloatResult(cmpOp.getLhs())) {
        // FCMP_LT: sub-op 0x02
        uint16_t enc = encodeRI(0xE, rd, 0x02 << 4);
        uint16_t operands = (rs << 4) | rt;
        instructions.push_back(
            {enc, llvm::formatv("FCMP_LT R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
        instructions.push_back(
            {operands,
             llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});
      } else {
        uint16_t enc = encodeRRR(0x6, rd, rs, rt);
        std::string asm_str = llvm::formatv("CMP_LT R{0}, R{1}, R{2}",
                                            (int)rd, (int)rs, (int)rt);
        instructions.push_back({enc, asm_str});
      }

    // --- Math intrinsics ---

    } else if (auto expOp = llvm::dyn_cast<tinyton::ExpOp>(&op)) {
      uint8_t rd = getReg(expOp.getResult());
      uint8_t rs = getReg(expOp.getOperand());
      uint8_t typeFlag = isF16Result(expOp.getResult()) ? 1 : 0;
      uint16_t enc = encodeRI(0xE, rd, 0x09 << 4 | typeFlag);
      uint16_t operands = (rs << 4);
      std::string pfx = typeFlag ? "H" : "F";
      instructions.push_back(
          {enc, llvm::formatv("{0}EXP R{1}, R{2}", pfx, (int)rd, (int)rs)});
      instructions.push_back(
          {operands, llvm::formatv("  .operands R{0}", (int)rs)});

    } else if (auto logOp = llvm::dyn_cast<tinyton::LogOp>(&op)) {
      uint8_t rd = getReg(logOp.getResult());
      uint8_t rs = getReg(logOp.getOperand());
      uint8_t typeFlag = isF16Result(logOp.getResult()) ? 1 : 0;
      uint16_t enc = encodeRI(0xE, rd, 0x0A << 4 | typeFlag);
      uint16_t operands = (rs << 4);
      std::string pfx = typeFlag ? "H" : "F";
      instructions.push_back(
          {enc, llvm::formatv("{0}LOG R{1}, R{2}", pfx, (int)rd, (int)rs)});
      instructions.push_back(
          {operands, llvm::formatv("  .operands R{0}", (int)rs)});

    } else if (auto sqrtOp = llvm::dyn_cast<tinyton::SqrtOp>(&op)) {
      uint8_t rd = getReg(sqrtOp.getResult());
      uint8_t rs = getReg(sqrtOp.getOperand());
      uint8_t typeFlag = isF16Result(sqrtOp.getResult()) ? 1 : 0;
      uint16_t enc = encodeRI(0xE, rd, 0x0B << 4 | typeFlag);
      uint16_t operands = (rs << 4);
      std::string pfx = typeFlag ? "H" : "F";
      instructions.push_back(
          {enc, llvm::formatv("{0}SQRT R{1}, R{2}", pfx, (int)rd, (int)rs)});
      instructions.push_back(
          {operands, llvm::formatv("  .operands R{0}", (int)rs)});

    } else if (auto rsqrtOp = llvm::dyn_cast<tinyton::RsqrtOp>(&op)) {
      uint8_t rd = getReg(rsqrtOp.getResult());
      uint8_t rs = getReg(rsqrtOp.getOperand());
      uint8_t typeFlag = isF16Result(rsqrtOp.getResult()) ? 1 : 0;
      uint16_t enc = encodeRI(0xE, rd, 0x0C << 4 | typeFlag);
      uint16_t operands = (rs << 4);
      std::string pfx = typeFlag ? "H" : "F";
      instructions.push_back(
          {enc,
           llvm::formatv("{0}RSQRT R{1}, R{2}", pfx, (int)rd, (int)rs)});
      instructions.push_back(
          {operands, llvm::formatv("  .operands R{0}", (int)rs)});

    } else if (auto absOp = llvm::dyn_cast<tinyton::AbsOp>(&op)) {
      uint8_t rd = getReg(absOp.getResult());
      uint8_t rs = getReg(absOp.getOperand());
      uint8_t typeFlag = isF16Result(absOp.getResult())    ? 1
                         : isFloatResult(absOp.getResult()) ? 0
                                                            : 2;
      uint16_t enc = encodeRI(0xE, rd, 0x0D << 4 | typeFlag);
      uint16_t operands = (rs << 4);
      std::string pfx = typeFlag == 1 ? "H" : typeFlag == 0 ? "F" : "";
      instructions.push_back(
          {enc, llvm::formatv("{0}ABS R{1}, R{2}", pfx, (int)rd, (int)rs)});
      instructions.push_back(
          {operands, llvm::formatv("  .operands R{0}", (int)rs)});

    } else if (auto maxOp = llvm::dyn_cast<tinyton::MaxOp>(&op)) {
      uint8_t rd = getReg(maxOp.getResult());
      uint8_t rs = getReg(maxOp.getLhs());
      uint8_t rt = getReg(maxOp.getRhs());
      uint8_t typeFlag = isF16Result(maxOp.getResult())    ? 1
                         : isFloatResult(maxOp.getResult()) ? 0
                                                            : 2;
      uint16_t enc = encodeRI(0xE, rd, 0x0E << 4 | typeFlag);
      uint16_t operands = (rs << 4) | rt;
      std::string pfx = typeFlag == 1 ? "H" : typeFlag == 0 ? "F" : "";
      instructions.push_back(
          {enc, llvm::formatv("{0}MAX R{1}, R{2}, R{3}", pfx, (int)rd,
                              (int)rs, (int)rt)});
      instructions.push_back(
          {operands,
           llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});

    // --- Reductions ---

    } else if (auto reduceSumOp = llvm::dyn_cast<tinyton::ReduceSumOp>(&op)) {
      uint8_t rd = getReg(reduceSumOp.getResult());
      uint8_t rs = getReg(reduceSumOp.getOperand());
      uint8_t typeFlag = isF16Result(reduceSumOp.getResult())    ? 1
                         : isFloatResult(reduceSumOp.getResult()) ? 0
                                                                  : 2;
      // sub-op 0x0F, lower nibble: bit2=0 (sum) | typeFlag
      uint16_t enc = encodeRI(0xE, rd, 0x0F << 4 | typeFlag);
      uint16_t operands = (rs << 4);
      std::string pfx = typeFlag == 1 ? "H" : typeFlag == 0 ? "F" : "";
      instructions.push_back(
          {enc, llvm::formatv("{0}REDUCE_SUM R{1}, R{2}", pfx, (int)rd,
                              (int)rs)});
      instructions.push_back(
          {operands, llvm::formatv("  .operands R{0}", (int)rs)});

    } else if (auto reduceMaxOp = llvm::dyn_cast<tinyton::ReduceMaxOp>(&op)) {
      uint8_t rd = getReg(reduceMaxOp.getResult());
      uint8_t rs = getReg(reduceMaxOp.getOperand());
      uint8_t typeFlag = isF16Result(reduceMaxOp.getResult())    ? 1
                         : isFloatResult(reduceMaxOp.getResult()) ? 0
                                                                  : 2;
      // sub-op 0x0F, lower nibble: bit2=1 (max) | typeFlag
      uint16_t enc = encodeRI(0xE, rd, 0x0F << 4 | (4 | typeFlag));
      uint16_t operands = (rs << 4);
      std::string pfx = typeFlag == 1 ? "H" : typeFlag == 0 ? "F" : "";
      instructions.push_back(
          {enc, llvm::formatv("{0}REDUCE_MAX R{1}, R{2}", pfx, (int)rd,
                              (int)rs)});
      instructions.push_back(
          {operands, llvm::formatv("  .operands R{0}", (int)rs)});

    // --- Memory ---

    } else if (auto loadOp = llvm::dyn_cast<tinyton::LoadOp>(&op)) {
      uint8_t rd = getReg(loadOp.getResult());
      uint8_t rs = getReg(loadOp.getAddr());
      bool resIsF16 = isF16Result(loadOp.getResult());
      bool resIsFloat = isFloatResult(loadOp.getResult());

      if (loadOp.getMask()) {
        uint8_t rm = getReg(loadOp.getMask());
        if (loadOp.getOther()) {
          uint8_t ro = getReg(loadOp.getOther());
          instructions.push_back(
              {encodeRRR(0x1, rd, ro, 0),
               llvm::formatv("MOV R{0}, R{1}", (int)rd, (int)ro).str()});
        } else if (resIsF16) {
          uint16_t hbits = halfBits(0.0f);
          instructions.push_back(
              {encodeRI(0xE, rd, 0x03 << 4),
               llvm::formatv("HCONST R{0}, #0.0", (int)rd).str()});
          instructions.push_back({hbits, "  .f16 0x0000"});
        } else if (resIsFloat) {
          uint32_t zeroBits = floatBits(0.0f);
          uint16_t hi = static_cast<uint16_t>((zeroBits >> 16) & 0xFFFF);
          uint16_t lo = static_cast<uint16_t>(zeroBits & 0xFFFF);
          instructions.push_back(
              {encodeRI(0xE, rd, 0x00),
               llvm::formatv("FCONST R{0}, #0.0", (int)rd).str()});
          instructions.push_back({hi, "  .hi 0x0000"});
          instructions.push_back({lo, "  .lo 0x0000"});
        } else {
          instructions.push_back(
              {encodeRI(0x9, rd, 0),
               llvm::formatv("CONST R{0}, #0", (int)rd).str()});
        }
        instructions.push_back(
            {encodeRI(0xD, rm, 1),
             llvm::formatv("BZ R{0}, #1", (int)rm).str()});
      }

      uint16_t enc = encodeRRR(0x7, rd, rs, 0);
      std::string asm_str =
          llvm::formatv("LDR R{0}, [R{1}]", (int)rd, (int)rs);
      instructions.push_back({enc, asm_str});

    } else if (auto storeOp = llvm::dyn_cast<tinyton::StoreOp>(&op)) {
      uint8_t rs = getReg(storeOp.getAddr());
      uint8_t rt = getReg(storeOp.getValue());

      if (storeOp.getMask()) {
        uint8_t rm = getReg(storeOp.getMask());
        instructions.push_back(
            {encodeRI(0xD, rm, 1),
             llvm::formatv("BZ R{0}, #1", (int)rm).str()});
      }

      uint16_t enc = encodeRRR(0x8, 0, rs, rt);
      std::string asm_str =
          llvm::formatv("STR [R{0}], R{1}", (int)rs, (int)rt);
      instructions.push_back({enc, asm_str});

    // --- Control flow ---

    } else if (auto bzOp = llvm::dyn_cast<tinyton::BranchZeroOp>(&op)) {
      uint8_t rs = getReg(bzOp.getCond());
      uint8_t imm = static_cast<uint8_t>(bzOp.getSkip() & 0xFF);
      uint16_t enc = encodeRI(0xD, rs, imm);
      std::string asm_str =
          llvm::formatv("BZ R{0}, #{1}", (int)rs, (int)imm);
      instructions.push_back({enc, asm_str});

    } else if (llvm::isa<tinyton::RetOp>(&op)) {
      instructions.push_back({0xF000, "RET"});

    } else if (llvm::isa<mlir::ModuleOp>(&op)) {
      // skip nested module ops
    }
  }

  return instructions;
}

} // namespace tinyton
