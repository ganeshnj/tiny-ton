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

    // --- Integer Arithmetic ---

    } else if (auto addOp = llvm::dyn_cast<tinyton::AddOp>(&op)) {
      uint8_t rd = getReg(addOp.getResult());
      uint8_t rs = getReg(addOp.getLhs());
      uint8_t rt = getReg(addOp.getRhs());
      bool isFloat = addOp.getResult().getType().isF32();
      uint8_t opc = isFloat ? 0x0 : 0x3;
      std::string mnemonic = isFloat ? "FADD" : "ADD";
      uint16_t enc = encodeRRR(opc, rd, rs, rt);
      std::string asm_str =
          llvm::formatv("{0} R{1}, R{2}, R{3}", mnemonic, (int)rd, (int)rs,
                        (int)rt);
      instructions.push_back({enc, asm_str});

    } else if (auto subOp = llvm::dyn_cast<tinyton::SubOp>(&op)) {
      uint8_t rd = getReg(subOp.getResult());
      uint8_t rs = getReg(subOp.getLhs());
      uint8_t rt = getReg(subOp.getRhs());
      bool isFloat = subOp.getResult().getType().isF32();
      uint8_t opc = isFloat ? 0x1 : 0x4;
      std::string mnemonic = isFloat ? "FSUB" : "SUB";
      uint16_t enc = encodeRRR(opc, rd, rs, rt);
      std::string asm_str =
          llvm::formatv("{0} R{1}, R{2}, R{3}", mnemonic, (int)rd, (int)rs,
                        (int)rt);
      instructions.push_back({enc, asm_str});

    } else if (auto mulOp = llvm::dyn_cast<tinyton::MulOp>(&op)) {
      uint8_t rd = getReg(mulOp.getResult());
      uint8_t rs = getReg(mulOp.getLhs());
      uint8_t rt = getReg(mulOp.getRhs());
      bool isFloat = mulOp.getResult().getType().isF32();
      uint8_t opc = isFloat ? 0x2 : 0x5;
      std::string mnemonic = isFloat ? "FMUL" : "MUL";
      uint16_t enc = encodeRRR(opc, rd, rs, rt);
      std::string asm_str =
          llvm::formatv("{0} R{1}, R{2}, R{3}", mnemonic, (int)rd, (int)rs,
                        (int)rt);
      instructions.push_back({enc, asm_str});

    } else if (auto divOp = llvm::dyn_cast<tinyton::DivOp>(&op)) {
      uint8_t rd = getReg(divOp.getResult());
      uint8_t rs = getReg(divOp.getLhs());
      uint8_t rt = getReg(divOp.getRhs());
      // FDIV uses opcode 0xE with sub-opcode 0x01 in imm field
      bool isFloat = divOp.getResult().getType().isF32();
      std::string mnemonic = isFloat ? "FDIV" : "DIV";
      // Encode as: 0xE_rd_01 for float div, re-use with rs/rt in next word
      // Simpler: use opcode 0xE, rd field, and pack rs|rt in low byte
      uint16_t enc = encodeRI(0xE, rd, (0x01 << 4) | 0x0);
      uint16_t operands = (rs << 4) | rt;
      instructions.push_back(
          {enc, llvm::formatv("{0} R{1}, R{2}, R{3}", mnemonic, (int)rd,
                              (int)rs, (int)rt)});
      instructions.push_back(
          {operands, llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});

    // --- Comparison ---

    } else if (auto cmpOp = llvm::dyn_cast<tinyton::CmpLtOp>(&op)) {
      uint8_t rd = getReg(cmpOp.getResult());
      uint8_t rs = getReg(cmpOp.getLhs());
      uint8_t rt = getReg(cmpOp.getRhs());
      bool isFloat = cmpOp.getLhs().getType().isF32();
      if (isFloat) {
        // FCMP_LT: opcode 0xE, sub-opcode 0x02
        uint16_t enc = encodeRI(0xE, rd, 0x02 << 4);
        uint16_t operands = (rs << 4) | rt;
        instructions.push_back(
            {enc, llvm::formatv("FCMP_LT R{0}, R{1}, R{2}", (int)rd, (int)rs,
                                (int)rt)});
        instructions.push_back(
            {operands, llvm::formatv("  .operands R{0}, R{1}", (int)rs, (int)rt)});
      } else {
        uint16_t enc = encodeRRR(0x6, rd, rs, rt);
        std::string asm_str = llvm::formatv("CMP_LT R{0}, R{1}, R{2}", (int)rd,
                                            (int)rs, (int)rt);
        instructions.push_back({enc, asm_str});
      }

    // --- Memory ---

    } else if (auto loadOp = llvm::dyn_cast<tinyton::LoadOp>(&op)) {
      uint8_t rd = getReg(loadOp.getResult());
      uint8_t rs = getReg(loadOp.getAddr());

      if (loadOp.getMask()) {
        uint8_t rm = getReg(loadOp.getMask());
        if (loadOp.getResult().getType().isF32()) {
          // Masked float load: FCONST rd,0.0 ; BZ mask,1 ; LDR rd,[rs]
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
