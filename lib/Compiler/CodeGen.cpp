#include "tiny-ton/Compiler/CodeGen.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonOps.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace tinyton {

static uint16_t encodeRRR(uint8_t opcode, uint8_t rd, uint8_t rs, uint8_t rt) {
  return (opcode << 12) | (rd << 8) | (rs << 4) | rt;
}

static uint16_t encodeRI(uint8_t opcode, uint8_t rd, uint8_t imm) {
  return (opcode << 12) | (rd << 8) | imm;
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

    // --- Arithmetic ---

    } else if (auto addOp = llvm::dyn_cast<tinyton::AddOp>(&op)) {
      uint8_t rd = getReg(addOp.getResult());
      uint8_t rs = getReg(addOp.getLhs());
      uint8_t rt = getReg(addOp.getRhs());
      uint16_t enc = encodeRRR(0x3, rd, rs, rt);
      std::string asm_str =
          llvm::formatv("ADD R{0}, R{1}, R{2}", (int)rd, (int)rs, (int)rt);
      instructions.push_back({enc, asm_str});

    } else if (auto subOp = llvm::dyn_cast<tinyton::SubOp>(&op)) {
      uint8_t rd = getReg(subOp.getResult());
      uint8_t rs = getReg(subOp.getLhs());
      uint8_t rt = getReg(subOp.getRhs());
      uint16_t enc = encodeRRR(0x4, rd, rs, rt);
      std::string asm_str =
          llvm::formatv("SUB R{0}, R{1}, R{2}", (int)rd, (int)rs, (int)rt);
      instructions.push_back({enc, asm_str});

    } else if (auto mulOp = llvm::dyn_cast<tinyton::MulOp>(&op)) {
      uint8_t rd = getReg(mulOp.getResult());
      uint8_t rs = getReg(mulOp.getLhs());
      uint8_t rt = getReg(mulOp.getRhs());
      uint16_t enc = encodeRRR(0x5, rd, rs, rt);
      std::string asm_str =
          llvm::formatv("MUL R{0}, R{1}, R{2}", (int)rd, (int)rs, (int)rt);
      instructions.push_back({enc, asm_str});

    // --- Comparison ---

    } else if (auto cmpOp = llvm::dyn_cast<tinyton::CmpLtOp>(&op)) {
      uint8_t rd = getReg(cmpOp.getResult());
      uint8_t rs = getReg(cmpOp.getLhs());
      uint8_t rt = getReg(cmpOp.getRhs());
      uint16_t enc = encodeRRR(0x6, rd, rs, rt);
      std::string asm_str =
          llvm::formatv("CMP_LT R{0}, R{1}, R{2}", (int)rd, (int)rs, (int)rt);
      instructions.push_back({enc, asm_str});

    // --- Memory ---

    } else if (auto loadOp = llvm::dyn_cast<tinyton::LoadOp>(&op)) {
      uint8_t rd = getReg(loadOp.getResult());
      uint8_t rs = getReg(loadOp.getAddr());

      if (loadOp.getMask()) {
        // Masked load: CONST rd,0 ; BZ mask,1 ; LDR rd,rs
        uint8_t rm = getReg(loadOp.getMask());
        instructions.push_back(
            {encodeRI(0x9, rd, 0),
             llvm::formatv("CONST R{0}, #0", (int)rd).str()});
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
        // Masked store: BZ mask,1 ; STR [rs],rt
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
