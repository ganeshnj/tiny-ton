#include "tiny-ton/Runtime/Simulator.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace tinyton {

namespace {

inline float regToFloat(int32_t r) {
  float f;
  std::memcpy(&f, &r, 4);
  return f;
}

inline int32_t floatToReg(float f) {
  int32_t r;
  std::memcpy(&r, &f, 4);
  return r;
}

// --- Portable f16 <-> f32 conversion using IEEE 754 bit manipulation --------

inline float halfToFloat(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0) {
      // +/- zero
      uint32_t bits = sign;
      float f;
      std::memcpy(&f, &bits, 4);
      return f;
    }
    // Denormalized: convert to normalized f32
    exp = 1;
    while (!(mant & 0x400)) {
      mant <<= 1;
      exp--;
    }
    mant &= 0x3FF;
    uint32_t bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  if (exp == 0x1F) {
    // Inf or NaN
    uint32_t bits = sign | 0x7F800000 | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  // Normalized
  uint32_t bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline uint16_t floatToHalf(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (bits >> 13) & 0x3FF;

  if (((bits >> 23) & 0xFF) == 0xFF) {
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 : 0));
  }
  if (exp <= 0) {
    return static_cast<uint16_t>(sign);
  }
  if (exp >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00);
  }
  return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

// f16 stored in low 16 bits of int32 register
inline float regToHalf(int32_t r) {
  return halfToFloat(static_cast<uint16_t>(r & 0xFFFF));
}

inline int32_t halfToReg(float f) {
  return static_cast<int32_t>(floatToHalf(f));
}

} // namespace

struct SimulatedGPU::Impl {
  std::vector<uint16_t> program;
  std::vector<int32_t> memory;
  std::vector<int32_t> kernelArgs;
};

SimulatedGPU::SimulatedGPU(int memWords) : impl_(std::make_unique<Impl>()) {
  impl_->memory.resize(memWords, 0);
}

SimulatedGPU::~SimulatedGPU() = default;

void SimulatedGPU::loadProgram(const std::vector<uint16_t> &instructions) {
  impl_->program = instructions;
}

void SimulatedGPU::setArgs(const std::vector<int32_t> &args) {
  impl_->kernelArgs = args;
}

void SimulatedGPU::writeMemory(int addr, const std::vector<int32_t> &data) {
  for (size_t i = 0; i < data.size(); ++i) {
    int a = addr + static_cast<int>(i);
    assert(a >= 0 && a < (int)impl_->memory.size() && "write out of bounds");
    impl_->memory[a] = data[i];
  }
}

std::vector<int32_t> SimulatedGPU::readMemory(int addr, int count) const {
  std::vector<int32_t> result(count);
  for (int i = 0; i < count; ++i) {
    int a = addr + i;
    assert(a >= 0 && a < (int)impl_->memory.size() && "read out of bounds");
    result[i] = impl_->memory[a];
  }
  return result;
}

void SimulatedGPU::run(int numBlocks, int threadsPerBlock) {
  auto &prog = impl_->program;
  auto &mem = impl_->memory;
  auto &args = impl_->kernelArgs;

  for (int blockId = 0; blockId < numBlocks; ++blockId) {
    for (int threadId = 0; threadId < threadsPerBlock; ++threadId) {

      int32_t regs[16] = {};
      int pc = 0;
      constexpr int kMaxCycles = 100000;

      for (int cycle = 0; cycle < kMaxCycles; ++cycle) {
        assert(pc >= 0 && pc < (int)prog.size() && "PC out of bounds");

        uint16_t inst = prog[pc];
        uint8_t opcode = (inst >> 12) & 0xF;
        uint8_t rd = (inst >> 8) & 0xF;
        uint8_t rs = (inst >> 4) & 0xF;
        uint8_t rt = inst & 0xF;
        uint8_t imm = inst & 0xFF;

        switch (opcode) {
        case 0x0: { // FADD
          float a = regToFloat(regs[rs]);
          float b = regToFloat(regs[rt]);
          regs[rd] = floatToReg(a + b);
          break;
        }
        case 0x1: { // FSUB
          float a = regToFloat(regs[rs]);
          float b = regToFloat(regs[rt]);
          regs[rd] = floatToReg(a - b);
          break;
        }
        case 0x2: { // FMUL
          float a = regToFloat(regs[rs]);
          float b = regToFloat(regs[rt]);
          regs[rd] = floatToReg(a * b);
          break;
        }
        case 0x3: // ADD
          regs[rd] = regs[rs] + regs[rt];
          break;
        case 0x4: // SUB
          regs[rd] = regs[rs] - regs[rt];
          break;
        case 0x5: // MUL
          regs[rd] = regs[rs] * regs[rt];
          break;
        case 0x6: // CMP_LT
          regs[rd] = (regs[rs] < regs[rt]) ? 1 : 0;
          break;
        case 0x7: { // LDR: rd = mem[regs[rs]]
          int32_t addr = regs[rs];
          assert(addr >= 0 && addr < (int)mem.size() && "LDR out of bounds");
          regs[rd] = mem[addr];
          break;
        }
        case 0x8: { // STR: mem[regs[rs]] = regs[rt]
          int32_t addr = regs[rs];
          assert(addr >= 0 && addr < (int)mem.size() && "STR out of bounds");
          mem[addr] = regs[rt];
          break;
        }
        case 0x9: // CONST: rd = sign-extended imm8
          regs[rd] = static_cast<int32_t>(static_cast<int8_t>(imm));
          break;
        case 0xA: // ARG: rd = kernelArgs[imm]
          assert(imm < (int)args.size() && "ARG index out of bounds");
          regs[rd] = args[imm];
          break;
        case 0xB: // PID: rd = blockId
          regs[rd] = blockId;
          break;
        case 0xC: // TID: rd = threadId
          regs[rd] = threadId;
          break;
        case 0xD: // BZ: if regs[rd] == 0, skip imm instructions
          if (regs[rd] == 0) {
            pc += imm;
          }
          break;
        case 0xE: { // Extended ops
          uint8_t subOp = (imm >> 4) & 0xF;
          switch (subOp) {
          case 0x00: {
            // FCONST: next two words are hi16 and lo16 of IEEE 754 float
            assert(pc + 2 < (int)prog.size() && "FCONST: missing data words");
            uint32_t hi = static_cast<uint32_t>(prog[pc + 1]) << 16;
            uint32_t lo = static_cast<uint32_t>(prog[pc + 2]);
            uint32_t bits = hi | lo;
            std::memcpy(&regs[rd], &bits, 4);
            pc += 2;
            break;
          }
          case 0x01: {
            // FDIV: next word has rs|rt
            assert(pc + 1 < (int)prog.size() && "FDIV: missing operand word");
            uint16_t operands = prog[pc + 1];
            uint8_t frs = (operands >> 4) & 0xF;
            uint8_t frt = operands & 0xF;
            float a = regToFloat(regs[frs]);
            float b = regToFloat(regs[frt]);
            regs[rd] = floatToReg(a / b);
            pc += 1;
            break;
          }
          case 0x02: {
            // FCMP_LT: next word has rs|rt
            assert(pc + 1 < (int)prog.size() &&
                   "FCMP_LT: missing operand word");
            uint16_t operands = prog[pc + 1];
            uint8_t frs = (operands >> 4) & 0xF;
            uint8_t frt = operands & 0xF;
            float a = regToFloat(regs[frs]);
            float b = regToFloat(regs[frt]);
            regs[rd] = (a < b) ? 1 : 0;
            pc += 1;
            break;
          }
          case 0x03: {
            // HCONST: next word is the 16-bit f16 value
            assert(pc + 1 < (int)prog.size() && "HCONST: missing data word");
            uint16_t hbits = prog[pc + 1];
            regs[rd] = static_cast<int32_t>(hbits);
            pc += 1;
            break;
          }
          case 0x04: {
            // HADD: next word has rs|rt
            assert(pc + 1 < (int)prog.size() && "HADD: missing operand word");
            uint16_t operands = prog[pc + 1];
            uint8_t hrs = (operands >> 4) & 0xF;
            uint8_t hrt = operands & 0xF;
            float a = regToHalf(regs[hrs]);
            float b = regToHalf(regs[hrt]);
            regs[rd] = halfToReg(a + b);
            pc += 1;
            break;
          }
          case 0x05: {
            // HSUB: next word has rs|rt
            assert(pc + 1 < (int)prog.size() && "HSUB: missing operand word");
            uint16_t operands = prog[pc + 1];
            uint8_t hrs = (operands >> 4) & 0xF;
            uint8_t hrt = operands & 0xF;
            float a = regToHalf(regs[hrs]);
            float b = regToHalf(regs[hrt]);
            regs[rd] = halfToReg(a - b);
            pc += 1;
            break;
          }
          case 0x06: {
            // HMUL: next word has rs|rt
            assert(pc + 1 < (int)prog.size() && "HMUL: missing operand word");
            uint16_t operands = prog[pc + 1];
            uint8_t hrs = (operands >> 4) & 0xF;
            uint8_t hrt = operands & 0xF;
            float a = regToHalf(regs[hrs]);
            float b = regToHalf(regs[hrt]);
            regs[rd] = halfToReg(a * b);
            pc += 1;
            break;
          }
          case 0x07: {
            // HDIV: next word has rs|rt
            assert(pc + 1 < (int)prog.size() && "HDIV: missing operand word");
            uint16_t operands = prog[pc + 1];
            uint8_t hrs = (operands >> 4) & 0xF;
            uint8_t hrt = operands & 0xF;
            float a = regToHalf(regs[hrs]);
            float b = regToHalf(regs[hrt]);
            regs[rd] = halfToReg(a / b);
            pc += 1;
            break;
          }
          case 0x08: {
            // HCMP_LT: next word has rs|rt
            assert(pc + 1 < (int)prog.size() &&
                   "HCMP_LT: missing operand word");
            uint16_t operands = prog[pc + 1];
            uint8_t hrs = (operands >> 4) & 0xF;
            uint8_t hrt = operands & 0xF;
            float a = regToHalf(regs[hrs]);
            float b = regToHalf(regs[hrt]);
            regs[rd] = (a < b) ? 1 : 0;
            pc += 1;
            break;
          }
          default:
            break;
          }
          break;
        }
        case 0xF: // RET
          goto thread_done;
        default:
          break;
        }

        ++pc;
      }
    thread_done:;
    }
  }
}

} // namespace tinyton
