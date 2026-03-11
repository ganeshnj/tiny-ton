#include "tiny-ton/Runtime/Simulator.h"

#include <cassert>
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
          if (subOp == 0x00) {
            // FCONST: next two words are hi16 and lo16 of IEEE 754 float
            assert(pc + 2 < (int)prog.size() && "FCONST: missing data words");
            uint32_t hi = static_cast<uint32_t>(prog[pc + 1]) << 16;
            uint32_t lo = static_cast<uint32_t>(prog[pc + 2]);
            uint32_t bits = hi | lo;
            std::memcpy(&regs[rd], &bits, 4);
            pc += 2; // skip the two data words
          } else if (subOp == 0x01) {
            // FDIV: next word has rs|rt
            assert(pc + 1 < (int)prog.size() && "FDIV: missing operand word");
            uint16_t operands = prog[pc + 1];
            uint8_t frs = (operands >> 4) & 0xF;
            uint8_t frt = operands & 0xF;
            float a = regToFloat(regs[frs]);
            float b = regToFloat(regs[frt]);
            regs[rd] = floatToReg(a / b);
            pc += 1;
          } else if (subOp == 0x02) {
            // FCMP_LT: next word has rs|rt
            assert(pc + 1 < (int)prog.size() && "FCMP_LT: missing operand word");
            uint16_t operands = prog[pc + 1];
            uint8_t frs = (operands >> 4) & 0xF;
            uint8_t frt = operands & 0xF;
            float a = regToFloat(regs[frs]);
            float b = regToFloat(regs[frt]);
            regs[rd] = (a < b) ? 1 : 0;
            pc += 1;
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
