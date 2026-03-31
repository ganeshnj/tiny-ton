#include "tiny-ton/Runtime/Simulator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
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

  struct ThreadState {
    int32_t regs[16] = {};
    int pc = 0;
    bool done = false;
  };

  enum class StepResult { Continue, Reduce, Done };

  // Executes one instruction, returns whether to continue, pause for reduce,
  // or stop. For REDUCE, reduceOperand receives the thread's contribution,
  // reduceRd/reduceFlags are set for the outer loop.
  auto execOne = [&](ThreadState &ts, int blockId, int threadId,
                     int32_t &reduceOperand, uint8_t &reduceRd,
                     uint8_t &reduceFlags) -> StepResult {
    assert(ts.pc >= 0 && ts.pc < (int)prog.size() && "PC out of bounds");

    uint16_t inst = prog[ts.pc];
    uint8_t opcode = (inst >> 12) & 0xF;
    uint8_t rd = (inst >> 8) & 0xF;
    uint8_t rs = (inst >> 4) & 0xF;
    uint8_t rt = inst & 0xF;
    uint8_t imm = inst & 0xFF;

    switch (opcode) {
    case 0x0: { // FADD
      float a = regToFloat(ts.regs[rs]);
      float b = regToFloat(ts.regs[rt]);
      ts.regs[rd] = floatToReg(a + b);
      break;
    }
    case 0x1: { // FSUB
      float a = regToFloat(ts.regs[rs]);
      float b = regToFloat(ts.regs[rt]);
      ts.regs[rd] = floatToReg(a - b);
      break;
    }
    case 0x2: { // FMUL
      float a = regToFloat(ts.regs[rs]);
      float b = regToFloat(ts.regs[rt]);
      ts.regs[rd] = floatToReg(a * b);
      break;
    }
    case 0x3: // ADD
      ts.regs[rd] = ts.regs[rs] + ts.regs[rt];
      break;
    case 0x4: // SUB
      ts.regs[rd] = ts.regs[rs] - ts.regs[rt];
      break;
    case 0x5: // MUL
      ts.regs[rd] = ts.regs[rs] * ts.regs[rt];
      break;
    case 0x6: // CMP_LT
      ts.regs[rd] = (ts.regs[rs] < ts.regs[rt]) ? 1 : 0;
      break;
    case 0x7: { // LDR
      int32_t addr = ts.regs[rs];
      assert(addr >= 0 && addr < (int)mem.size() && "LDR out of bounds");
      ts.regs[rd] = mem[addr];
      break;
    }
    case 0x8: { // STR
      int32_t addr = ts.regs[rs];
      assert(addr >= 0 && addr < (int)mem.size() && "STR out of bounds");
      mem[addr] = ts.regs[rt];
      break;
    }
    case 0x9:
      ts.regs[rd] = static_cast<int32_t>(static_cast<int8_t>(imm));
      break;
    case 0xA:
      assert(imm < (int)args.size() && "ARG index out of bounds");
      ts.regs[rd] = args[imm];
      break;
    case 0xB:
      ts.regs[rd] = blockId;
      break;
    case 0xC:
      ts.regs[rd] = threadId;
      break;
    case 0xD:
      if (ts.regs[rd] == 0) {
        ts.pc += imm;
      }
      break;
    case 0xE: {
      uint8_t subOp = (imm >> 4) & 0xF;
      switch (subOp) {
      case 0x00: {
        assert(ts.pc + 2 < (int)prog.size() && "FCONST: missing data words");
        uint32_t hi = static_cast<uint32_t>(prog[ts.pc + 1]) << 16;
        uint32_t lo = static_cast<uint32_t>(prog[ts.pc + 2]);
        uint32_t bits = hi | lo;
        std::memcpy(&ts.regs[rd], &bits, 4);
        ts.pc += 2;
        break;
      }
      case 0x01: {
        assert(ts.pc + 1 < (int)prog.size() && "FDIV: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t frs = (operands >> 4) & 0xF;
        uint8_t frt = operands & 0xF;
        float a = regToFloat(ts.regs[frs]);
        float b = regToFloat(ts.regs[frt]);
        ts.regs[rd] = floatToReg(a / b);
        ts.pc += 1;
        break;
      }
      case 0x02: {
        assert(ts.pc + 1 < (int)prog.size() &&
               "FCMP_LT: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t frs = (operands >> 4) & 0xF;
        uint8_t frt = operands & 0xF;
        float a = regToFloat(ts.regs[frs]);
        float b = regToFloat(ts.regs[frt]);
        ts.regs[rd] = (a < b) ? 1 : 0;
        ts.pc += 1;
        break;
      }
      case 0x03: {
        assert(ts.pc + 1 < (int)prog.size() && "HCONST: missing data word");
        uint16_t hbits = prog[ts.pc + 1];
        ts.regs[rd] = static_cast<int32_t>(hbits);
        ts.pc += 1;
        break;
      }
      case 0x04: {
        assert(ts.pc + 1 < (int)prog.size() && "HADD: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t hrs = (operands >> 4) & 0xF;
        uint8_t hrt = operands & 0xF;
        float a = regToHalf(ts.regs[hrs]);
        float b = regToHalf(ts.regs[hrt]);
        ts.regs[rd] = halfToReg(a + b);
        ts.pc += 1;
        break;
      }
      case 0x05: {
        assert(ts.pc + 1 < (int)prog.size() && "HSUB: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t hrs = (operands >> 4) & 0xF;
        uint8_t hrt = operands & 0xF;
        float a = regToHalf(ts.regs[hrs]);
        float b = regToHalf(ts.regs[hrt]);
        ts.regs[rd] = halfToReg(a - b);
        ts.pc += 1;
        break;
      }
      case 0x06: {
        assert(ts.pc + 1 < (int)prog.size() && "HMUL: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t hrs = (operands >> 4) & 0xF;
        uint8_t hrt = operands & 0xF;
        float a = regToHalf(ts.regs[hrs]);
        float b = regToHalf(ts.regs[hrt]);
        ts.regs[rd] = halfToReg(a * b);
        ts.pc += 1;
        break;
      }
      case 0x07: {
        assert(ts.pc + 1 < (int)prog.size() && "HDIV: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t hrs = (operands >> 4) & 0xF;
        uint8_t hrt = operands & 0xF;
        float a = regToHalf(ts.regs[hrs]);
        float b = regToHalf(ts.regs[hrt]);
        ts.regs[rd] = halfToReg(a / b);
        ts.pc += 1;
        break;
      }
      case 0x08: {
        assert(ts.pc + 1 < (int)prog.size() &&
               "HCMP_LT: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t hrs = (operands >> 4) & 0xF;
        uint8_t hrt = operands & 0xF;
        float a = regToHalf(ts.regs[hrs]);
        float b = regToHalf(ts.regs[hrt]);
        ts.regs[rd] = (a < b) ? 1 : 0;
        ts.pc += 1;
        break;
      }
      case 0x09: {
        assert(ts.pc + 1 < (int)prog.size() && "EXP: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t frs = (operands >> 4) & 0xF;
        uint8_t typeFlag = imm & 0xF;
        if (typeFlag == 1) {
          ts.regs[rd] = halfToReg(std::exp(regToHalf(ts.regs[frs])));
        } else {
          ts.regs[rd] = floatToReg(std::exp(regToFloat(ts.regs[frs])));
        }
        ts.pc += 1;
        break;
      }
      case 0x0A: {
        assert(ts.pc + 1 < (int)prog.size() && "LOG: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t frs = (operands >> 4) & 0xF;
        uint8_t typeFlag = imm & 0xF;
        if (typeFlag == 1) {
          ts.regs[rd] = halfToReg(std::log(regToHalf(ts.regs[frs])));
        } else {
          ts.regs[rd] = floatToReg(std::log(regToFloat(ts.regs[frs])));
        }
        ts.pc += 1;
        break;
      }
      case 0x0B: {
        assert(ts.pc + 1 < (int)prog.size() && "SQRT: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t frs = (operands >> 4) & 0xF;
        uint8_t typeFlag = imm & 0xF;
        if (typeFlag == 1) {
          ts.regs[rd] = halfToReg(std::sqrt(regToHalf(ts.regs[frs])));
        } else {
          ts.regs[rd] = floatToReg(std::sqrt(regToFloat(ts.regs[frs])));
        }
        ts.pc += 1;
        break;
      }
      case 0x0C: {
        assert(ts.pc + 1 < (int)prog.size() && "RSQRT: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t frs = (operands >> 4) & 0xF;
        uint8_t typeFlag = imm & 0xF;
        if (typeFlag == 1) {
          float val = regToHalf(ts.regs[frs]);
          ts.regs[rd] = halfToReg(1.0f / std::sqrt(val));
        } else {
          float val = regToFloat(ts.regs[frs]);
          ts.regs[rd] = floatToReg(1.0f / std::sqrt(val));
        }
        ts.pc += 1;
        break;
      }
      case 0x0D: {
        assert(ts.pc + 1 < (int)prog.size() && "ABS: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t frs = (operands >> 4) & 0xF;
        uint8_t typeFlag = imm & 0xF;
        if (typeFlag == 1) {
          ts.regs[rd] = halfToReg(std::fabs(regToHalf(ts.regs[frs])));
        } else if (typeFlag == 2) {
          ts.regs[rd] = std::abs(ts.regs[frs]);
        } else {
          ts.regs[rd] = floatToReg(std::fabs(regToFloat(ts.regs[frs])));
        }
        ts.pc += 1;
        break;
      }
      case 0x0E: {
        assert(ts.pc + 1 < (int)prog.size() && "MAX: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t frs = (operands >> 4) & 0xF;
        uint8_t frt = operands & 0xF;
        uint8_t typeFlag = imm & 0xF;
        if (typeFlag == 1) {
          float a = regToHalf(ts.regs[frs]);
          float b = regToHalf(ts.regs[frt]);
          ts.regs[rd] = halfToReg(std::fmax(a, b));
        } else if (typeFlag == 2) {
          ts.regs[rd] = std::max(ts.regs[frs], ts.regs[frt]);
        } else {
          float a = regToFloat(ts.regs[frs]);
          float b = regToFloat(ts.regs[frt]);
          ts.regs[rd] = floatToReg(std::fmax(a, b));
        }
        ts.pc += 1;
        break;
      }
      case 0x0F: {
        // REDUCE: pause thread, let outer loop handle collective operation.
        assert(ts.pc + 1 < (int)prog.size() &&
               "REDUCE: missing operand word");
        uint16_t operands = prog[ts.pc + 1];
        uint8_t frs = (operands >> 4) & 0xF;
        reduceOperand = ts.regs[frs];
        reduceRd = rd;
        reduceFlags = imm & 0xF;
        return StepResult::Reduce;
      }
      default:
        break;
      }
      break;
    }
    case 0xF:
      ts.done = true;
      return StepResult::Done;
    default:
      break;
    }

    ++ts.pc;
    return StepResult::Continue;
  };

  for (int blockId = 0; blockId < numBlocks; ++blockId) {
    std::vector<ThreadState> threads(threadsPerBlock);

    constexpr int kMaxPhases = 10000;
    for (int phase = 0; phase < kMaxPhases; ++phase) {
      // Run all threads until they hit a REDUCE barrier or RET.
      std::vector<int32_t> reduceValues(threadsPerBlock, 0);
      uint8_t reduceRd = 0;
      uint8_t reduceFlags = 0;
      bool anyReduced = false;

      for (int tid = 0; tid < threadsPerBlock; ++tid) {
        auto &ts = threads[tid];
        if (ts.done)
          continue;

        constexpr int kMaxCycles = 100000;
        for (int cycle = 0; cycle < kMaxCycles; ++cycle) {
          int32_t operand = 0;
          uint8_t rRd = 0, rFlags = 0;
          auto result = execOne(ts, blockId, tid, operand, rRd, rFlags);
          if (result == StepResult::Reduce) {
            reduceValues[tid] = operand;
            reduceRd = rRd;
            reduceFlags = rFlags;
            anyReduced = true;
            break;
          }
          if (result == StepResult::Done)
            break;
        }
      }

      if (!anyReduced) {
        // All threads finished — block is done.
        break;
      }

      // Compute the reduction across all (non-done) thread contributions.
      // reduceFlags encoding: bit2 = isMax, bits[1:0] = type (0=f32,1=f16,2=i32)
      bool isMax = (reduceFlags & 4) != 0;
      uint8_t typeFlag = reduceFlags & 3;

      int32_t reduced;
      if (typeFlag == 2) {
        // Integer reduction
        int32_t acc = isMax ? threads[0].regs[0] : 0;
        bool first = true;
        for (int tid = 0; tid < threadsPerBlock; ++tid) {
          if (threads[tid].done)
            continue;
          int32_t v = reduceValues[tid];
          if (first) {
            acc = v;
            first = false;
          } else {
            acc = isMax ? std::max(acc, v) : (acc + v);
          }
        }
        reduced = acc;
      } else {
        // Float reduction (f32 or f16 — values are in register encoding)
        auto toFloat = (typeFlag == 1) ? regToHalf : regToFloat;
        auto fromFloat = (typeFlag == 1) ? halfToReg : floatToReg;
        float acc = 0.0f;
        bool first = true;
        for (int tid = 0; tid < threadsPerBlock; ++tid) {
          if (threads[tid].done)
            continue;
          float v = toFloat(reduceValues[tid]);
          if (first) {
            acc = isMax ? v : v;
            first = false;
          } else {
            acc = isMax ? std::fmax(acc, v) : (acc + v);
          }
        }
        reduced = fromFloat(acc);
      }

      // Distribute result to all non-done threads, advance past the 2-word
      // REDUCE instruction.
      for (int tid = 0; tid < threadsPerBlock; ++tid) {
        auto &ts = threads[tid];
        if (ts.done)
          continue;
        ts.regs[reduceRd] = reduced;
        ts.pc += 2; // skip opcode word + operands word
      }
    }
  }
}

} // namespace tinyton
