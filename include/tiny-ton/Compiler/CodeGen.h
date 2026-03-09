#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tinyton {

struct Function;

struct Instruction {
  uint16_t encoding;
  std::string assembly;
};

std::vector<Instruction> emit(const Function &func);

} // namespace tinyton
