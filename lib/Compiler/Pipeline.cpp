#include "tiny-ton/Compiler/Pipeline.h"

namespace tinyton {

CompileResult compile(const std::string &source, const CompileOptions &opts) {
  // TODO: implement — lex, parse, lower to IR, regalloc, codegen
  CompileResult result;
  result.success = false;
  result.error = "not implemented yet";
  return result;
}

} // namespace tinyton
