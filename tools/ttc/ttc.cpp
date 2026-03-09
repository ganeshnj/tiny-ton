#include "tiny-ton/Compiler/Pipeline.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " [--emit mlir|asm|hex|bin|trace] <input.ttn>\n";
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string emitMode = "asm";
  std::string inputFile;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--emit" && i + 1 < argc) {
      emitMode = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else {
      inputFile = arg;
    }
  }

  if (inputFile.empty()) {
    std::cerr << "error: no input file\n";
    return 1;
  }

  std::ifstream ifs(inputFile);
  if (!ifs) {
    std::cerr << "error: cannot open '" << inputFile << "'\n";
    return 1;
  }

  std::ostringstream ss;
  ss << ifs.rdbuf();
  std::string source = ss.str();

  tinyton::CompileOptions opts;
  if (emitMode == "mlir")
    opts.emitMode = tinyton::CompileOptions::EmitMode::MLIR;
  else if (emitMode == "asm")
    opts.emitMode = tinyton::CompileOptions::EmitMode::Asm;
  else if (emitMode == "hex")
    opts.emitMode = tinyton::CompileOptions::EmitMode::Hex;
  else if (emitMode == "bin")
    opts.emitMode = tinyton::CompileOptions::EmitMode::Bin;
  else if (emitMode == "trace")
    opts.emitMode = tinyton::CompileOptions::EmitMode::Trace;
  else {
    std::cerr << "error: unknown emit mode '" << emitMode << "'\n";
    return 1;
  }

  auto result = tinyton::compile(source, opts);
  if (!result.success) {
    std::cerr << "error: " << result.error << "\n";
    return 1;
  }

  std::cout << result.output;
  return 0;
}
