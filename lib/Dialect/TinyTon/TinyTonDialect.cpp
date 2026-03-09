#include "tiny-ton/Dialect/TinyTon/TinyTonDialect.h"
#include "tiny-ton/Dialect/TinyTon/TinyTonOps.h"

#include "tiny-ton/Dialect/TinyTon/TinyTonDialect.cpp.inc"

namespace tinyton {

void TinyTonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny-ton/Dialect/TinyTon/TinyTonOps.cpp.inc"
      >();
}

} // namespace tinyton
