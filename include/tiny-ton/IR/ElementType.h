#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
#include <string>

namespace tinyton {

enum class ElementType : uint8_t {
  I32 = 0,
  F32 = 1,
  F16 = 2,
};

inline mlir::Type elementTypeToMLIR(ElementType et, mlir::MLIRContext *ctx) {
  switch (et) {
  case ElementType::F32:
    return mlir::Float32Type::get(ctx);
  case ElementType::F16:
    return mlir::Float16Type::get(ctx);
  case ElementType::I32:
    return mlir::IntegerType::get(ctx, 32);
  }
}

inline ElementType mlirTypeToElementType(mlir::Type ty) {
  if (ty.isF32())
    return ElementType::F32;
  if (ty.isF16())
    return ElementType::F16;
  return ElementType::I32;
}

inline bool isFloatElementType(ElementType et) {
  return et == ElementType::F32 || et == ElementType::F16;
}

inline ElementType elementTypeFromString(const std::string &s) {
  if (s == "f32")
    return ElementType::F32;
  if (s == "f16")
    return ElementType::F16;
  return ElementType::I32;
}

} // namespace tinyton
