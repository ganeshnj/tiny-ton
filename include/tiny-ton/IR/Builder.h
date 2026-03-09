#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tinyton {

struct Value;
struct Function;

class IRBuilder {
public:
  IRBuilder();
  ~IRBuilder();

  void beginFunction(const std::string &name, int numArgs);
  Value *getArg(int index);

  Value *emitConst(int64_t val);
  Value *emitAdd(Value *lhs, Value *rhs);
  Value *emitSub(Value *lhs, Value *rhs);
  Value *emitMul(Value *lhs, Value *rhs);
  Value *emitDiv(Value *lhs, Value *rhs);
  Value *emitLoad(Value *addr);
  void emitStore(Value *addr, Value *val);
  Value *emitProgramId(int axis);
  void emitRet();

  std::unique_ptr<Function> build();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tinyton
