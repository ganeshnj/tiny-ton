#include "tiny-ton/IR/Builder.h"

namespace tinyton {

struct Value {
  int id;
  std::string name;
};

struct Function {
  std::string name;
  std::vector<Value *> args;
};

struct IRBuilder::Impl {
  std::unique_ptr<Function> func;
  int nextId = 0;
};

IRBuilder::IRBuilder() : impl_(std::make_unique<Impl>()) {}
IRBuilder::~IRBuilder() = default;

void IRBuilder::beginFunction(const std::string &name, int numArgs) {
  // TODO: implement
}

Value *IRBuilder::getArg(int index) {
  // TODO: implement
  return nullptr;
}

Value *IRBuilder::emitConst(int64_t val) {
  // TODO: implement
  return nullptr;
}

Value *IRBuilder::emitAdd(Value *lhs, Value *rhs) {
  // TODO: implement
  return nullptr;
}

Value *IRBuilder::emitSub(Value *lhs, Value *rhs) {
  // TODO: implement
  return nullptr;
}

Value *IRBuilder::emitMul(Value *lhs, Value *rhs) {
  // TODO: implement
  return nullptr;
}

Value *IRBuilder::emitDiv(Value *lhs, Value *rhs) {
  // TODO: implement
  return nullptr;
}

Value *IRBuilder::emitLoad(Value *addr) {
  // TODO: implement
  return nullptr;
}

void IRBuilder::emitStore(Value *addr, Value *val) {
  // TODO: implement
}

Value *IRBuilder::emitProgramId(int axis) {
  // TODO: implement
  return nullptr;
}

void IRBuilder::emitRet() {
  // TODO: implement
}

std::unique_ptr<Function> IRBuilder::build() {
  // TODO: implement
  return std::move(impl_->func);
}

} // namespace tinyton
