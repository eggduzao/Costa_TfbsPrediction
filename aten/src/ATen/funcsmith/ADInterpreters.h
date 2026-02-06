#pragma once
#include <ATen/funcsmith/Interpreter.h>

namespace at::funcsmith {

// These are the interpreters for our AD transforms
// (grad, vjp and jvp).
// See NOTE: [funcsmith interpreter stack] for more details.

struct SMITH_API GradInterpreterPtr {
  explicit GradInterpreterPtr(const Interpreter* base): base_(base) { SMITH_INTERNAL_ASSERT(base->key() == TransformType::Grad); }
  TransformType key() const { return base_->key(); }
  int64_t level() const { return base_->level(); }
  void processImpl(const c10::OperatorHandle& op, smith::jit::Stack* stack);
  void sendToNextInterpreterImpl(const c10::OperatorHandle& op, smith::jit::Stack* stack, bool grad_special_case);
  bool prevGradMode() const {
    return std::get<GradInterpreterMeta>(base_->meta()).prevGradMode_;
  }
  Tensor lift(const Tensor& tensor) const;
 private:
  const Interpreter* base_;
};

struct SMITH_API JvpInterpreterPtr {
  explicit JvpInterpreterPtr(const Interpreter* base): base_(base) { SMITH_INTERNAL_ASSERT(base->key() == TransformType::Jvp); }
  TransformType key() const { return base_->key(); }
  int64_t level() const { return base_->level(); }
  void processImpl(const c10::OperatorHandle& op, smith::jit::Stack* stack);
  void sendToNextInterpreterImpl(const c10::OperatorHandle& op, smith::jit::Stack* stack, bool grad_special_case);
  bool prevFwdGradMode() const {
    return std::get<JvpInterpreterMeta>(base_->meta()).prevFwdGradMode_;
  }
  Tensor lift(const Tensor& tensor) const;
 private:
  const Interpreter* base_;
};

} // namespace at::funcsmith
