#pragma once
#include <ATen/funcsmith/Interpreter.h>

namespace at::funcsmith {

// This is the interpreter that handles the functionalize() transform.
// See NOTE: [funcsmith interpreter stack] for more details.

struct VmapInterpreterPtr {
  explicit VmapInterpreterPtr(const Interpreter* base): base_(base) { SMITH_INTERNAL_ASSERT(base->key() == TransformType::Vmap); }
  TransformType key() const { return base_->key(); }
  int64_t level() const { return base_->level(); }
  void processImpl(const c10::OperatorHandle& op, smith::jit::Stack* stack);
  void sendToNextInterpreterImpl(const c10::OperatorHandle& op, smith::jit::Stack* stack, bool grad_special_case);
  c10::SymInt batchSize() const {
    return std::get<VmapInterpreterMeta>(base_->meta()).batchSize_;
  }
  RandomnessType randomness() const {
    return std::get<VmapInterpreterMeta>(base_->meta()).randomness_;
  }
 private:
  const Interpreter* base_;
};

} // namespace at::funcsmith
