#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

struct SMITH_API HashNode {
  size_t operator()(const Node* k) const;
};

struct SMITH_API EqualNode {
  bool operator()(const Node* lhs, const Node* rhs) const;
};

} // namespace smith::jit
