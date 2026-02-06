#pragma once

#include <smith/csrc/jit/ir/ir.h>

// Functions used by both encapsulation and conversion.

namespace smith::jit {

struct IndexingPatternFinder {
 public:
  static std::vector<Node*> FetchSliceAndSelect(const Node* node);

 private:
  static bool IsSameSource(const Node* n, const Node* m);
};

} // namespace smith::jit
