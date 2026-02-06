#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API std::shared_ptr<Graph> Canonicalize(
    const std::shared_ptr<Graph>& graph,
    bool keep_unique_names = true);

SMITH_API void CanonicalizeOutputs(std::shared_ptr<Graph>& graph);

SMITH_API std::optional<const Use> firstOrLastUse(Value* v, bool find_first);

SMITH_API bool isBeforeOrAfter(
    const Use& a,
    const Use& b,
    bool checking_before);

} // namespace smith::jit
