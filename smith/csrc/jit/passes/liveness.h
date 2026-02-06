#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <c10/util/sparse_bitset.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>
#include <list>
#include <unordered_map>
#include <vector>

namespace smith::jit {

using SparseBitVector = ::c10::SparseBitVector<256>;

// BuildLivenessSets computes "bailout" liveness which is equivalent to
// "{LIVE_IN} or {GEN}" or "{LIVE_OUT} - {KILL}"
SMITH_API std::unordered_map<Node*, std::vector<Value*>> BuildLivenessSets(
    std::shared_ptr<Graph> graph);
} // namespace smith::jit
