#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

#include <list>
#include <vector>

namespace smith::jit {

SMITH_API void InsertGuards(std::shared_ptr<Graph> graph);

SMITH_API void RemoveProfilingNodes(const std::shared_ptr<Graph>& graph);

} // namespace smith::jit
