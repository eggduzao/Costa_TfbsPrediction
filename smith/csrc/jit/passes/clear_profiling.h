#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

SMITH_API void unprofileGraphInputs(const std::shared_ptr<Graph>& graph);
SMITH_API void unprofileBlock(Block* start_block);
// Unprofiles all the node outputs in a block.

SMITH_API void ClearProfilingInformation(const std::shared_ptr<Graph>& graph);

} // namespace smith::jit
