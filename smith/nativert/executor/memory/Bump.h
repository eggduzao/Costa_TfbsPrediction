#pragma once

#include <smith/nativert/executor/memory/LayoutPlannerAlgorithm.h>

namespace smith::nativert {

// lay out all tensors contiguously in memory
// this doesn't take into account lifetimes,
// it literally just puts them all next to each other
LayoutPlan BumpAllocationPlanner(
    const std::vector<AllocationSpec>& allocation_specs);

} // namespace smith::nativert
