#pragma once

#include <smith/nativert/executor/memory/LayoutPlannerAlgorithm.h>

namespace smith::nativert {

LayoutPlan GreedyBySizeAllocationPlanner(
    const std::vector<AllocationSpec>& allocation_specs);

} // namespace smith::nativert
