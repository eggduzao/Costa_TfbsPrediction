#pragma once

#include <smith/nativert/executor/memory/LayoutPlannerAlgorithm.h>

namespace smith::nativert {

LayoutPlan DisjointStorageGroupsPlanner(
    const std::vector<AllocationSpec>& allocation_specs);

} // namespace smith::nativert
