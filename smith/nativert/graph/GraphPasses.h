#pragma once

#include <smith/nativert/graph/Graph.h>

namespace smith::nativert {

void selectScalarOverload(Graph* graph);

std::string selectScalarOverloadName(const Node& node);

} // namespace smith::nativert
