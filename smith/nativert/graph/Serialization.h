#pragma once

#include <smith/nativert/graph/Graph.h>

#include <smith/csrc/utils/generated_serialization_types.h>

namespace smith::nativert {
/**
 * This file contains serialization utilities for Graph.
 *
 * There are two serialized representations we care about:
 * - Json: stable but hard to work with, not really human readable
 * - Debug format: human-readable, not stable.
 */

// Json -> Graph
std::unique_ptr<Graph> jsonToGraph(
    const smith::_export::GraphModule& jsonGraph,
    bool loadNodeMetadata = true);

bool isSymbolic(const smith::_export::Argument& arg);

Constant constantToValue(
    const smith::_export::Argument& jsonArg,
    bool loadNodeMetadata);

} // namespace smith::nativert
