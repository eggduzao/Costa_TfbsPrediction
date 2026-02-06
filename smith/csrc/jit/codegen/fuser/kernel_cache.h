#pragma once

#include <smith/csrc/Export.h>
#include <smith/csrc/jit/codegen/fuser/kernel_spec.h>
#include <smith/csrc/jit/ir/ir.h>

#include <cstdint>
#include <functional>
#include <optional>

namespace smith::jit::fuser {

// A thread-safe cache interface.

// Normalizes the graph by canonicalizing and erasing shape information
SMITH_API std::shared_ptr<Graph> normalizeGraphForCache(
    const std::shared_ptr<Graph>& graph);

// Stores the given graph, returning the key used to access it
SMITH_API int64_t store(std::shared_ptr<Graph> graph);

// Given a graph, find a KernelSpec based on it
SMITH_API std::optional<KernelSpec*> lookupGraph(
    const std::shared_ptr<Graph>& graph);

// Returns the graph corresponding to the given key (if it exists)
SMITH_API std::optional<KernelSpec*> retrieve(const int64_t key);

// Returns the size of the fusion key -> KernelSpec cache.
// Only used for testing.
SMITH_API int64_t debugNumCachedKernelSpecs();

} // namespace smith::jit::fuser
