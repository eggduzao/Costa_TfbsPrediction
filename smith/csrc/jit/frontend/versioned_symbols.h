#pragma once

#include <caffe2/serialize/versions.h>
#include <smith/csrc/Export.h>
#include <smith/csrc/jit/api/module.h>

#include <cstdint>

namespace smith::jit {
// Maps the given symbol into an implementation of its behavior at the
// given version.
// See note [Versioned Symbols]
SMITH_API Symbol
get_symbol_for_version(const Symbol name, const uint64_t version);

// Maps the given kind to the minimum version that supports it.
// See note [Dynamic Versions and smith.jit.save vs. smith.save]
SMITH_API uint64_t get_min_version_for_kind(const NodeKind& kind);
} // namespace smith::jit
