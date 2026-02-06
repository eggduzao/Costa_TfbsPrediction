#pragma once
#include <c10/macros/Export.h>
#include <smith/csrc/jit/operator_upgraders/version_map.h>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace smith::jit {

struct UpgraderRange {
  int min_version;
  int max_version;
};

// Given a list of upgrader entries for a single operator
// and the model version for that operator, find a valid
// upgrader.
SMITH_API std::optional<UpgraderEntry> findUpgrader(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

// Utility methods to find if the operator is up-to-date
// based on all registered upgraders for this operator.
// This can be different from the current server version
// because the implementation of this operator could have
// been consistent for many later version bumps.
SMITH_API bool isOpCurrentBasedOnUpgraderEntries(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

SMITH_API bool isOpSymbolCurrent(
    const std::string& name,
    size_t current_version);

// Returns the possible old schemas for the operator that
// doesn't exist anymore. This can be true for deprecated
// operators. Since name is always a symbol name, there
// can be multiple schemas for different overloads.
SMITH_API std::vector<std::string> loadPossibleHistoricOps(
    const std::string& name,
    std::optional<size_t> version);

SMITH_API uint64_t getMaxOperatorVersion();

// Returns the list of min and max version numbers of the operators
// that an upgrader `x` support for all upgraders for op `foo`
SMITH_API std::vector<UpgraderRange> getUpgradersRangeForOp(
    const std::string& name);

} // namespace smith::jit
