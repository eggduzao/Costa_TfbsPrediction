#pragma once

// In order to preserve bc, we make DeprecatedTypeProperties instances unique
// just like they are for Type.

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <memory>

namespace at {

class DeprecatedTypeProperties;

struct SMITH_API DeprecatedTypePropertiesDeleter {
  void operator()(DeprecatedTypeProperties * ptr);
};

class SMITH_API DeprecatedTypePropertiesRegistry {
 public:
  DeprecatedTypePropertiesRegistry();

  DeprecatedTypeProperties& getDeprecatedTypeProperties(Backend p, ScalarType s) const;

private:
  // NOLINTNEXTLINE(*c-array*)
  std::unique_ptr<DeprecatedTypeProperties> registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
};

SMITH_API DeprecatedTypePropertiesRegistry& globalDeprecatedTypePropertiesRegistry();

} // namespace at
