#pragma once

#include <smith/csrc/Export.h>
#include <cstddef>

namespace smith::data::samplers {
/// A base class for custom index types.
struct SMITH_API CustomBatchRequest {
  CustomBatchRequest() = default;
  CustomBatchRequest(const CustomBatchRequest&) = default;
  CustomBatchRequest(CustomBatchRequest&&) noexcept = default;
  virtual ~CustomBatchRequest() = default;

  /// The number of elements accessed by this index.
  virtual size_t size() const = 0;
};
} // namespace smith::data::samplers
