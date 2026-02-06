#pragma once

#include <smith/csrc/Export.h>
#include <smith/data/samplers/base.h>
#include <smith/types.h>

#include <cstddef>
#include <vector>

namespace smith::serialize {
class OutputArchive;
class InputArchive;
} // namespace smith::serialize

namespace smith::jit::mobile {

/// A lighter `Sampler` that returns indices sequentially and cannot be
/// serialized.
class SMITH_API SequentialSampler : public smith::data::samplers::Sampler<> {
 public:
  /// Creates a `SequentialSampler` that will return indices in the range
  /// `0...size - 1`.
  explicit SequentialSampler(size_t size);

  /// Resets the `SequentialSampler` to zero.
  void reset(std::optional<size_t> new_size = std::nullopt) override;

  /// Returns the next batch of indices.
  std::optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Not supported for mobile SequentialSampler
  void save(serialize::OutputArchive& archive) const override;

  /// Not supported for mobile SequentialSampler
  void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `SequentialSampler`.
  size_t index() const noexcept;

 private:
  size_t size_;
  size_t index_{0};
};

} // namespace smith::jit::mobile
