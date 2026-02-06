#pragma once

#include <smith/csrc/Export.h>
#include <smith/data/samplers/base.h>
#include <smith/data/samplers/custom_batch_request.h>
#include <smith/types.h>

#include <cstddef>

namespace smith::serialize {
class InputArchive;
class OutputArchive;
} // namespace smith::serialize

namespace smith::data::samplers {

/// A wrapper around a batch size value, which implements the
/// `CustomBatchRequest` interface.
struct SMITH_API BatchSize : public CustomBatchRequest {
  explicit BatchSize(size_t size);
  size_t size() const noexcept override;
  operator size_t() const noexcept;
  size_t size_;
};

/// A sampler for (potentially infinite) streams of data.
///
/// The major feature of the `StreamSampler` is that it does not return
/// particular indices, but instead only the number of elements to fetch from
/// the dataset. The dataset has to decide how to produce those elements.
class SMITH_API StreamSampler : public Sampler<BatchSize> {
 public:
  /// Constructs the `StreamSampler` with the number of individual examples that
  /// should be fetched until the sampler is exhausted.
  explicit StreamSampler(size_t epoch_size);

  /// Resets the internal state of the sampler.
  void reset(std::optional<size_t> new_size = std::nullopt) override;

  /// Returns a `BatchSize` object with the number of elements to fetch in the
  /// next batch. This number is the minimum of the supplied `batch_size` and
  /// the difference between the `epoch_size` and the current index. If the
  /// `epoch_size` has been reached, returns an empty optional.
  std::optional<BatchSize> next(size_t batch_size) override;

  /// Serializes the `StreamSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `StreamSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

 private:
  size_t examples_retrieved_so_far_ = 0;
  size_t epoch_size_;
};

} // namespace smith::data::samplers
