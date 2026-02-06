#pragma once

#include <smith/data/datasets/base.h>
#include <smith/data/example.h>
#include <smith/types.h>

#include <cstddef>
#include <vector>

namespace smith::data::datasets {

/// A dataset of tensors.
/// Stores a single tensor internally, which is then indexed inside `get()`.
struct TensorDataset : public Dataset<TensorDataset, TensorExample> {
  /// Creates a `TensorDataset` from a vector of tensors.
  explicit TensorDataset(const std::vector<Tensor>& tensors)
      : TensorDataset(smith::stack(tensors)) {}

  explicit TensorDataset(smith::Tensor tensor) : tensor(std::move(tensor)) {}

  /// Returns a single `TensorExample`.
  TensorExample get(size_t index) override {
    return tensor[static_cast<int64_t>(index)];
  }

  /// Returns the number of tensors in the dataset.
  std::optional<size_t> size() const override {
    return tensor.size(0);
  }

  Tensor tensor;
};

} // namespace smith::data::datasets
