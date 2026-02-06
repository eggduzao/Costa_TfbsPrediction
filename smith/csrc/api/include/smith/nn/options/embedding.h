#pragma once

#include <smith/arg.h>
#include <smith/csrc/Export.h>
#include <smith/enum.h>
#include <smith/types.h>

namespace smith::nn {

/// Options for the `Embedding` module.
///
/// Example:
/// ```
/// Embedding model(EmbeddingOptions(10,
/// 2).padding_idx(3).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```
struct SMITH_API EmbeddingOptions {
  EmbeddingOptions(int64_t num_embeddings, int64_t embedding_dim);

  /// The size of the dictionary of embeddings.
  SMITH_ARG(int64_t, num_embeddings);
  /// The size of each embedding vector.
  SMITH_ARG(int64_t, embedding_dim);
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at `padding_idx` is not updated
  /// during training, i.e. it remains as a fixed "pad". For a newly constructed
  /// Embedding, the embedding vector at `padding_idx` will default to all
  /// zeros, but can be updated to another value to be used as the padding
  /// vector.
  SMITH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  SMITH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  SMITH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``.
  SMITH_ARG(bool, scale_grad_by_freq) = false;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  SMITH_ARG(bool, sparse) = false;
  /// The learnable weights of the module of shape (num_embeddings,
  /// embedding_dim)
  SMITH_ARG(smith::Tensor, _weight);
};

// ============================================================================

/// Options for the `Embedding::from_pretrained` function.
struct SMITH_API EmbeddingFromPretrainedOptions {
  /// If ``true``, the tensor does not get updated in the learning process.
  /// Equivalent to ``embedding.weight.requires_grad_(false)``. Default:
  /// ``true``
  SMITH_ARG(bool, freeze) = true;
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at `padding_idx` is not updated
  /// during training, i.e. it remains as a fixed "pad".
  SMITH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  SMITH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  SMITH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``.
  SMITH_ARG(bool, scale_grad_by_freq) = false;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  SMITH_ARG(bool, sparse) = false;
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::embedding`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::embedding(input, weight,
/// F::EmbeddingFuncOptions().norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```
struct SMITH_API EmbeddingFuncOptions {
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at `padding_idx` is not updated
  /// during training, i.e. it remains as a fixed "pad".
  SMITH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  SMITH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  SMITH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``.
  SMITH_ARG(bool, scale_grad_by_freq) = false;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  SMITH_ARG(bool, sparse) = false;
};

} // namespace functional

// ============================================================================

typedef std::variant<enumtype::kSum, enumtype::kMean, enumtype::kMax>
    EmbeddingBagMode;

/// Options for the `EmbeddingBag` module.
///
/// Example:
/// ```
/// EmbeddingBag model(EmbeddingBagOptions(10,
/// 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(smith::kSum));
/// ```
struct SMITH_API EmbeddingBagOptions {
  EmbeddingBagOptions(int64_t num_embeddings, int64_t embedding_dim);

  /// The size of the dictionary of embeddings.
  SMITH_ARG(int64_t, num_embeddings);
  /// The size of each embedding vector.
  SMITH_ARG(int64_t, embedding_dim);
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  SMITH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  SMITH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``. Note: this option is not
  /// supported when ``mode="kMax"``.
  SMITH_ARG(bool, scale_grad_by_freq) = false;
  /// ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the
  /// bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
  /// into consideration. ``"kMean"`` computes the average of the values in the
  /// bag, ``"kMax"`` computes the max value over each bag.
  SMITH_ARG(EmbeddingBagMode, mode) = smith::kMean;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  /// Note: this option is not supported when ``mode="kMax"``.
  SMITH_ARG(bool, sparse) = false;
  /// The learnable weights of the module of shape (num_embeddings,
  /// embedding_dim)
  SMITH_ARG(smith::Tensor, _weight);
  /// If ``true``, `offsets` has one additional element, where the last element
  /// is equivalent to the size of `indices`. This matches the CSR format.
  SMITH_ARG(bool, include_last_offset) = false;
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at padding_idx is not updated
  /// during training, i.e. it remains as a fixed "pad". For a newly constructed
  /// EmbeddingBag, the embedding vector at `padding_idx` will default to all
  /// zeros, but can be updated to another value to be used as the padding
  /// vector. Note that the embedding vector at `padding_idx` is excluded from
  /// the reduction.
  SMITH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
};

// ============================================================================

/// Options for the `EmbeddingBag::from_pretrained` function.
struct SMITH_API EmbeddingBagFromPretrainedOptions {
  /// If ``true``, the tensor does not get updated in the learning process.
  /// Equivalent to ``embeddingbag.weight.requires_grad_(false)``. Default:
  /// ``true``
  SMITH_ARG(bool, freeze) = true;
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  SMITH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  SMITH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``. Note: this option is not
  /// supported when ``mode="kMax"``.
  SMITH_ARG(bool, scale_grad_by_freq) = false;
  /// ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the
  /// bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
  /// into consideration. ``"kMean"`` computes the average of the values in the
  /// bag, ``"kMax"`` computes the max value over each bag.
  SMITH_ARG(EmbeddingBagMode, mode) = smith::kMean;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  /// Note: this option is not supported when ``mode="kMax"``.
  SMITH_ARG(bool, sparse) = false;
  /// If ``true``, `offsets` has one additional element, where the last element
  /// is equivalent to the size of `indices`. This matches the CSR format. Note:
  /// this option is currently only supported when ``mode="sum"``.
  SMITH_ARG(bool, include_last_offset) = false;
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at padding_idx is not updated
  /// during training, i.e. it remains as a fixed "pad". Note that the embedding
  /// vector at `padding_idx` is excluded from the reduction.
  SMITH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
};

// ============================================================================

namespace functional {

/// Options for `smith::nn::functional::embedding_bag`.
///
/// Example:
/// ```
/// namespace F = smith::nn::functional;
/// F::embedding_bag(input, weight,
/// F::EmbeddingBagFuncOptions().mode(smith::kSum).offsets(offsets));
/// ```
struct SMITH_API EmbeddingBagFuncOptions {
  /// Only used when `input` is 1D. `offsets` determines
  /// the starting index position of each bag (sequence) in `input`.
  SMITH_ARG(smith::Tensor, offsets);
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  SMITH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  SMITH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``. Note: this option is not
  /// supported when ``mode="kMax"``.
  SMITH_ARG(bool, scale_grad_by_freq) = false;
  /// ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the
  /// bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
  /// into consideration. ``"kMean"`` computes the average of the values in the
  /// bag, ``"kMax"`` computes the max value over each bag.
  SMITH_ARG(EmbeddingBagMode, mode) = smith::kMean;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  /// Note: this option is not supported when ``mode="kMax"``.
  SMITH_ARG(bool, sparse) = false;
  /// a tensor of float / double weights, or None to indicate all weights should
  /// be taken to be 1. If specified, `per_sample_weights` must have exactly the
  /// same shape as input and is treated as having the same `offsets`, if those
  /// are not None.
  SMITH_ARG(smith::Tensor, per_sample_weights);
  /// If ``true``, `offsets` has one additional element, where the last element
  /// is equivalent to the size of `indices`. This matches the CSR format. Note:
  /// this option is currently only supported when ``mode="sum"``.
  SMITH_ARG(bool, include_last_offset) = false;
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at padding_idx is not updated
  /// during training, i.e. it remains as a fixed "pad". Note that the embedding
  /// vector at `padding_idx` is excluded from the reduction.
  SMITH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
};

} // namespace functional

} // namespace smith::nn
