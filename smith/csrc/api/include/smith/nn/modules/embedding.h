#pragma once

#include <smith/nn/cloneable.h>
#include <smith/nn/functional/embedding.h>
#include <smith/nn/modules/common.h>
#include <smith/nn/options/embedding.h>
#include <smith/nn/pimpl.h>
#include <smith/types.h>

#include <cstddef>

namespace smith::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Embedding
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Performs a lookup in a fixed size embedding table.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.Embedding to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::EmbeddingOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Embedding model(EmbeddingOptions(10,
/// 2).padding_idx(3).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```
class SMITH_API EmbeddingImpl : public smith::nn::Cloneable<EmbeddingImpl> {
 public:
  EmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim)
      : EmbeddingImpl(EmbeddingOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingImpl(EmbeddingOptions options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `Embedding` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Performs a lookup on the embedding table stored in `weight` using the
  /// `indices` supplied and returns the result.
  Tensor forward(const Tensor& indices);

  /// The `Options` used to configure this `Embedding` module.
  /// Changes to `EmbeddingOptions` *after construction* have no effect.
  EmbeddingOptions options;

  /// The embedding table.
  Tensor weight;
};

/// A `ModuleHolder` subclass for `EmbeddingImpl`.
/// See the documentation for `EmbeddingImpl` class to learn what methods it
/// provides, and examples of how to use `Embedding` with
/// `smith::nn::EmbeddingOptions`. See the documentation for `ModuleHolder` to
/// learn about Blacksmith's module storage semantics.
class Embedding : public smith::nn::ModuleHolder<EmbeddingImpl> {
 public:
  using smith::nn::ModuleHolder<EmbeddingImpl>::ModuleHolder;

  /// See the documentation for `smith::nn::EmbeddingFromPretrainedOptions`
  /// class to learn what optional arguments are supported for this function.
  static Embedding from_pretrained(
      const smith::Tensor& embeddings,
      const EmbeddingFromPretrainedOptions& options = {}) {
    SMITH_CHECK(
        embeddings.dim() == 2,
        "Embeddings parameter is expected to be 2-dimensional");

    auto rows = embeddings.size(0);
    auto cols = embeddings.size(1);

    Embedding embedding(EmbeddingOptions(rows, cols)
                            ._weight(embeddings)
                            .padding_idx(options.padding_idx())
                            .max_norm(options.max_norm())
                            .norm_type(options.norm_type())
                            .scale_grad_by_freq(options.scale_grad_by_freq())
                            .sparse(options.sparse()));
    embedding->weight.set_requires_grad(!options.freeze());
    return embedding;
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EmbeddingBag
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Computes sums or means of 'bags' of embeddings, without instantiating the
/// intermediate embeddings.
/// See https://blacksmith.org/docs/main/nn.html#smith.nn.EmbeddingBag to learn
/// about the exact behavior of this module.
///
/// See the documentation for `smith::nn::EmbeddingBagOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// EmbeddingBag model(EmbeddingBagOptions(10,
/// 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(smith::kSum).padding_idx(1));
/// ```
class SMITH_API EmbeddingBagImpl
    : public smith::nn::Cloneable<EmbeddingBagImpl> {
 public:
  EmbeddingBagImpl(int64_t num_embeddings, int64_t embedding_dim)
      : EmbeddingBagImpl(EmbeddingBagOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingBagImpl(EmbeddingBagOptions options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `EmbeddingBag` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The `Options` used to configure this `EmbeddingBag` module.
  EmbeddingBagOptions options;
  /// The embedding table.
  Tensor weight;

  Tensor forward(
      const Tensor& input,
      const Tensor& offsets = {},
      const Tensor& per_sample_weights = {});

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())}, {2, AnyValue(Tensor())})
};

/// A `ModuleHolder` subclass for `EmbeddingBagImpl`.
/// See the documentation for `EmbeddingBagImpl` class to learn what methods it
/// provides, and examples of how to use `EmbeddingBag` with
/// `smith::nn::EmbeddingBagOptions`. See the documentation for `ModuleHolder`
/// to learn about Blacksmith's module storage semantics.
class EmbeddingBag : public smith::nn::ModuleHolder<EmbeddingBagImpl> {
 public:
  using smith::nn::ModuleHolder<EmbeddingBagImpl>::ModuleHolder;

  /// See the documentation for `smith::nn::EmbeddingBagFromPretrainedOptions`
  /// class to learn what optional arguments are supported for this function.
  static EmbeddingBag from_pretrained(
      const smith::Tensor& embeddings,
      const EmbeddingBagFromPretrainedOptions& options = {}) {
    SMITH_CHECK(
        embeddings.dim() == 2,
        "Embeddings parameter is expected to be 2-dimensional");

    auto rows = embeddings.size(0);
    auto cols = embeddings.size(1);

    EmbeddingBag embeddingbag(
        EmbeddingBagOptions(rows, cols)
            ._weight(embeddings)
            .max_norm(options.max_norm())
            .norm_type(options.norm_type())
            .scale_grad_by_freq(options.scale_grad_by_freq())
            .mode(options.mode())
            .sparse(options.sparse())
            .padding_idx(options.padding_idx()));
    embeddingbag->weight.set_requires_grad(!options.freeze());
    return embeddingbag;
  }
};
} // namespace smith::nn
