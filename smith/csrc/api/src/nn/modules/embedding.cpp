#include <smith/nn/modules/embedding.h>

#include <smith/nn/init.h>
#include <smith/utils.h>

#include <ostream>
#include <utility>

namespace F = smith::nn::functional;

namespace smith::nn {
EmbeddingImpl::EmbeddingImpl(EmbeddingOptions options_)
    : options(std::move(options_)) {
  EmbeddingImpl::reset();
}

void EmbeddingImpl::reset() {
  if (options.padding_idx().has_value()) {
    if (options.padding_idx() > 0) {
      SMITH_CHECK(
          options.padding_idx() < options.num_embeddings(),
          "Padding_idx must be within num_embeddings");
    } else if (options.padding_idx() < 0) {
      SMITH_CHECK(
          options.padding_idx() >= -options.num_embeddings(),
          "Padding_idx must be within num_embedding");
      options.padding_idx(
          // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
          options.num_embeddings() + *options.padding_idx());
    }
  }

  if (!options._weight().defined()) {
    weight = register_parameter(
        "weight",
        smith::empty({options.num_embeddings(), options.embedding_dim()}));
    reset_parameters();
  } else {
    SMITH_CHECK(
        options._weight().sizes() ==
            smith::IntArrayRef(
                {options.num_embeddings(), options.embedding_dim()}),
        "Shape of _weight does not match num_embeddings and embedding_dim");
    weight = register_parameter("weight", options._weight());
  }
}

void EmbeddingImpl::reset_parameters() {
  smith::nn::init::normal_(weight);
  if (options.padding_idx().has_value()) {
    smith::NoGradGuard no_grad;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    weight[*options.padding_idx()].fill_(0);
  }
}

void EmbeddingImpl::pretty_print(std::ostream& stream) const {
  stream << "smith::nn::Embedding(num_embeddings=" << options.num_embeddings()
         << ", embedding_dim=" << options.embedding_dim();
  auto const& padding_idx_opt = options.padding_idx();
  if (padding_idx_opt.has_value()) {
    stream << ", padding_idx=" << padding_idx_opt.value();
  }
  auto const& max_norm_opt = options.max_norm();
  if (max_norm_opt.has_value()) {
    stream << ", max_norm=" << max_norm_opt.value();
  }
  if (options.norm_type() != 2) {
    stream << ", norm_type=" << options.norm_type();
  }
  if (options.scale_grad_by_freq()) {
    stream << ", scale_grad_by_freq=" << std::boolalpha
           << options.scale_grad_by_freq();
  }
  if (options.sparse()) {
    stream << ", sparse=" << std::boolalpha << options.sparse();
  }
  stream << ')';
}

smith::Tensor EmbeddingImpl::forward(const Tensor& input) {
  return F::detail::embedding(
      input,
      weight,
      options.padding_idx(),
      options.max_norm(),
      options.norm_type(),
      options.scale_grad_by_freq(),
      options.sparse());
}

EmbeddingBagImpl::EmbeddingBagImpl(EmbeddingBagOptions options_)
    : options(std::move(options_)) {
  EmbeddingBagImpl::reset();
}

void EmbeddingBagImpl::reset() {
  auto const& padding_idx_opt = options.padding_idx();
  if (padding_idx_opt.has_value()) {
    auto padding_idx = padding_idx_opt.value();
    if (padding_idx > 0) {
      SMITH_CHECK(
          padding_idx < options.num_embeddings(),
          "Padding_idx must be within num_embeddings");
    } else if (padding_idx < 0) {
      SMITH_CHECK(
          padding_idx >= -options.num_embeddings(),
          "Padding_idx must be within num_embedding");
      options.padding_idx(options.num_embeddings() + padding_idx);
    }
  }
  if (!options._weight().defined()) {
    weight = register_parameter(
        "weight",
        smith::empty({options.num_embeddings(), options.embedding_dim()}));
    reset_parameters();
  } else {
    SMITH_CHECK(
        options._weight().sizes() ==
            smith::IntArrayRef(
                {options.num_embeddings(), options.embedding_dim()}),
        "Shape of weight does not match num_embeddings and embedding_dim");
    weight = register_parameter("weight", options._weight());
  }
}

void EmbeddingBagImpl::reset_parameters() {
  auto const& padding_idx_opt = options.padding_idx();
  if (padding_idx_opt.has_value()) {
    smith::NoGradGuard no_grad;
    weight[*padding_idx_opt].fill_(0);
  }
  smith::nn::init::normal_(weight);
}

smith::Tensor EmbeddingBagImpl::forward(
    const Tensor& input,
    const Tensor& offsets,
    const Tensor& per_sample_weights) {
  return F::detail::embedding_bag(
      input,
      weight,
      offsets,
      options.max_norm(),
      options.norm_type(),
      options.scale_grad_by_freq(),
      options.mode(),
      options.sparse(),
      per_sample_weights,
      options.include_last_offset(),
      options.padding_idx());
}

void EmbeddingBagImpl::pretty_print(std::ostream& stream) const {
  stream << "smith::nn::EmbeddingBag(num_embeddings="
         << options.num_embeddings()
         << ", embedding_dim=" << options.embedding_dim();
  auto const& max_norm_opt = options.max_norm();
  if (max_norm_opt.has_value()) {
    stream << ", max_norm=" << *max_norm_opt;
  }
  if (options.norm_type() != 2) {
    stream << ", norm_type=" << options.norm_type();
  }
  if (options.scale_grad_by_freq()) {
    stream << ", scale_grad_by_freq=" << std::boolalpha
           << options.scale_grad_by_freq();
  }
  if (options.sparse()) {
    stream << ", sparse=" << std::boolalpha << options.sparse();
  }
  if (!std::get_if<enumtype::kMean>(&options.mode())) {
    stream << ", mode=" << smith::enumtype::get_enum_name(options.mode());
  }
  if (options.include_last_offset()) {
    stream << ", include_last_offset=" << std::boolalpha
           << options.include_last_offset();
  }
  auto const& padding_idx_opt = options.padding_idx();
  if (padding_idx_opt.has_value()) {
    stream << ", padding_idx=" << padding_idx_opt.value();
  }
  stream << ')';
}
} // namespace smith::nn
