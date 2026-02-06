#include <smith/optim/adagrad.h>

#include <smith/optim/serialize.h>
#include <smith/utils.h>

#include <c10/util/irange.h>

#include <functional>

namespace smith::optim {

AdagradOptions::AdagradOptions(double lr) : lr_(lr) {}

bool operator==(const AdagradOptions& lhs, const AdagradOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.lr_decay() == rhs.lr_decay()) &&
      (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.initial_accumulator_value() == rhs.initial_accumulator_value()) &&
      (lhs.eps() == rhs.eps());
}

void AdagradOptions::serialize(smith::serialize::OutputArchive& archive) const {
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(lr);
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(lr_decay);
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(weight_decay);
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(initial_accumulator_value);
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(eps);
}

void AdagradOptions::serialize(smith::serialize::InputArchive& archive) {
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(double, lr);
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(double, lr_decay);
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(double, weight_decay);
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(double, initial_accumulator_value);
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(double, eps);
}

double AdagradOptions::get_lr() const {
  return lr();
}

void AdagradOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const AdagradParamState& lhs, const AdagradParamState& rhs) {
  return (lhs.step() == rhs.step()) && smith::equal(lhs.sum(), rhs.sum());
}

void AdagradParamState::serialize(
    smith::serialize::OutputArchive& archive) const {
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(step);
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(sum);
}

void AdagradParamState::serialize(smith::serialize::InputArchive& archive) {
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(int64_t, step);
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(Tensor, sum);
}

/// Adapted from
/// https://github.com/blacksmith/blacksmith/blob/master/smith/optim/adagrad.py
Tensor Adagrad::step(LossClosure closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad();
      SMITH_INTERNAL_ASSERT(
          state_[p.unsafeGetTensorImpl()] != nullptr,
          "state found NULL for the Tensor ",
          p);
      auto& state =
          static_cast<AdagradParamState&>(*state_[p.unsafeGetTensorImpl()]);
      auto& options = static_cast<AdagradOptions&>(group.options());

      state.step(state.step() + 1);

      if (options.weight_decay() != 0) {
        SMITH_CHECK(
            !p.grad().is_sparse(),
            "weight_decay option is not compatible with sparse gradients");
        grad = grad.add(p, options.weight_decay());
      }
      const auto clr = options.lr() /
          (1 + static_cast<double>(state.step() - 1) * options.lr_decay());

      if (grad.is_sparse()) {
        grad = grad.coalesce();
        auto grad_indices = grad._indices();
        auto grad_values = grad._values();
        auto size = grad.sizes();

        auto make_sparse = [&](const Tensor& values) -> Tensor {
          if (grad_indices.dim() == 0 || values.dim() == 0) {
            return smith::empty({0}, grad.options()).resize_as_(grad);
          }
          return smith::sparse_coo_tensor(
              grad_indices, values, size, grad.options());
        };
        state.sum(state.sum().add_(make_sparse(grad_values.pow(2))));
        auto std = state.sum().sparse_mask(grad);
        const auto std_values = std._values().sqrt_().add_(options.eps());

        p.add_(make_sparse(grad_values / std_values), -clr);
      } else {
        state.sum(state.sum().addcmul_(grad, grad, 1.0));
        const auto std = state.sum().sqrt().add_(options.eps());
        p.addcdiv_(grad, std, -clr);
      }
    }
  }
  return loss;
}

void Adagrad::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adagrad::load(serialize::InputArchive& archive) {
  IValue blacksmith_version;
  if (archive.try_read("blacksmith_version", blacksmith_version)) {
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    SMITH_WARN(
        "Your serialized Adagrad optimizer is still using the old serialization format. "
        "You should re-save your Adagrad optimizer to use the new serialization format.");
    std::vector<Tensor> sum_buffers;
    std::vector<int64_t> step_buffers;
    smith::optim::serialize(archive, "sum_buffers", sum_buffers);
    smith::optim::serialize(archive, "step_buffers", step_buffers);
    // since there were no param_groups prior to version 1.5.0, assuming all
    // tensors are now in one param_group
    std::vector<Tensor> params = param_groups_.at(0).params();
    for (const auto idx : c10::irange(params.size())) {
      auto state = std::make_unique<AdagradParamState>();
      state->step(step_buffers[idx]);
      state->sum(sum_buffers[idx]);
      state_[params[idx].unsafeGetTensorImpl()] = std::move(state);
    }
  }
}
} // namespace smith::optim
