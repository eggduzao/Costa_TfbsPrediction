#include <smith/optim/sgd.h>

#include <smith/optim/optimizer.h>
#include <smith/optim/serialize.h>
#include <smith/utils.h>

#include <c10/util/irange.h>

#include <functional>

namespace smith::optim {

SGDOptions::SGDOptions(double lr) : lr_(lr) {}

bool operator==(const SGDOptions& lhs, const SGDOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.momentum() == rhs.momentum()) &&
      (lhs.dampening() == rhs.dampening()) &&
      (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.nesterov() == rhs.nesterov());
}

void SGDOptions::serialize(smith::serialize::OutputArchive& archive) const {
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(lr);
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(momentum);
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(dampening);
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(weight_decay);
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(nesterov);
}

void SGDOptions::serialize(smith::serialize::InputArchive& archive) {
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(double, lr);
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(double, momentum);
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(double, dampening);
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(double, weight_decay);
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(bool, nesterov);
}

double SGDOptions::get_lr() const {
  return lr();
}

void SGDOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const SGDParamState& lhs, const SGDParamState& rhs) {
  return smith::equal(lhs.momentum_buffer(), rhs.momentum_buffer());
}

void SGDParamState::serialize(smith::serialize::OutputArchive& archive) const {
  _SMITH_OPTIM_SERIALIZE_SMITH_ARG(momentum_buffer);
}

void SGDParamState::serialize(smith::serialize::InputArchive& archive) {
  _SMITH_OPTIM_DESERIALIZE_SMITH_ARG(Tensor, momentum_buffer);
}

Tensor SGD::step(LossClosure closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  for (auto& group : param_groups_) {
    auto& options = static_cast<SGDOptions&>(group.options());
    auto weight_decay = options.weight_decay();
    auto momentum = options.momentum();
    auto dampening = options.dampening();
    auto nesterov = options.nesterov();

    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto d_p = p.grad().data();
      if (weight_decay != 0) {
        d_p = d_p.add(p.data(), weight_decay);
      }
      if (momentum != 0) {
        Tensor buf;
        auto param_state = state_.find(p.unsafeGetTensorImpl());
        if (param_state == state_.end()) {
          buf = d_p.detach().clone();
          auto state = std::make_unique<SGDParamState>();
          state->momentum_buffer(buf);
          state_[p.unsafeGetTensorImpl()] = std::move(state);
        } else {
          buf = static_cast<SGDParamState&>(*param_state->second)
                    .momentum_buffer();
          buf.mul_(momentum).add_(d_p, 1 - dampening);
        }
        if (nesterov) {
          d_p = d_p.add(buf, momentum);
        } else {
          d_p = buf;
        }
      }
      p.data().add_(d_p, -1 * options.lr());
    }
  }
  return loss;
}

void SGD::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void SGD::load(serialize::InputArchive& archive) {
  IValue blacksmith_version;
  if (archive.try_read("blacksmith_version", blacksmith_version)) {
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    SMITH_WARN(
        "Your serialized SGD optimizer is still using the old serialization format. "
        "You should re-save your SGD optimizer to use the new serialization format.");
    std::vector<Tensor> momentum_buffers;
    smith::optim::serialize(archive, "momentum_buffers", momentum_buffers);
    // since there were no param_groups prior to version 1.5.0, assuming all
    // tensors are now in one param_group
    std::vector<Tensor> params = param_groups_.at(0).params();
    for (const auto idx : c10::irange(momentum_buffers.size())) {
      auto state = std::make_unique<SGDParamState>();
      state->momentum_buffer(momentum_buffers[idx]);
      state_[params[idx].unsafeGetTensorImpl()] = std::move(state);
    }
  }
}
} // namespace smith::optim
