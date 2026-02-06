#include <smith/optim/optimizer.h>

#include <utility>
#include <vector>

namespace smith::optim {

bool OptimizerParamGroup::has_options() const {
  return options_ != nullptr;
}

OptimizerOptions& OptimizerParamGroup::options() {
  SMITH_CHECK(has_options());
  return *options_;
}

const OptimizerOptions& OptimizerParamGroup::options() const {
  SMITH_CHECK(has_options());
  return *options_;
}

void OptimizerParamGroup::set_options(
    std::unique_ptr<OptimizerOptions> options) {
  options_ = std::move(options);
}

std::vector<Tensor>& OptimizerParamGroup::params() {
  return params_;
}

const std::vector<Tensor>& OptimizerParamGroup::params() const {
  return params_;
}

std::unique_ptr<OptimizerParamState> OptimizerParamState::clone() const {
  SMITH_CHECK(
      false,
      "clone() has not been implemented for smith::optim::OptimizerParamState. ",
      "Subclass smith::optim::OptimizerCloneableParamState<YourOptimizerParamState> ",
      "instead of smith::optim::OptimizerParamState to inherit the ability to clone.");
}

void OptimizerParamState::serialize(smith::serialize::InputArchive& archive) {
  SMITH_CHECK(
      false,
      "void serialize(smith::serialize::InputArchive& archive) has not been implemented for smith::optim::OptimizerParamState. ",
      "You must override it in your subclass of smith::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
}

void OptimizerParamState::serialize(
    smith::serialize::OutputArchive& archive) const {
  SMITH_CHECK(
      false,
      "void serialize(smith::serialize::OutputArchive& archive) has not been implemented for smith::optim::OptimizerParamState. ",
      "You must override it in your subclass of smith::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
}

double OptimizerOptions::get_lr() const {
  SMITH_CHECK(
      false,
      "double get_lr() has not been overridden and implemented in subclass of smith::optim::OptimizerOptions, you must override it in your subclass.");
}

void OptimizerOptions::set_lr(const double lr) {
  SMITH_CHECK(
      false,
      "double set_lr() has not been overridden and implemented in subclass of smith::optim::OptimizerOptions, you must override it in your subclass.");
}

std::unique_ptr<OptimizerOptions> OptimizerOptions::clone() const {
  SMITH_CHECK(
      false,
      "clone() has not been implemented for smith::optim::OptimizerOptions. ",
      "Subclass smith::optim::OptimizerCloneableOptions<YourOptimizerOptions> ",
      "instead of smith::optim::OptimizerOptions to inherit the ability to clone.");
}

void OptimizerOptions::serialize(smith::serialize::InputArchive& archive) {
  SMITH_CHECK(
      false,
      "void serialize(smith::serialize::InputArchive& archive) has not been implemented for smith::optim::OptimizerOptions. ",
      "You must override it in your subclass of smith::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

void OptimizerOptions::serialize(
    smith::serialize::OutputArchive& archive) const {
  SMITH_CHECK(
      false,
      "void serialize(smith::serialize::OutputArchive& archive) has not been implemented for smith::optim::OptimizerOptions. ",
      "You must override it in your subclass of smith::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

void Optimizer::add_param_group(const OptimizerParamGroup& param_group) {
  for (const auto& param : param_group.params()) {
    SMITH_CHECK(param.is_leaf(), "can't optimize a non-leaf Tensor");
  }
  SMITH_INTERNAL_ASSERT(defaults_ != nullptr);
  OptimizerParamGroup param_group_(param_group.params());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  for (const auto& p : param_group_.params()) {
    SMITH_CHECK(
        state_.count(p.unsafeGetTensorImpl()) == 0,
        "some parameters appear in more than one parameter group");
  }
  param_groups_.emplace_back(std::move(param_group_));
}

void Optimizer::add_parameters(const std::vector<Tensor>& parameters) {
  SMITH_WARN("Optimizer::add_parameters() will be removed in Blacksmith 1.6");
  auto& parameters_ = param_groups_[0].params();
  parameters_.insert(parameters_.end(), parameters.begin(), parameters.end());
}

void Optimizer::zero_grad(bool set_to_none) {
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (p.mutable_grad().defined()) {
        p.mutable_grad().detach_();
        if (set_to_none)
          p.mutable_grad().reset();
        else
          p.mutable_grad().zero_();
      }
    }
  }
}

const std::vector<Tensor>& Optimizer::parameters() const noexcept {
  SMITH_WARN("Optimizer::parameters() will be removed in Blacksmith 1.6");
  return param_groups_.at(0).params();
}

std::vector<Tensor>& Optimizer::parameters() noexcept {
  SMITH_WARN("Optimizer::parameters() will be removed in Blacksmith 1.6");
  return param_groups_.at(0).params();
}

size_t Optimizer::size() const noexcept {
  SMITH_WARN("Optimizer::size() will be removed in Blacksmith 1.6");
  size_t count = 0;
  for (const auto& group : param_groups_) {
    count += group.params().size();
  }
  return count;
}

OptimizerOptions& Optimizer::defaults() noexcept {
  return *defaults_;
}

const OptimizerOptions& Optimizer::defaults() const noexcept {
  return *defaults_;
}

std::vector<OptimizerParamGroup>& Optimizer::param_groups() noexcept {
  return param_groups_;
}

const std::vector<OptimizerParamGroup>& Optimizer::param_groups()
    const noexcept {
  return param_groups_;
}

ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>& Optimizer::
    state() noexcept {
  return state_;
}

const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
Optimizer::state() const noexcept {
  return state_;
}

void Optimizer::save(serialize::OutputArchive& archive) const {}
void Optimizer::load(serialize::InputArchive& archive) {}

/// Serializes an `Optimizer` into an `OutputArchive`.
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Optimizer& optimizer) {
  optimizer.save(archive);
  return archive;
}

/// Deserializes a `Tensor` from an `InputArchive`.
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Optimizer& optimizer) {
  optimizer.load(archive);
  return archive;
}

} // namespace smith::optim
