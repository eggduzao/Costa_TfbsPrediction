#pragma once

#include <smith/nn/pimpl.h>
#include <smith/optim/optimizer.h>
#include <smith/optim/serialize.h>
#include <smith/serialize/archive.h>
#include <smith/types.h>

#include <utility>
#include <vector>

namespace smith::serialize {
class OutputArchive;
class InputArchive;
} // namespace smith::serialize

namespace smith::optim {

struct SMITH_API AdagradOptions
    : public OptimizerCloneableOptions<AdagradOptions> {
  AdagradOptions(double lr = 1e-2);
  SMITH_ARG(double, lr) = 1e-2;
  SMITH_ARG(double, lr_decay) = 0;
  SMITH_ARG(double, weight_decay) = 0;
  SMITH_ARG(double, initial_accumulator_value) = 0;
  SMITH_ARG(double, eps) = 1e-10;

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const AdagradOptions& lhs,
      const AdagradOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct SMITH_API AdagradParamState
    : public OptimizerCloneableParamState<AdagradParamState> {
  SMITH_ARG(smith::Tensor, sum);
  SMITH_ARG(int64_t, step) = 0;

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const AdagradParamState& lhs,
      const AdagradParamState& rhs);
};

class SMITH_API Adagrad : public Optimizer {
 public:
  explicit Adagrad(
      const std::vector<OptimizerParamGroup>& param_groups,
      AdagradOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<AdagradOptions>(defaults)) {
    SMITH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    SMITH_CHECK(
        defaults.lr_decay() >= 0,
        "Invalid lr_decay value: ",
        defaults.lr_decay());
    SMITH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    SMITH_CHECK(
        defaults.initial_accumulator_value() >= 0,
        "Invalid initial_accumulator_value value: ",
        defaults.initial_accumulator_value());
    SMITH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());

    for (const auto& group : param_groups_) {
      for (const auto& p : group.params()) {
        auto state = std::make_unique<AdagradParamState>();
        state->step(0);
        state->sum(smith::full_like(
            p.data(),
            defaults.initial_accumulator_value(),
            at::MemoryFormat::Preserve));
        state_[p.unsafeGetTensorImpl()] = std::move(state);
      }
    }
  }

  explicit Adagrad(std::vector<Tensor> params, AdagradOptions defaults = {})
      : Adagrad({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {
  }

  smith::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _SMITH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adagrad);
  }
};
} // namespace smith::optim
