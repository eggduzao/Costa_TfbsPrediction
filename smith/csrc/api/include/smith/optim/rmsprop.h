#pragma once

#include <smith/nn/module.h>
#include <smith/optim/optimizer.h>
#include <smith/optim/serialize.h>
#include <smith/serialize/archive.h>
#include <smith/types.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace smith::serialize {
class OutputArchive;
class InputArchive;
} // namespace smith::serialize

namespace smith::optim {

struct SMITH_API RMSpropOptions
    : public OptimizerCloneableOptions<RMSpropOptions> {
  RMSpropOptions(double lr = 1e-2);
  SMITH_ARG(double, lr) = 1e-2;
  SMITH_ARG(double, alpha) = 0.99;
  SMITH_ARG(double, eps) = 1e-8;
  SMITH_ARG(double, weight_decay) = 0;
  SMITH_ARG(double, momentum) = 0;
  SMITH_ARG(bool, centered) = false;

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const RMSpropOptions& lhs,
      const RMSpropOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct SMITH_API RMSpropParamState
    : public OptimizerCloneableParamState<RMSpropParamState> {
  SMITH_ARG(int64_t, step) = 0;
  SMITH_ARG(smith::Tensor, square_avg);
  SMITH_ARG(smith::Tensor, momentum_buffer);
  SMITH_ARG(smith::Tensor, grad_avg);

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const RMSpropParamState& lhs,
      const RMSpropParamState& rhs);
};

class SMITH_API RMSprop : public Optimizer {
 public:
  explicit RMSprop(
      const std::vector<OptimizerParamGroup>& param_groups,
      RMSpropOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<RMSpropOptions>(defaults)) {
    SMITH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    SMITH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    SMITH_CHECK(
        defaults.momentum() >= 0,
        "Invalid momentum value: ",
        defaults.momentum());
    SMITH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    SMITH_CHECK(
        defaults.alpha() >= 0, "Invalid alpha value: ", defaults.alpha());
  }

  explicit RMSprop(std::vector<Tensor> params, RMSpropOptions defaults = {})
      : RMSprop({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {
  }

  smith::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _SMITH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(RMSprop);
  }
};
} // namespace smith::optim
