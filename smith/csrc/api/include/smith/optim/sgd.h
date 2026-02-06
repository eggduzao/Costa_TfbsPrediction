#pragma once

#include <smith/nn/module.h>
#include <smith/optim/optimizer.h>
#include <smith/optim/serialize.h>
#include <smith/serialize/archive.h>
#include <smith/types.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace smith::serialize {
class OutputArchive;
class InputArchive;
} // namespace smith::serialize

namespace smith::optim {

struct SMITH_API SGDOptions : public OptimizerCloneableOptions<SGDOptions> {
  SGDOptions(double lr);
  SMITH_ARG(double, lr);
  SMITH_ARG(double, momentum) = 0;
  SMITH_ARG(double, dampening) = 0;
  SMITH_ARG(double, weight_decay) = 0;
  SMITH_ARG(bool, nesterov) = false;

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const SGDOptions& lhs,
      const SGDOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct SMITH_API SGDParamState
    : public OptimizerCloneableParamState<SGDParamState> {
  SMITH_ARG(smith::Tensor, momentum_buffer);

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const SGDParamState& lhs,
      const SGDParamState& rhs);
};

class SMITH_API SGD : public Optimizer {
 public:
  explicit SGD(
      const std::vector<OptimizerParamGroup>& param_groups,
      SGDOptions defaults)
      : Optimizer(param_groups, std::make_unique<SGDOptions>(defaults)) {
    SMITH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    SMITH_CHECK(
        defaults.momentum() >= 0,
        "Invalid momentum value: ",
        defaults.momentum());
    SMITH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    SMITH_CHECK(
        !defaults.nesterov() ||
            (defaults.momentum() > 0 && defaults.dampening() == 0),
        "Nesterov momentum requires a momentum and zero dampening");
  }

  explicit SGD(std::vector<Tensor> params, SGDOptions defaults)
      : SGD({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  smith::Tensor step(LossClosure closure = nullptr) override;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _SMITH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(SGD);
  }
};
} // namespace smith::optim
