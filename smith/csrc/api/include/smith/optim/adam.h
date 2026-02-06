#pragma once

#include <smith/nn/module.h>
#include <smith/optim/optimizer.h>
#include <smith/optim/serialize.h>

#include <utility>
#include <vector>

namespace smith::serialize {
class OutputArchive;
class InputArchive;
} // namespace smith::serialize

namespace smith::optim {

struct SMITH_API AdamOptions : public OptimizerCloneableOptions<AdamOptions> {
  AdamOptions(double lr = 1e-3);
  SMITH_ARG(double, lr) = 1e-3;
  typedef std::tuple<double, double> betas_t;
  SMITH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  SMITH_ARG(double, eps) = 1e-8;
  SMITH_ARG(double, weight_decay) = 0;
  SMITH_ARG(bool, amsgrad) = false;

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const AdamOptions& lhs,
      const AdamOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct SMITH_API AdamParamState
    : public OptimizerCloneableParamState<AdamParamState> {
  SMITH_ARG(int64_t, step) = 0;
  SMITH_ARG(smith::Tensor, exp_avg);
  SMITH_ARG(smith::Tensor, exp_avg_sq);
  SMITH_ARG(smith::Tensor, max_exp_avg_sq);

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const AdamParamState& lhs,
      const AdamParamState& rhs);
};

class SMITH_API Adam : public Optimizer {
 public:
  explicit Adam(
      const std::vector<OptimizerParamGroup>& param_groups,
      AdamOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<AdamOptions>(defaults)) {
    SMITH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    SMITH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    SMITH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));
    SMITH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));
    SMITH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
  }
  explicit Adam(std::vector<Tensor> params, AdamOptions defaults = {})
      : Adam({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  smith::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _SMITH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adam);
  }
};
} // namespace smith::optim
