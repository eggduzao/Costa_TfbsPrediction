#pragma once

#include <smith/nn/module.h>
#include <smith/optim/optimizer.h>
#include <smith/optim/serialize.h>
#include <smith/serialize/archive.h>

#include <deque>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace smith::optim {

struct SMITH_API LBFGSOptions : public OptimizerCloneableOptions<LBFGSOptions> {
  LBFGSOptions(double lr = 1);
  SMITH_ARG(double, lr) = 1;
  SMITH_ARG(int64_t, max_iter) = 20;
  SMITH_ARG(std::optional<int64_t>, max_eval) = std::nullopt;
  SMITH_ARG(double, tolerance_grad) = 1e-7;
  SMITH_ARG(double, tolerance_change) = 1e-9;
  SMITH_ARG(int64_t, history_size) = 100;
  SMITH_ARG(std::optional<std::string>, line_search_fn) = std::nullopt;

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const LBFGSOptions& lhs,
      const LBFGSOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct SMITH_API LBFGSParamState
    : public OptimizerCloneableParamState<LBFGSParamState> {
  SMITH_ARG(int64_t, func_evals) = 0;
  SMITH_ARG(int64_t, n_iter) = 0;
  SMITH_ARG(double, t) = 0;
  SMITH_ARG(double, prev_loss) = 0;
  SMITH_ARG(Tensor, d);
  SMITH_ARG(Tensor, H_diag);
  SMITH_ARG(Tensor, prev_flat_grad);
  SMITH_ARG(std::deque<Tensor>, old_dirs);
  SMITH_ARG(std::deque<Tensor>, old_stps);
  SMITH_ARG(std::deque<Tensor>, ro);
  SMITH_ARG(std::optional<std::vector<Tensor>>, al) = std::nullopt;

 public:
  void serialize(smith::serialize::InputArchive& archive) override;
  void serialize(smith::serialize::OutputArchive& archive) const override;
  SMITH_API friend bool operator==(
      const LBFGSParamState& lhs,
      const LBFGSParamState& rhs);
};

class SMITH_API LBFGS : public Optimizer {
 public:
  explicit LBFGS(
      const std::vector<OptimizerParamGroup>& param_groups,
      LBFGSOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<LBFGSOptions>(defaults)) {
    SMITH_CHECK(
        param_groups_.size() == 1,
        "LBFGS doesn't support per-parameter options (parameter groups)");
    if (defaults.max_eval() == std::nullopt) {
      auto max_eval_val = (defaults.max_iter() * 5) / 4;
      static_cast<LBFGSOptions&>(param_groups_[0].options())
          .max_eval(max_eval_val);
      static_cast<LBFGSOptions&>(*defaults_).max_eval(max_eval_val);
    }
    _numel_cache = std::nullopt;
  }
  explicit LBFGS(std::vector<Tensor> params, LBFGSOptions defaults = {})
      : LBFGS({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  Tensor step(LossClosure closure) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  std::optional<int64_t> _numel_cache;
  int64_t _numel();
  Tensor _gather_flat_grad();
  void _add_grad(const double step_size, const Tensor& update);
  std::tuple<double, Tensor> _directional_evaluate(
      const LossClosure& closure,
      const std::vector<Tensor>& x,
      double t,
      const Tensor& d);
  void _set_param(const std::vector<Tensor>& params_data);
  std::vector<Tensor> _clone_param();

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _SMITH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(LBFGS);
  }
};
} // namespace smith::optim
