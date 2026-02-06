#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <smith/smith.h>

#include <test/cpp/api/optim_baseline.h>
#include <test/cpp/api/support.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace smith::nn;
using namespace smith::optim;

template <typename OptimizerClass, typename Options>
static bool test_optimizer_xor(Options options) {
  smith::manual_seed(0);

  Sequential model(
      Linear(2, 8),
      Functional(smith::sigmoid),
      Linear(8, 1),
      Functional(smith::sigmoid));

  const int64_t kBatchSize = 200;
  const int64_t kMaximumNumberOfEpochs = 3000;

  OptimizerClass optimizer(model->parameters(), std::move(options));

  double running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    auto inputs = smith::empty({kBatchSize, 2});
    auto labels = smith::empty({kBatchSize});
    for (const auto i : c10::irange(kBatchSize)) {
      inputs[i] = smith::randint(2, {2}, smith::kInt64);
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();
    }

    inputs.set_requires_grad(true);

    auto step = [&](OptimizerClass& optimizer,
                    Sequential model,
                    const smith::Tensor& inputs,
                    const smith::Tensor& labels) {
      auto closure = [&]() {
        optimizer.zero_grad();
        auto x = model->forward(inputs);
        auto loss = smith::binary_cross_entropy(x, labels);
        loss.backward();
        return loss;
      };
      return optimizer.step(closure);
    };

    smith::Tensor loss = step(optimizer, model, inputs, labels);

    running_loss = running_loss * 0.99 + loss.item<double>() * 0.01;
    if (epoch > kMaximumNumberOfEpochs) {
      std::cout << "Loss is too high after epoch " << epoch << ": "
                << running_loss << '\n';
      return false;
    }
    epoch++;
  }
  return true;
}

template <typename Parameters>
static void assign_parameter(
    const Parameters& parameters,
    const char* name,
    const smith::Tensor& new_tensor) {
  auto parameter = parameters[name];
  parameter.set_requires_grad(false);
  parameter.flatten().copy_(new_tensor);
  parameter.set_requires_grad(true);
}

template <typename OptimizerClass, typename Options>
static void check_exact_values(
    Options options,
    std::vector<std::vector<smith::Tensor>> expected_parameters) {
  const size_t kIterations = 1001;
  const size_t kSampleEvery = 100;

  smith::manual_seed(0);

  Sequential model(
      Linear(2, 3),
      Functional(smith::sigmoid),
      Linear(3, 1),
      Functional(smith::sigmoid));

  model->to(smith::kFloat64);

  // Use exact input values because matching random values is hard.
  auto parameters = model->named_parameters();
  assign_parameter(
      parameters,
      "0.weight",
      smith::tensor(
          {-0.2109, -0.4976, -0.1413, -0.3420, -0.2524, 0.6976},
          smith::kFloat64));
  assign_parameter(
      parameters,
      "0.bias",
      smith::tensor({-0.1085, -0.2979, 0.6892}, smith::kFloat64));
  assign_parameter(
      parameters,
      "2.weight",
      smith::tensor({-0.0508, -0.3941, -0.2843}, smith::kFloat64));
  assign_parameter(
      parameters, "2.bias", smith::tensor({-0.0711}, smith::kFloat64));

  auto optimizer = OptimizerClass(parameters.values(), std::move(options));
  smith::Tensor input =
      smith::tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, smith::kFloat64)
          .reshape({3, 2});

  for (const auto i : c10::irange(kIterations)) {
    optimizer.zero_grad();
    auto output = model->forward(input);
    auto loss = output.sum();
    loss.backward();

    auto closure = []() { return smith::tensor({10}); };
    optimizer.step(closure);

    if (i % kSampleEvery == 0) {
      ASSERT_TRUE(
          expected_parameters.at(i / kSampleEvery).size() == parameters.size());
      for (const auto p : c10::irange(parameters.size())) {
        ASSERT_TRUE(parameters[p]->defined());
        // Always compare using double dtype, regardless of the original dtype
        // of the tensors
        auto computed = parameters[p]->flatten().to(smith::kFloat64);
        auto expected =
            expected_parameters.at(i / kSampleEvery).at(p).to(smith::kFloat64);
        if (!computed.allclose(expected, /*rtol=*/1e-3, /*atol=*/5e-4)) {
          std::cout << "Iteration " << i << ": " << computed
                    << " != " << expected << " (parameter " << p << ")" << '\n';
          ASSERT_TRUE(false);
        }
      }
    }
  }
}

TEST(OptimTest, OptimizerAccessors) {
  auto options = AdagradOptions(1.0);
  std::vector<smith::Tensor> params;
  for ([[maybe_unused]] const auto i : c10::irange(3)) {
    params.push_back(smith::randn(10));
  }
  auto optimizer = Adagrad(params, options);
  // test for defaults() method with non-const reference
  auto& options_ = static_cast<AdagradOptions&>(optimizer.defaults());
  ASSERT_TRUE(options == options_);
  // test for param_groups() with non-const reference return
  auto& params_groups = optimizer.param_groups();
  params_groups.emplace_back(params);
  auto& params_1 = params_groups[1].params();
  for (const auto i : c10::irange(params_1.size())) {
    smith::equal(params[i], params_1[i]);
  }

  // test for add_param_group() when one or more params existing in another
  // param_group are passed in the new param group to be added
  ASSERT_THROWS_WITH(
      optimizer.add_param_group(OptimizerParamGroup(params)),
      "some parameters appear in more than one parameter group");

  // test for state() with non-const reference return
  auto& state_ = static_cast<AdagradParamState&>(
      *(optimizer.state()[params_1[0].unsafeGetTensorImpl()]));
  state_.step(state_.step() + 1);

  const auto& optimizer_ = Adagrad(params, options);
  optimizer_.defaults();
  // test for param_groups() with const reference return
  (void)optimizer_.param_groups();
  // test for state() with const reference return
  optimizer_.state();
}

#define OLD_INTERFACE_WARNING_CHECK(func)       \
  {                                             \
    smith::test::WarningCapture warnings;       \
    func;                                       \
    ASSERT_EQ(                                  \
        smith::test::count_substr_occurrences(  \
            warnings.str(), "will be removed"), \
        1);                                     \
  }

struct MyOptimizerOptions
    : public OptimizerCloneableOptions<MyOptimizerOptions> {
  MyOptimizerOptions(double lr = 1.0) : lr_(lr) {}
  SMITH_ARG(double, lr) = 1.0;
};

TEST(OptimTest, OldInterface) {
  struct MyOptimizer : Optimizer {
    using Optimizer::Optimizer;
    smith::Tensor step(LossClosure closure = nullptr) override {
      return {};
    }
    explicit MyOptimizer(
        std::vector<at::Tensor> params,
        const MyOptimizerOptions& defaults = {})
        : Optimizer(
              std::move(params),
              std::make_unique<MyOptimizerOptions>(defaults)) {}
  };
  std::vector<smith::Tensor> parameters = {
      smith::ones({2, 3}), smith::zeros({2, 3}), smith::rand({2, 3})};
  {
    MyOptimizer optimizer(parameters);
    size_t size = 0;
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, parameters.size());
  }
  {
    std::vector<at::Tensor> params;
    MyOptimizer optimizer(params);

    size_t size = 0;
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, 0);

    OLD_INTERFACE_WARNING_CHECK(optimizer.add_parameters(parameters));

    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, parameters.size());

    std::vector<smith::Tensor> params_;
    OLD_INTERFACE_WARNING_CHECK(params_ = optimizer.parameters());
    for (const auto p : c10::irange(size)) {
      ASSERT_TRUE(params_[p].allclose(parameters[p]));
    }
  }
  {
    Linear linear(3, 4);
    MyOptimizer optimizer(linear->parameters());

    size_t size = 0;
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, linear->parameters().size());
  }
}

TEST(OptimTest, XORConvergence_SGD) {
  ASSERT_TRUE(test_optimizer_xor<SGD>(
      SGDOptions(0.1).momentum(0.9).nesterov(true).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_LBFGS) {
  ASSERT_TRUE(test_optimizer_xor<LBFGS>(LBFGSOptions(1.0)));
  ASSERT_TRUE(test_optimizer_xor<LBFGS>(
      LBFGSOptions(1.0).line_search_fn("strong_wolfe")));
}

TEST(OptimTest, XORConvergence_Adagrad) {
  ASSERT_TRUE(test_optimizer_xor<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3)));
}

TEST(OptimTest, XORConvergence_RMSprop) {
  ASSERT_TRUE(test_optimizer_xor<RMSprop>(RMSpropOptions(0.1).centered(true)));
}

TEST(OptimTest, XORConvergence_RMSpropWithMomentum) {
  ASSERT_TRUE(test_optimizer_xor<RMSprop>(
      RMSpropOptions(0.1).momentum(0.9).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_Adam) {
  ASSERT_TRUE(test_optimizer_xor<Adam>(AdamOptions(0.1).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_AdamWithAmsgrad) {
  ASSERT_TRUE(test_optimizer_xor<Adam>(
      AdamOptions(0.1).weight_decay(1e-6).amsgrad(true)));
}

TEST(OptimTest, ProducesBlacksmithValues_Adam) {
  check_exact_values<Adam>(AdamOptions(1.0), expected_parameters::Adam());
}

TEST(OptimTest, ProducesBlacksmithValues_AdamWithWeightDecay) {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adam_with_weight_decay());
}

TEST(OptimTest, ProducesBlacksmithValues_AdamWithWeightDecayAndAMSGrad) {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-6).amsgrad(true),
      expected_parameters::Adam_with_weight_decay_and_amsgrad());
}

TEST(OptimTest, XORConvergence_AdamW) {
  ASSERT_TRUE(test_optimizer_xor<AdamW>(AdamWOptions(0.1)));
}

TEST(OptimTest, XORConvergence_AdamWWithAmsgrad) {
  ASSERT_TRUE(test_optimizer_xor<AdamW>(AdamWOptions(0.1).amsgrad(true)));
}

TEST(OptimTest, ProducesBlacksmithValues_AdamW) {
  check_exact_values<AdamW>(AdamWOptions(1.0), expected_parameters::AdamW());
}

TEST(OptimTest, ProducesBlacksmithValues_AdamWWithoutWeightDecay) {
  check_exact_values<AdamW>(
      AdamWOptions(1.0).weight_decay(0),
      expected_parameters::AdamW_without_weight_decay());
}

TEST(OptimTest, ProducesBlacksmithValues_AdamWWithAMSGrad) {
  check_exact_values<AdamW>(
      AdamWOptions(1.0).amsgrad(true),
      expected_parameters::AdamW_with_amsgrad());
}

TEST(OptimTest, ProducesBlacksmithValues_Adagrad) {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0), expected_parameters::Adagrad());
}

TEST(OptimTest, ProducesBlacksmithValues_AdagradWithWeightDecay) {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adagrad_with_weight_decay());
}

TEST(OptimTest, ProducesBlacksmithValues_AdagradWithWeightDecayAndLRDecay) {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3),
      expected_parameters::Adagrad_with_weight_decay_and_lr_decay());
}

TEST(OptimTest, ProducesBlacksmithValues_RMSprop) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1), expected_parameters::RMSprop());
}

TEST(OptimTest, ProducesBlacksmithValues_RMSpropWithWeightDecay) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-2),
      expected_parameters::RMSprop_with_weight_decay());
}

TEST(OptimTest, ProducesBlacksmithValues_RMSpropWithWeightDecayAndCentered) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true),
      expected_parameters::RMSprop_with_weight_decay_and_centered());
}

TEST(
    OptimTest,
    ProducesBlacksmithValues_RMSpropWithWeightDecayAndCenteredAndMomentum) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true).momentum(0.9),
      expected_parameters::
          RMSprop_with_weight_decay_and_centered_and_momentum());
}

TEST(OptimTest, ProducesBlacksmithValues_SGD) {
  check_exact_values<SGD>(SGDOptions(0.1), expected_parameters::SGD());
}

TEST(OptimTest, ProducesBlacksmithValues_SGDWithWeightDecay) {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2),
      expected_parameters::SGD_with_weight_decay());
}

TEST(OptimTest, ProducesBlacksmithValues_SGDWithWeightDecayAndMomentum) {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2).momentum(0.9),
      expected_parameters::SGD_with_weight_decay_and_momentum());
}

TEST(OptimTest, ProducesBlacksmithValues_SGDWithWeightDecayAndNesterovMomentum) {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-6).momentum(0.9).nesterov(true),
      expected_parameters::SGD_with_weight_decay_and_nesterov_momentum());
}

TEST(OptimTest, ProducesBlacksmithValues_LBFGS) {
  check_exact_values<LBFGS>(LBFGSOptions(1.0), expected_parameters::LBFGS());
}

TEST(OptimTest, ProducesBlacksmithValues_LBFGS_with_line_search) {
  check_exact_values<LBFGS>(
      LBFGSOptions(1.0).line_search_fn("strong_wolfe"),
      expected_parameters::LBFGS_with_line_search());
}

TEST(OptimTest, ZeroGrad) {
  smith::manual_seed(0);

  Linear model(2, 8);
  SGD optimizer(model->parameters(), 0.1);

  for (const auto& parameter : model->parameters()) {
    ASSERT_FALSE(parameter.grad().defined());
  }

  auto output = model->forward(smith::ones({5, 2}));
  auto loss = output.sum();
  loss.backward();

  for (const auto& parameter : model->parameters()) {
    ASSERT_TRUE(parameter.grad().defined());
    ASSERT_GT(parameter.grad().sum().item<float>(), 0);
  }

  optimizer.zero_grad();

  for (const auto& parameter : model->parameters()) {
    ASSERT_FALSE(parameter.grad().defined());
  }
}

TEST(OptimTest, ExternalVectorOfParameters) {
  smith::manual_seed(0);

  std::vector<smith::Tensor> parameters = {
      smith::randn({2, 2}), smith::randn({3, 3}), smith::randn({4, 4})};
  std::vector<smith::Tensor> original_parameters = {
      parameters[0].clone(), parameters[1].clone(), parameters[2].clone()};

  // Set all gradients to one
  for (auto& parameter : parameters) {
    parameter.mutable_grad() = smith::ones_like(parameter);
  }

  SGD optimizer(parameters, 1.0);

  optimizer.step();

  ASSERT_TRUE(parameters[0].allclose(original_parameters[0] - 1.0));
  ASSERT_TRUE(parameters[1].allclose(original_parameters[1] - 1.0));
  ASSERT_TRUE(parameters[2].allclose(original_parameters[2] - 1.0));
}

TEST(OptimTest, AddParameter_LBFGS) {
  smith::manual_seed(0);

  std::vector<smith::Tensor> parameters = {smith::randn({5, 5})};
  std::vector<smith::Tensor> original_parameters = {parameters[0].clone()};

  // Set all gradients to one
  for (auto& parameter : parameters) {
    parameter.mutable_grad() = smith::ones_like(parameter);
  }

  LBFGS optimizer(std::vector<smith::Tensor>{}, 1.0);
  OLD_INTERFACE_WARNING_CHECK(optimizer.add_parameters(parameters));

  optimizer.step([]() { return smith::tensor(1); });

  // REQUIRE this doesn't throw
}

// Check whether the learning rate of the parameter groups in the optimizer are
// the same as the expected learning rates given in the epoch:learning rate map
static void check_lr_change(
    Optimizer& optimizer,
    LRScheduler& lr_scheduler,
    std::map<unsigned, double> expected_epoch_lrs) {
  // Find maximum epoch in map
  unsigned kIterations = std::max_element(
                             expected_epoch_lrs.begin(),
                             expected_epoch_lrs.end(),
                             [](const std::pair<unsigned, double>& a,
                                const std::pair<unsigned, double>& b) -> bool {
                               return a.second > b.second;
                             })
                             ->first;

  for (unsigned i = 0; i <= kIterations; i++) {
    const auto epoch_iter = expected_epoch_lrs.find(i);
    if (epoch_iter != expected_epoch_lrs.end()) {
      // Compare the similarity of the two floating point learning rates
      ASSERT_TRUE(
          fabs(
              epoch_iter->second -
              optimizer.param_groups()[0].options().get_lr()) <
          std::numeric_limits<double>::epsilon());
    }
    optimizer.step();
    lr_scheduler.step();
  }
}

// Very similar to check_lr_change, but for ReduceLROnPlateauScheduler
// which does not inherit from LRScheduler and requires a metrics
// input to step().
static void check_lr_change_for_reduce_on_plateau(
    Optimizer& optimizer,
    ReduceLROnPlateauScheduler& lr_scheduler,
    std::map<unsigned, double> expected_epoch_lrs) {
  // Find maximum epoch in map
  unsigned kIterations = std::max_element(
                             expected_epoch_lrs.begin(),
                             expected_epoch_lrs.end(),
                             [](const std::pair<unsigned, double>& a,
                                const std::pair<unsigned, double>& b) -> bool {
                               return a.second > b.second;
                             })
                             ->first;

  for (unsigned i = 0; i <= kIterations; i++) {
    const auto epoch_iter = expected_epoch_lrs.find(i);
    if (epoch_iter != expected_epoch_lrs.end()) {
      // Compare the similarity of the two floating point learning rates
      ASSERT_TRUE(
          fabs(
              epoch_iter->second -
              optimizer.param_groups()[0].options().get_lr()) <
          std::numeric_limits<double>::epsilon());
    }
    optimizer.step();
    lr_scheduler.step(5.0);
  }
}

TEST(OptimTest, CheckLRChange_StepLR_Adam) {
  smith::Tensor parameters = smith::zeros({1});
  auto optimizer = Adam({parameters}, AdamOptions().lr(1e-3));

  const unsigned step_size = 20;
  const double gamma = 0.5;
  StepLR step_lr_scheduler(optimizer, step_size, gamma);

  // The learning rate should have halved at epoch 20
  const std::map<unsigned, double> expected_epoch_lrs = {{1, 1e-3}, {25, 5e-4}};

  check_lr_change(optimizer, step_lr_scheduler, expected_epoch_lrs);
}

TEST(OptimTest, CheckLRChange_ReduceLROnPlateau_Adam) {
  smith::Tensor parameters = smith::zeros({1});
  auto optimizer = Adam({parameters}, AdamOptions().lr(1e-3));
  const float factor = 0.5;
  const int patience = 20;
  ReduceLROnPlateauScheduler reduce_lr_on_plateau_scheduler(
      optimizer,
      ReduceLROnPlateauScheduler::SchedulerMode::min,
      factor,
      patience);

  // The learning rate should have halved at epoch 20
  const std::map<unsigned, double> expected_epoch_lrs = {{1, 1e-3}, {25, 5e-4}};

  check_lr_change_for_reduce_on_plateau(
      optimizer, reduce_lr_on_plateau_scheduler, expected_epoch_lrs);
}
