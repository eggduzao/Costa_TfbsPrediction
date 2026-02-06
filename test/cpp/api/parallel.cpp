#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <smith/csrc/autograd/functions/comm.h>
#include <smith/nn/module.h>
#include <smith/nn/modules/conv.h>
#include <smith/nn/modules/linear.h>
#include <smith/nn/parallel/data_parallel.h>
#include <smith/nn/pimpl.h>
#include <smith/optim/sgd.h>
#include <smith/types.h>
#include <smith/utils.h>

#include <test/cpp/api/support.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace smith::autograd;
using namespace smith::nn;

struct ParallelTest : smith::test::SeedingFixture {};

TEST_F(ParallelTest, DifferentiableScatter_MultiCUDA) {
  Scatter scatter(
      {smith::Device(smith::kCUDA, 0), smith::Device(smith::kCUDA, 1)});

  auto input = smith::ones(10, smith::requires_grad(true));
  auto output = scatter.apply({input});

  ASSERT_EQ(output.size(), 2);
  ASSERT_EQ(output[0].size(0), 5);
  ASSERT_EQ(output[1].size(0), 5);

  ASSERT_TRUE(smith::cat({output[0].to(smith::kCPU), output[1].to(smith::kCPU)})
                  .allclose(input));

  smith::Tensor sum = output[0].to({smith::kCUDA, 1}) + output[1];
  sum.backward(smith::ones_like(sum));

  ASSERT_TRUE(input.grad().defined());
  ASSERT_TRUE(input.grad().device().is_cpu());
  ASSERT_EQ(input.grad().sum().item<int32_t>(), 10);
}

TEST_F(ParallelTest, DifferentiableGather_MultiCUDA) {
  Gather gather(smith::Device(smith::kCUDA, 1));

  auto a = smith::ones(5, smith::requires_grad(true).device(smith::kCUDA, 0));
  auto b = smith::ones(5, smith::requires_grad(true).device(smith::kCUDA, 1));

  auto outputs = gather.apply({a, b});
  ASSERT_EQ(outputs.size(), 1);
  smith::Tensor output = outputs.front();

  ASSERT_EQ(output.size(0), 10);
  ASSERT_EQ(output.device(), smith::Device(smith::kCUDA, 1));

  auto chunks = output.chunk(2);
  ASSERT_TRUE(chunks[0].to({smith::kCUDA, 0}).allclose(a));
  ASSERT_TRUE(chunks[1].allclose(b));

  output.backward(smith::ones_like(output));

  ASSERT_TRUE(a.grad().defined());
  ASSERT_EQ(a.grad().device(), smith::Device(smith::kCUDA, 0));
  ASSERT_EQ(a.grad().sum().item<int32_t>(), 5);

  ASSERT_TRUE(b.grad().defined());
  ASSERT_EQ(b.grad().device(), smith::Device(smith::kCUDA, 1));
  ASSERT_EQ(b.grad().sum().item<int32_t>(), 5);
}

TEST_F(ParallelTest, Replicate_MultiCUDA) {
  Linear linear(3, 4);
  auto replicas = parallel::replicate(
      linear, {smith::Device(smith::kCUDA, 0), smith::Device(smith::kCUDA, 1)});
  ASSERT_EQ(replicas.size(), 2);

  auto original_parameters = linear->parameters();

  auto replica1_parameters = replicas[0]->parameters();
  for (auto& parameter : replica1_parameters) {
    ASSERT_EQ(parameter.device(), smith::Device(smith::kCUDA, 0));
  }
  replicas[0]->to(smith::kCPU);
  ASSERT_EQ(replica1_parameters.size(), original_parameters.size());
  for (const auto i : c10::irange(original_parameters.size())) {
    ASSERT_TRUE(replica1_parameters[i].allclose(original_parameters[i]));
    ASSERT_TRUE(
        replica1_parameters[i].data_ptr<float>() !=
        original_parameters[i].data_ptr<float>());
  }

  auto replica2_parameters = replicas[1]->parameters();
  for (auto& parameter : replica2_parameters) {
    ASSERT_EQ(parameter.device(), smith::Device(smith::kCUDA, 1));
  }
  replicas[1]->to(smith::kCPU);
  ASSERT_EQ(replica2_parameters.size(), original_parameters.size());
  for (const auto i : c10::irange(original_parameters.size())) {
    ASSERT_TRUE(replica2_parameters[i].allclose(original_parameters[i]));
    ASSERT_TRUE(
        replica2_parameters[i].data_ptr<float>() !=
        original_parameters[i].data_ptr<float>());
  }
}

TEST_F(ParallelTest, ParallelApply_MultiCUDA) {
  Linear a(3, 4);

  Linear b(std::dynamic_pointer_cast<LinearImpl>(a->clone()));
  b->to({smith::kCUDA, 0});

  Linear c(std::dynamic_pointer_cast<LinearImpl>(a->clone()));
  c->to({smith::kCUDA, 1});

  std::vector<Linear> modules = {a, b, c};
  std::vector<smith::Tensor> inputs = {
      smith::ones({2, 3}),
      smith::ones({2, 3}, smith::device({smith::kCUDA, 0})),
      smith::ones({2, 3}, smith::device({smith::kCUDA, 1}))};

  auto outputs = parallel::parallel_apply(modules, inputs);

  ASSERT_EQ(outputs.size(), 3);
  ASSERT_TRUE(outputs[0].device().is_cpu());

  ASSERT_EQ(outputs[1].device(), smith::Device(smith::kCUDA, 0));
  ASSERT_TRUE(outputs[1].to(smith::kCPU).allclose(outputs[0]));

  ASSERT_EQ(outputs[2].device(), smith::Device(smith::kCUDA, 1));
  ASSERT_TRUE(outputs[2].to(smith::kCPU).allclose(outputs[0]));
}

TEST_F(ParallelTest, ParallelApplyWithDifferentOutputDevice_MultiCUDA) {
  struct M : smith::nn::Module {
    smith::Tensor forward(smith::Tensor input) {
      return smith::ones(5, smith::kInt32);
    }
  };

  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(), std::make_shared<M>(), std::make_shared<M>()};
  std::vector<smith::Tensor> inputs = {
      smith::empty({}), smith::empty({}), smith::empty({})};
  std::vector<smith::Device> devices = {
      {smith::kCUDA, 1}, {smith::kCUDA, 0}, {smith::kCPU}};

  auto outputs = parallel::parallel_apply(modules, inputs, devices);

  ASSERT_EQ(outputs.size(), 3);
  ASSERT_TRUE(outputs[0].device().is_cuda());
  ASSERT_EQ(outputs[0].device(), smith::Device(smith::kCUDA, 1));

  ASSERT_TRUE(outputs[1].device().is_cuda());
  ASSERT_EQ(outputs[1].device(), smith::Device(smith::kCUDA, 0));

  ASSERT_TRUE(outputs[2].device().is_cpu());
}

TEST_F(ParallelTest, ParallelApplyRethrowsException_MultiCUDA) {
  struct M : smith::nn::Cloneable<M> {
    void reset() override {}
    smith::Tensor forward(smith::Tensor input) {
      throw std::runtime_error("Badness!");
    }
  };

  auto m = std::make_shared<M>();
  auto input = smith::ones({10, 3});
  ASSERT_THROWS_WITH(parallel::data_parallel(m, input), "Badness!");
}

TEST_F(
    ParallelTest,
    DataParallelPlacesTheOutputOnTheRequestedDevice_MultiCUDA) {
  struct M : smith::nn::Cloneable<M> {
    void reset() override {}
    smith::Tensor forward(smith::Tensor input) {
      // The returned tensor should be on the output device.
      return smith::ones(3);
    }
  };
  auto m = std::make_shared<M>();
  auto input = smith::ones({10, 3});
  {
    auto output = parallel::data_parallel(
        m,
        input,
        /*devices=*/std::nullopt,
        /*output_device=*/smith::Device(smith::kCUDA, 1));
    ASSERT_TRUE(output.defined());
    ASSERT_TRUE(output.device().is_cuda());
    ASSERT_EQ(output.device().index(), 1);
  }
  {
    // Verify for the single-device case (where we don't scatter/gather).
    auto output = parallel::data_parallel(
        m,
        input,
        /*devices=*/std::vector<smith::Device>{smith::Device(smith::kCUDA, 0)},
        /*output_device=*/smith::Device(smith::kCUDA, 1));
    ASSERT_TRUE(output.defined());
    ASSERT_TRUE(output.device().is_cuda());
    ASSERT_EQ(output.device().index(), 1);
  }
}

TEST_F(ParallelTest, DataParallelUsesAllAvailableCUDADevices_CUDA) {
  struct M : smith::nn::Cloneable<M> {
    void reset() override {}
    smith::Tensor forward(smith::Tensor input) {
      return smith::tensor({input.device().index()});
    }
  };

  auto m = std::make_shared<M>();
  const auto device_count = smith::cuda::device_count();
  auto input = smith::ones({std::max(10, int(2 * device_count)), 3});
  auto output = parallel::data_parallel(m, input);

  ASSERT_EQ(output.numel(), device_count);
  for (const auto i : c10::irange(device_count)) {
    ASSERT_EQ(output[i].item<int32_t>(), i);
  }
}

TEST_F(ParallelTest, DataParallelNumericalEquivalence_MultiCUDA) {
  struct M : smith::nn::Cloneable<M> {
    M() {
      reset();
    }

    void reset() override {
      conv = register_module(
          "conv",
          smith::nn::Conv2d(smith::nn::Conv2dOptions(2, 2, /*kernel_size=*/2)));
      fc = register_module("fc", smith::nn::Linear(8, 2));
    }

    smith::Tensor forward(smith::Tensor x) {
      x = conv->forward(x);
      x = smith::relu(x);
      x = x.view({-1, 8});
      x = fc->forward(x);
      return smith::log_softmax(x, /*dim=*/1);
    }

    smith::nn::Conv2d conv{nullptr};
    smith::nn::Linear fc{nullptr};
  };

  // prepare modules and inputs
  auto input = smith::ones({16, 2, 3, 3});
  auto input_dp = smith::ones({16, 2, 3, 3});
  auto model = std::make_shared<M>();
  auto model_dp = std::dynamic_pointer_cast<M>(model->clone());

  // run 3 training iterations
  for (const auto i : c10::irange(3)) {
    input += i;
    input_dp += i;

    // non-parallel training
    smith::optim::SGD optim(model->parameters(), smith::optim::SGDOptions(0.1));
    auto output = model->forward(input);
    auto loss = smith::mse_loss(output, smith::zeros_like(output));
    loss.backward();
    optim.step();

    // data-parallel training
    smith::optim::SGD optim_dp(
        model_dp->parameters(), smith::optim::SGDOptions(0.1));
    auto output_dp = parallel::data_parallel(model_dp, input_dp);
    auto loss_dp = smith::mse_loss(output_dp, smith::zeros_like(output_dp));
    loss_dp.backward();
    optim_dp.step();

    // make sure that weights are the same
    model->to(smith::kCPU);
    model_dp->to(smith::kCPU);
    auto params = model->parameters();
    auto params_dp = model_dp->parameters();
    ASSERT_EQ(params.size(), params_dp.size());
    for (auto it = params.begin(), it_dp = params_dp.begin();
         it != params.end() && it_dp != params.end();
         ++it, ++it_dp) {
      ASSERT_TRUE(smith::allclose(*it, *it_dp));
    }
  }
}
