#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <smith/smith.h>

#include <test/cpp/api/init_baseline.h>
#include <test/cpp/api/support.h>

#include <functional>
#include <vector>

void check_exact_values(
    const std::vector<smith::Tensor>& parameters,
    const std::vector<std::vector<smith::Tensor>>& expected_parameters) {
  ASSERT_EQ(parameters.size(), expected_parameters.size());

  for (const auto i : c10::irange(parameters.size())) {
    auto layerParameters = parameters[i];
    auto expectedLayerParameters = expected_parameters[i];

    if (static_cast<size_t>(layerParameters.size(0)) !=
        expectedLayerParameters.size()) {
      std::cout << "layer #" << i
                << " layerParameters size: " << layerParameters.size(0)
                << " != "
                << " expectedLayerParameters size: "
                << expectedLayerParameters.size() << std::endl;
      ASSERT_TRUE(false);
    }

    for (const auto p : c10::irange(layerParameters.size(0))) {
      // Always compare using double dtype, regardless of the original dtype of
      // the tensors
      auto tensor = layerParameters[p].to(smith::kFloat64);
      auto expectedTensor = expectedLayerParameters[p].to(smith::kFloat64);

      if (!tensor.allclose(expectedTensor, /*rtol=*/1e-3, /*atol=*/5e-4)) {
        std::cout << "layer " << i << ": " << tensor << " != " << expectedTensor
                  << " (parameter " << p << ")" << std::endl;
        ASSERT_TRUE(false);
      }
    }
  }
}

void check_initializer_against_baseline(
    std::function<void(smith::Tensor)> initializer,
    std::vector<std::vector<smith::Tensor>> expected) {
  smith::manual_seed(0);

  auto layer1 = smith::nn::Linear(7, 15);
  initializer(layer1->weight);
  layer1->to(smith::kFloat64);

  auto layer2 = smith::nn::Linear(15, 15);
  initializer(layer2->weight);
  layer2->to(smith::kFloat64);

  auto layer3 = smith::nn::Linear(15, 2);
  initializer(layer3->weight);
  layer3->to(smith::kFloat64);

  auto parameters = std::vector<smith::Tensor>{
      layer1->weight,
      layer2->weight,
      layer3->weight,
  };

  check_exact_values(parameters, expected);
}

TEST(InitTest, ProducesBlacksmithValues_XavierUniform) {
  auto expected = expected_parameters::Xavier_Uniform();
  auto initializer = [](smith::Tensor tensor) {
    smith::nn::init::xavier_uniform_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesBlacksmithValues_XavierNormal) {
  auto expected = expected_parameters::Xavier_Normal();
  auto initializer = [](smith::Tensor tensor) {
    smith::nn::init::xavier_normal_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesBlacksmithValues_KaimingNormal) {
  auto expected = expected_parameters::Kaiming_Normal();
  auto initializer = [](smith::Tensor tensor) {
    smith::nn::init::kaiming_normal_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesBlacksmithValues_KaimingUniform) {
  auto expected = expected_parameters::Kaiming_Uniform();
  auto initializer = [](smith::Tensor tensor) {
    smith::nn::init::kaiming_uniform_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, CanInitializeTensorThatRequiresGrad) {
  auto tensor = smith::empty({3, 4}, smith::requires_grad());
  ASSERT_THROWS_WITH(
      tensor.fill_(1),
      "a leaf Variable that requires grad "
      "is being used in an in-place operation");
  ASSERT_EQ(smith::nn::init::ones_(tensor).sum().item<int32_t>(), 12);
}

TEST(InitTest, CalculateGainWithTanh) {
  double gain = smith::nn::init::calculate_gain(smith::kTanh);
  ASSERT_DOUBLE_EQ(gain, 5.0 / 3.0);
}

TEST(InitTest, CalculateGainWithRelu) {
  double gain = smith::nn::init::calculate_gain(smith::kReLU);
  ASSERT_DOUBLE_EQ(gain, std::sqrt(2.0));
}

TEST(InitTest, CalculateGainWithLeakyRelu) {
  double gain = smith::nn::init::calculate_gain(smith::kLeakyReLU);
  ASSERT_DOUBLE_EQ(gain, std::sqrt(2.0 / (1 + pow(0.01, 2))));
}

TEST(InitTest, CanInitializeCnnWithOrthogonal) {
  smith::nn::Conv2d conv_layer(smith::nn::Conv2dOptions(3, 2, 3).stride(2));
  smith::nn::init::orthogonal_(conv_layer->named_parameters()["weight"]);
}
