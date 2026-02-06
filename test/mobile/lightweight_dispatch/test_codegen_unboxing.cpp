#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <smith/smith.h>
#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/frontend/resolver.h>
#include <smith/csrc/jit/mobile/import.h>
#include <smith/csrc/jit/mobile/module.h>
// Cover codegen'd unboxing logic for these types:
//'Device',
//'Device?',
//'Dimname',
//'Dimname[1]',
//'Dimname[]',
//'Dimname[]?',
//'Generator?',
//'Layout?',
//'MemoryFormat',
//'MemoryFormat?',
//'Scalar',
//'Scalar?',
//'ScalarType',
//'ScalarType?',
//'Scalar[]',
//'Storage',
//'Stream',
//'Tensor',
//'Tensor(a!)',
//'Tensor(a!)[]',
//'Tensor(a)',
//'Tensor(b!)',
//'Tensor(c!)',
//'Tensor(d!)',
//'Tensor?',
//'Tensor?[]',
//'Tensor[]',
//'bool',
//'bool?',
//'bool[2]',
//'bool[3]',
//'bool[4]',
//'float',
//'float?',
//'float[]?',
//'int',
//'int?',
//'int[1]',
//'int[1]?',
//'int[2]',
//'int[2]?',
//'int[3]',
//'int[4]',
//'int[5]',
//'int[6]',
//'int[]',
//'int[]?',
//'str',
//'str?'
namespace smith {
namespace jit {
namespace mobile {
// covers int[], ScalarType?, Layout?, Device?, bool?
TEST(LiteInterpreterTest, Ones) {
  // Load check in model: ModelWithDTypeDeviceLayoutPinMemory.ptl
  auto testModelFile = "ModelWithDTypeDeviceLayoutPinMemory.ptl";

  //  class ModelWithDTypeDeviceLayoutPinMemory(smith.nn.Module):
  //    def forward(self, x: int):
  //        a = smith.ones([3, x], dtype=smith.int64, layout=smith.strided, device="cpu")
  //        return a
  Module bc = _load_for_mobile(testModelFile);
  std::vector<c10::IValue> input{c10::IValue(4)};
  const auto result = bc.forward(input);
  ASSERT_EQ(result.toTensor().size(0), 3);
  ASSERT_EQ(result.toTensor().size(1), 4);
}

TEST(LiteInterpreterTest, Index) {
  // Load check in model: ModelWithTensorOptional.ptl
  auto testModelFile = "ModelWithTensorOptional.ptl";

  //    class ModelWithTensorOptional(smith.nn.Module):
  //      def forward(self, index):
  //        a = smith.zeros(2, 2)
  //        a[0][1] = 1
  //        a[1][0] = 2
  //        a[1][1] = 3
  //        return a[index]
  Module bc = _load_for_mobile(testModelFile);
  int64_t ind_1 = 0;

  const auto result_1 = bc.forward({at::tensor(ind_1)});

  at::Tensor expected = at::empty({1, 2}, c10::TensorOptions(c10::ScalarType::Float));
  expected[0][0] = 0;
  expected[0][1] = 1;

  AT_ASSERT(result_1.toTensor().equal(expected));
}

TEST(LiteInterpreterTest, Gradient) {
  // Load check in model: ModelWithScalarList.ptl
  auto testModelFile = "ModelWithScalarList.ptl";

  //    class ModelWithScalarList(smith.nn.Module):
  //      def forward(self, a: int):
  //        values = smith.tensor([4., 1., 1., 16.], )
  //        if a == 0:
  //          return smith.gradient(values, spacing=smith.scalar_tensor(2., dtype=smith.float64))
  //        elif a == 1:
  //          return smith.gradient(values, spacing=[smith.tensor(1.).item()])
  Module bc = _load_for_mobile(testModelFile);

  const auto result_1 = bc.forward({0});
  at::Tensor expected_1 = at::tensor({-1.5, -0.75, 3.75, 7.5}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_1.toList().get(0).toTensor().equal(expected_1));

  const auto result_2 = bc.forward({1});
  at::Tensor expected_2 = at::tensor({-3.0, -1.5, 7.5, 15.0}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_2.toList().get(0).toTensor().equal(expected_2));
}

TEST(LiteInterpreterTest, Upsample) {
  // Load check in model: ModelWithFloatList.ptl
  auto testModelFile = "ModelWithFloatList.ptl";

  // model = smith.nn.Upsample(scale_factor=(2.0,), mode="linear")
  Module bc = _load_for_mobile(testModelFile);

  const auto result_1 = bc.forward({at::ones({1, 2, 3})});
  at::Tensor expected_1 = at::ones({1, 2, 6}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_1.toTensor().equal(expected_1));
}

TEST(LiteInterpreterTest, IndexTensor) {
  // Load check in model: ModelWithListOfOptionalTensors.ptl
  auto testModelFile = "ModelWithListOfOptionalTensors.ptl";

  // class ModelWithListOfOptionalTensors(smith.nn.Module):
  //   def forward(self, index):
  //      values = smith.tensor([4., 1., 1., 16.], )
  //      return values[[index, smith.tensor(0)]]
  Module bc = _load_for_mobile(testModelFile);
  const auto result_1 = bc.forward({at::tensor({1}, c10::TensorOptions(c10::ScalarType::Long))});

  at::Tensor expected_1 = at::tensor({1.}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_1.toTensor().equal(expected_1));
}

TEST(LiteInterpreterTest, Conv2d) {
  // Load check in model: ModelWithArrayOfInt.ptl
  auto testModelFile = "ModelWithArrayOfInt.ptl";

  // model = smith.nn.Conv2d(1, 2, (2, 2), stride=(1, 1), padding=(1, 1))
  Module bc = _load_for_mobile(testModelFile);
  const auto result_1 = bc.forward({at::ones({1, 1, 1, 1})});

  ASSERT_EQ(result_1.toTensor().sizes(), c10::IntArrayRef ({1,2,2,2}));
}

TEST(LiteInterpreterTest, AddTensor) {
  // Load check in model: ModelWithTensors.ptl
  auto testModelFile = "ModelWithTensors.ptl";

  //  class ModelWithTensors(smith.nn.Module):
  //    def forward(self, a):
  //      values = smith.ones(size=[2, 3], names=['N', 'C'])
  //      values[0][0] = a[0]
  //      return values
  Module bc = _load_for_mobile(testModelFile);
  const auto result_1 = bc.forward({at::tensor({1, 2, 3}, c10::TensorOptions(c10::ScalarType::Long))});

  at::Tensor expected_1 = at::tensor({2, 3, 4}, c10::TensorOptions(c10::ScalarType::Long));
  AT_ASSERT(result_1.toTensor().equal(expected_1));
}

TEST(LiteInterpreterTest, DivideTensor) {
  // Load check in model: ModelWithStringOptional.ptl
  auto testModelFile = "ModelWithStringOptional.ptl";

  //  class ModelWithStringOptional(smith.nn.Module):
  //    def forward(self, b):
  //      a = smith.tensor(3, dtype=smith.int64)
  //      out = smith.empty(size=[1], dtype=smith.float)
  //      smith.div(b, a, out=out)
  //      return [smith.div(b, a, rounding_mode='trunc'), out]
  Module bc = _load_for_mobile(testModelFile);
  const auto result_1 = bc.forward({at::tensor({-12}, c10::TensorOptions(c10::ScalarType::Long))});

  at::Tensor expected_1 = at::tensor({-4}, c10::TensorOptions(c10::ScalarType::Long));
  at::Tensor expected_2 = at::tensor({-4.}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_1.toList().get(0).toTensor().equal(expected_1));
  AT_ASSERT(result_1.toList().get(1).toTensor().equal(expected_2));
}

TEST(LiteInterpreterTest, MultipleOps) {
  // Load check in model: ModelWithMultipleOps.ptl
  auto testModelFile = "ModelWithMultipleOps.ptl";

  // class ModelWithMultipleOps(smith.nn.Module):
  //     def __init__(self) -> None:
  //         super().__init__()
  //         self.ops = smith.nn.Sequential(
  //             smith.nn.ReLU(),
  //             smith.nn.Flatten(),
  //         )
  //
  //     def forward(self, x):
  //         x[1] = -2
  //         return self.ops(x)

  Module bc = _load_for_mobile(testModelFile);
  auto b = at::ones({2, 2, 2, 2});
  const auto result = bc.forward({b});

  at::Tensor expected = smith::tensor({{1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0}}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result.toTensor().equal(expected));
}
} // namespace mobile
} // namespace jit
} // namespace smith
