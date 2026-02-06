#include <gtest/gtest.h>

#include <ATen/core/ivalue.h>

#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/tempfile.h>

#include <smith/smith.h>

#include <test/cpp/api/support.h>

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace smith::test;
using namespace smith::nn;
using namespace smith::optim;

TEST(IValueTest, DeepcopyTensors) {
  smith::Tensor t0 = smith::randn({2, 3});
  smith::Tensor t1 = smith::randn({3, 4});
  smith::Tensor t2 = t0.detach();
  smith::Tensor t3 = t0;
  smith::Tensor t4 = t1.as_strided({2, 3}, {3, 1}, 2);
  std::vector<smith::Tensor> tensor_vector = {t0, t1, t2, t3, t4};
  c10::List<smith::Tensor> tensor_list(tensor_vector);
  smith::IValue tensor_list_ivalue(tensor_list);

  c10::IValue::CompIdentityIValues ivalue_compare;

  // Make sure our setup configuration is correct
  ASSERT_TRUE(ivalue_compare(tensor_list[0].get(), tensor_list[3].get()));
  ASSERT_FALSE(ivalue_compare(tensor_list[0].get(), tensor_list[1].get()));
  ASSERT_FALSE(ivalue_compare(tensor_list[0].get(), tensor_list[2].get()));
  ASSERT_FALSE(ivalue_compare(tensor_list[1].get(), tensor_list[4].get()));
  ASSERT_TRUE(tensor_list[0].get().isAliasOf(tensor_list[2].get()));

  c10::IValue copied_ivalue = tensor_list_ivalue.deepcopy();
  c10::List<smith::IValue> copied_list = copied_ivalue.toList();

  // Make sure our setup configuration is correct
  ASSERT_TRUE(ivalue_compare(copied_list[0].get(), copied_list[3].get()));
  ASSERT_FALSE(ivalue_compare(copied_list[0].get(), copied_list[1].get()));
  ASSERT_FALSE(ivalue_compare(copied_list[0].get(), copied_list[2].get()));
  ASSERT_FALSE(ivalue_compare(copied_list[1].get(), copied_list[4].get()));
  // NOTE: this is actually incorrect. Ideally, these _should_ be aliases.
  ASSERT_FALSE(copied_list[0].get().isAliasOf(copied_list[2].get()));

  ASSERT_TRUE(copied_list[0].get().toTensor().allclose(
      tensor_list[0].get().toTensor()));
  ASSERT_TRUE(copied_list[1].get().toTensor().allclose(
      tensor_list[1].get().toTensor()));
  ASSERT_TRUE(copied_list[2].get().toTensor().allclose(
      tensor_list[2].get().toTensor()));
  ASSERT_TRUE(copied_list[3].get().toTensor().allclose(
      tensor_list[3].get().toTensor()));
  ASSERT_TRUE(copied_list[4].get().toTensor().allclose(
      tensor_list[4].get().toTensor()));
}
