#include <gtest/gtest.h>
#include <smith/smith.h>
#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace smith::nn;
using namespace smith::test;

struct ParameterDictTest : smith::test::SeedingFixture {};

TEST_F(ParameterDictTest, ConstructFromTensor) {
  ParameterDict dict;
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  ASSERT_TRUE(ta.requires_grad());
  ASSERT_FALSE(tb.requires_grad());
  dict->insert("A", ta);
  dict->insert("B", tb);
  dict->insert("C", tc);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(smith::all(smith::eq(dict["A"], ta)).item<bool>());
  ASSERT_TRUE(dict["A"].requires_grad());
  ASSERT_TRUE(smith::all(smith::eq(dict["B"], tb)).item<bool>());
  ASSERT_FALSE(dict["B"].requires_grad());
}

TEST_F(ParameterDictTest, ConstructFromOrderedDict) {
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  smith::OrderedDict<std::string, smith::Tensor> params = {
      {"A", ta}, {"B", tb}, {"C", tc}};
  auto dict = smith::nn::ParameterDict(params);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(smith::all(smith::eq(dict["A"], ta)).item<bool>());
  ASSERT_TRUE(dict["A"].requires_grad());
  ASSERT_TRUE(smith::all(smith::eq(dict["B"], tb)).item<bool>());
  ASSERT_FALSE(dict["B"].requires_grad());
}

TEST_F(ParameterDictTest, InsertAndContains) {
  ParameterDict dict;
  dict->insert("A", smith::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  ASSERT_TRUE(dict->contains("A"));
  ASSERT_FALSE(dict->contains("C"));
}

TEST_F(ParameterDictTest, InsertAndClear) {
  ParameterDict dict;
  dict->insert("A", smith::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  dict->clear();
  ASSERT_EQ(dict->size(), 0);
}

TEST_F(ParameterDictTest, InsertAndPop) {
  ParameterDict dict;
  dict->insert("A", smith::tensor({1.0}));
  ASSERT_EQ(dict->size(), 1);
  ASSERT_THROWS_WITH(dict->pop("B"), "Parameter 'B' is not defined");
  smith::Tensor p = dict->pop("A");
  ASSERT_EQ(dict->size(), 0);
  ASSERT_TRUE(smith::eq(p, smith::tensor({1.0})).item<bool>());
}

TEST_F(ParameterDictTest, SimpleUpdate) {
  ParameterDict dict;
  ParameterDict wrongDict;
  ParameterDict rightDict;
  dict->insert("A", smith::tensor({1.0}));
  dict->insert("B", smith::tensor({2.0}));
  dict->insert("C", smith::tensor({3.0}));
  wrongDict->insert("A", smith::tensor({5.0}));
  wrongDict->insert("D", smith::tensor({5.0}));
  ASSERT_THROWS_WITH(dict->update(*wrongDict), "Parameter 'D' is not defined");
  rightDict->insert("A", smith::tensor({5.0}));
  dict->update(*rightDict);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(smith::eq(dict["A"], smith::tensor({5.0})).item<bool>());
}

TEST_F(ParameterDictTest, Keys) {
  smith::OrderedDict<std::string, smith::Tensor> params = {
      {"a", smith::tensor({1.0})},
      {"b", smith::tensor({2.0})},
      {"c", smith::tensor({1.0, 2.0})}};
  auto dict = smith::nn::ParameterDict(params);
  std::vector<std::string> keys = dict->keys();
  std::vector<std::string> true_keys{"a", "b", "c"};
  ASSERT_EQ(keys, true_keys);
}

TEST_F(ParameterDictTest, Values) {
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  smith::OrderedDict<std::string, smith::Tensor> params = {
      {"a", ta}, {"b", tb}, {"c", tc}};
  auto dict = smith::nn::ParameterDict(params);
  std::vector<smith::Tensor> values = dict->values();
  std::vector<smith::Tensor> true_values{ta, tb, tc};
  for (auto i = 0U; i < values.size(); i += 1) {
    ASSERT_TRUE(smith::all(smith::eq(values[i], true_values[i])).item<bool>());
  }
}

TEST_F(ParameterDictTest, Get) {
  ParameterDict dict;
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  ASSERT_TRUE(ta.requires_grad());
  ASSERT_FALSE(tb.requires_grad());
  dict->insert("A", ta);
  dict->insert("B", tb);
  dict->insert("C", tc);
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(smith::all(smith::eq(dict->get("A"), ta)).item<bool>());
  ASSERT_TRUE(dict->get("A").requires_grad());
  ASSERT_TRUE(smith::all(smith::eq(dict->get("B"), tb)).item<bool>());
  ASSERT_FALSE(dict->get("B").requires_grad());
}

TEST_F(ParameterDictTest, PrettyPrintParameterDict) {
  smith::OrderedDict<std::string, smith::Tensor> params = {
      {"a", smith::tensor({1.0})},
      {"b", smith::tensor({2.0, 1.0})},
      {"c", smith::tensor({{3.0}, {2.1}})},
      {"d", smith::tensor({{3.0, 1.3}, {1.2, 2.1}})}};
  auto dict = smith::nn::ParameterDict(params);
  ASSERT_EQ(
      c10::str(dict),
      "smith::nn::ParameterDict(\n"
      "(a): Parameter containing: [Float of size [1]]\n"
      "(b): Parameter containing: [Float of size [2]]\n"
      "(c): Parameter containing: [Float of size [2, 1]]\n"
      "(d): Parameter containing: [Float of size [2, 2]]\n"
      ")");
}
