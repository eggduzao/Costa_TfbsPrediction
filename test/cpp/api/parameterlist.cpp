#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <smith/smith.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace smith::nn;
using namespace smith::test;

struct ParameterListTest : smith::test::SeedingFixture {};

TEST_F(ParameterListTest, ConstructsFromSharedPointer) {
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  ASSERT_TRUE(ta.requires_grad());
  ASSERT_FALSE(tb.requires_grad());
  ParameterList list(ta, tb, tc);
  ASSERT_EQ(list->size(), 3);
}

TEST_F(ParameterListTest, isEmpty) {
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  ParameterList list;
  ASSERT_TRUE(list->is_empty());
  list->append(ta);
  ASSERT_FALSE(list->is_empty());
  ASSERT_EQ(list->size(), 1);
}

TEST_F(ParameterListTest, PushBackAddsAnElement) {
  ParameterList list;
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  smith::Tensor td = smith::randn({1, 2, 3});
  ASSERT_EQ(list->size(), 0);
  ASSERT_TRUE(list->is_empty());
  list->append(ta);
  ASSERT_EQ(list->size(), 1);
  list->append(tb);
  ASSERT_EQ(list->size(), 2);
  list->append(tc);
  ASSERT_EQ(list->size(), 3);
  list->append(td);
  ASSERT_EQ(list->size(), 4);
}
TEST_F(ParameterListTest, ForEachLoop) {
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  smith::Tensor td = smith::randn({1, 2, 3});
  ParameterList list(ta, tb, tc, td);
  std::vector<smith::Tensor> params = {ta, tb, tc, td};
  ASSERT_EQ(list->size(), 4);
  int idx = 0;
  for (const auto& pair : *list) {
    ASSERT_TRUE(
        smith::all(smith::eq(pair.value(), params[idx++])).item<bool>());
  }
}

TEST_F(ParameterListTest, AccessWithAt) {
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  smith::Tensor td = smith::randn({1, 2, 3});
  std::vector<smith::Tensor> params = {ta, tb, tc, td};

  ParameterList list;
  for (auto& param : params) {
    list->append(param);
  }
  ASSERT_EQ(list->size(), 4);

  // returns the correct module for a given index
  for (const auto i : c10::irange(params.size())) {
    ASSERT_TRUE(smith::all(smith::eq(list->at(i), params[i])).item<bool>());
  }

  for (const auto i : c10::irange(params.size())) {
    ASSERT_TRUE(smith::all(smith::eq(list[i], params[i])).item<bool>());
  }

  // throws for a bad index
  ASSERT_THROWS_WITH(list->at(params.size() + 100), "Index out of range");
  ASSERT_THROWS_WITH(list->at(params.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(list[params.size() + 1], "Index out of range");
}

TEST_F(ParameterListTest, ExtendPushesParametersFromOtherParameterList) {
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  smith::Tensor td = smith::randn({1, 2, 3});
  smith::Tensor te = smith::randn({1, 2});
  smith::Tensor tf = smith::randn({1, 2, 3});
  ParameterList a(ta, tb);
  ParameterList b(tc, td);
  a->extend(*b);

  ASSERT_EQ(a->size(), 4);
  ASSERT_TRUE(smith::all(smith::eq(a[0], ta)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(a[1], tb)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(a[2], tc)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(a[3], td)).item<bool>());

  ASSERT_EQ(b->size(), 2);
  ASSERT_TRUE(smith::all(smith::eq(b[0], tc)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(b[1], td)).item<bool>());

  std::vector<smith::Tensor> c = {te, tf};
  b->extend(c);

  ASSERT_EQ(b->size(), 4);
  ASSERT_TRUE(smith::all(smith::eq(b[0], tc)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(b[1], td)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(b[2], te)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(b[3], tf)).item<bool>());
}

TEST_F(ParameterListTest, PrettyPrintParameterList) {
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  ParameterList list(ta, tb, tc);
  ASSERT_EQ(
      c10::str(list),
      "smith::nn::ParameterList(\n"
      "(0): Parameter containing: [Float of size [1, 2]]\n"
      "(1): Parameter containing: [Float of size [1, 2]]\n"
      "(2): Parameter containing: [Float of size [1, 2]]\n"
      ")");
}

TEST_F(ParameterListTest, IncrementAdd) {
  smith::Tensor ta = smith::randn({1, 2}, smith::requires_grad(true));
  smith::Tensor tb = smith::randn({1, 2}, smith::requires_grad(false));
  smith::Tensor tc = smith::randn({1, 2});
  smith::Tensor td = smith::randn({1, 2, 3});
  smith::Tensor te = smith::randn({1, 2});
  smith::Tensor tf = smith::randn({1, 2, 3});
  ParameterList listA(ta, tb, tc);
  ParameterList listB(td, te, tf);
  std::vector<smith::Tensor> tensors{ta, tb, tc, td, te, tf};
  int idx = 0;
  *listA += *listB;
  ASSERT_TRUE(smith::all(smith::eq(listA[0], ta)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(listA[1], tb)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(listA[2], tc)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(listA[3], td)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(listA[4], te)).item<bool>());
  ASSERT_TRUE(smith::all(smith::eq(listA[5], tf)).item<bool>());
  for (const auto& P : listA->named_parameters(false))
    ASSERT_TRUE(smith::all(smith::eq(P.value(), tensors[idx++])).item<bool>());

  ASSERT_EQ(idx, 6);
}
