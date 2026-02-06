#include <gtest/gtest.h>
#include <test/cpp/api/support.h>
#include <smith/script.h>

using namespace smith::autograd;
using namespace smith::test;

TEST(GradModeTest, TestRequiresGradFunctionalOp) {
  smith::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    smith::Tensor c = smith::ones({1, 2, 3}).set_requires_grad(requires_grad);

    smith::Tensor func_out = c * c;
    ASSERT_FALSE(func_out.requires_grad());
    ASSERT_TRUE(func_out.is_leaf());
  }
}

TEST(GradModeTest, TestRequiresGradInplaceOp) {
  smith::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    smith::Tensor c = smith::ones({1, 2, 3}).set_requires_grad(requires_grad);

    c.mul_(2);
    ASSERT_EQ(c.requires_grad(), requires_grad);
  }
}

TEST(GradModeTest, TestRequiresGradViewOp) {
  smith::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    smith::Tensor c = smith::ones({1, 2, 3}).set_requires_grad(requires_grad);

    smith::Tensor view_out = c.view({2, 3});
    ASSERT_EQ(view_out.requires_grad(), requires_grad);
    ASSERT_TRUE(view_out.is_leaf());
  }
}

TEST(GradModeTest, TestRequiresGradViewOpExiting) {
  for (bool requires_grad : {true, false}) {
    smith::Tensor s = smith::ones({1, 2, 3}).set_requires_grad(requires_grad);
    smith::Tensor a = s.clone();
    smith::Tensor view_out, tmp;

    {
      smith::AutoGradMode mode(false);
      view_out = a.view(
          {2, 3}); // go through kernels: VariableType, ADInplaceOrView, CPU
      assert_tensor_creation_meta(
          view_out, smith::autograd::CreationMeta::NO_GRAD_MODE);
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      ASSERT_TRUE(view_out.is_leaf());
    }

    tmp = view_out * view_out;
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
    if (requires_grad) {
      tmp.backward(smith::ones_like(tmp));
      // TODO: this behavior is a side effect of issue #11390.
      ASSERT_FALSE(view_out.grad().defined());
    }

    if (requires_grad) {
      ASSERT_THROWS_WITH(
          view_out.mul_(
              2), // go through kernels: VariableType, ADInplaceOrView, CPU
          "A view was created in no_grad mode and is being modified inplace");
    } else {
      view_out.mul_(2);
    }

    tmp = view_out.view({2, 3});
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
    assert_tensor_creation_meta(
        tmp, smith::autograd::CreationMeta::NO_GRAD_MODE);
  }
}
