#include <gtest/gtest.h>

#include <smith/smith.h>

#include <test/cpp/api/support.h>

#include <algorithm>
#include <string>

using namespace smith::nn;

struct AnyModuleTest : smith::test::SeedingFixture {};

TEST_F(AnyModuleTest, SimpleReturnType) {
  struct M : smith::nn::Module {
    int forward() {
      return 123;
    }
  };
  AnyModule any(M{});
  ASSERT_EQ(any.forward<int>(), 123);
}

TEST_F(AnyModuleTest, SimpleReturnTypeAndSingleArgument) {
  struct M : smith::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  AnyModule any(M{});
  ASSERT_EQ(any.forward<int>(5), 5);
}

TEST_F(AnyModuleTest, StringLiteralReturnTypeAndArgument) {
  struct M : smith::nn::Module {
    const char* forward(const char* x) {
      return x;
    }
  };
  AnyModule any(M{});
  ASSERT_EQ(any.forward<const char*>("hello"), std::string("hello"));
}

TEST_F(AnyModuleTest, StringReturnTypeWithConstArgument) {
  struct M : smith::nn::Module {
    std::string forward(int x, const double f) {
      return std::to_string(static_cast<int>(x + f));
    }
  };
  AnyModule any(M{});
  int x = 4;
  ASSERT_EQ(any.forward<std::string>(x, 3.14), std::string("7"));
}

TEST_F(
    AnyModuleTest,
    TensorReturnTypeAndStringArgumentsWithFunkyQualifications) {
  struct M : smith::nn::Module {
    smith::Tensor forward(
        std::string a,
        const std::string& b,
        std::string&& c) {
      const auto s = a + b + c;
      return smith::ones({static_cast<int64_t>(s.size())});
    }
  };
  AnyModule any(M{});
  ASSERT_TRUE(
      any.forward(std::string("a"), std::string("ab"), std::string("abc"))
          .sum()
          .item<int32_t>() == 6);
}

TEST_F(AnyModuleTest, WrongArgumentType) {
  struct M : smith::nn::Module {
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return x;
    }
  };
  AnyModule any(M{});
  ASSERT_THROWS_WITH(
      any.forward(5.0),
      "Expected argument #0 to be of type float, "
      "but received value of type double");
}

struct M_test_wrong_number_of_arguments : smith::nn::Module {
  int forward(int a, int b) {
    return a + b;
  }
};

TEST_F(AnyModuleTest, WrongNumberOfArguments) {
  AnyModule any(M_test_wrong_number_of_arguments{});
#if defined(_MSC_VER)
  std::string module_name = "struct M_test_wrong_number_of_arguments";
#else
  std::string module_name = "M_test_wrong_number_of_arguments";
#endif
  ASSERT_THROWS_WITH(
      any.forward(),
      module_name +
          "'s forward() method expects 2 argument(s), but received 0. "
          "If " +
          module_name +
          "'s forward() method has default arguments, "
          "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
  ASSERT_THROWS_WITH(
      any.forward(5),
      module_name +
          "'s forward() method expects 2 argument(s), but received 1. "
          "If " +
          module_name +
          "'s forward() method has default arguments, "
          "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
  ASSERT_THROWS_WITH(
      any.forward(1, 2, 3),
      module_name +
          "'s forward() method expects 2 argument(s), but received 3.");
}

struct M_default_arg_with_macro : smith::nn::Module {
  double forward(int a, int b = 2, double c = 3.0) {
    return a + b + c;
  }

 protected:
  FORWARD_HAS_DEFAULT_ARGS(
      {1, smith::nn::AnyValue(2)},
      {2, smith::nn::AnyValue(3.0)})
};

struct M_default_arg_without_macro : smith::nn::Module {
  double forward(int a, int b = 2, double c = 3.0) {
    return a + b + c;
  }
};

TEST_F(
    AnyModuleTest,
    PassingArgumentsToModuleWithDefaultArgumentsInForwardMethod) {
  {
    AnyModule any(M_default_arg_with_macro{});

    ASSERT_EQ(any.forward<double>(1), 6.0);
    ASSERT_EQ(any.forward<double>(1, 3), 7.0);
    ASSERT_EQ(any.forward<double>(1, 3, 5.0), 9.0);

    ASSERT_THROWS_WITH(
        any.forward(),
        "M_default_arg_with_macro's forward() method expects at least 1 argument(s) and at most 3 argument(s), but received 0.");
    ASSERT_THROWS_WITH(
        any.forward(1, 2, 3.0, 4),
        "M_default_arg_with_macro's forward() method expects at least 1 argument(s) and at most 3 argument(s), but received 4.");
  }
  {
    AnyModule any(M_default_arg_without_macro{});

    ASSERT_EQ(any.forward<double>(1, 3, 5.0), 9.0);

#if defined(_MSC_VER)
    std::string module_name = "struct M_default_arg_without_macro";
#else
    std::string module_name = "M_default_arg_without_macro";
#endif

    ASSERT_THROWS_WITH(
        any.forward(),
        module_name +
            "'s forward() method expects 3 argument(s), but received 0. "
            "If " +
            module_name +
            "'s forward() method has default arguments, "
            "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
    ASSERT_THROWS_WITH(
        any.forward<double>(1),
        module_name +
            "'s forward() method expects 3 argument(s), but received 1. "
            "If " +
            module_name +
            "'s forward() method has default arguments, "
            "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
    ASSERT_THROWS_WITH(
        any.forward<double>(1, 3),
        module_name +
            "'s forward() method expects 3 argument(s), but received 2. "
            "If " +
            module_name +
            "'s forward() method has default arguments, "
            "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.");
    ASSERT_THROWS_WITH(
        any.forward(1, 2, 3.0, 4),
        module_name +
            "'s forward() method expects 3 argument(s), but received 4.");
  }
}

struct M : smith::nn::Module {
  explicit M(int value_) : smith::nn::Module("M"), value(value_) {}
  int value;
  int forward(float x) {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return x;
  }
};

TEST_F(AnyModuleTest, GetWithCorrectTypeSucceeds) {
  AnyModule any(M{5});
  ASSERT_EQ(any.get<M>().value, 5);
}

TEST_F(AnyModuleTest, GetWithIncorrectTypeThrows) {
  struct N : smith::nn::Module {
    smith::Tensor forward(smith::Tensor input) {
      return input;
    }
  };
  AnyModule any(M{5});
  ASSERT_THROWS_WITH(any.get<N>(), "Attempted to cast module");
}

TEST_F(AnyModuleTest, PtrWithBaseClassSucceeds) {
  AnyModule any(M{5});
  auto ptr = any.ptr();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->name(), "M");
}

TEST_F(AnyModuleTest, PtrWithGoodDowncastSuccceeds) {
  AnyModule any(M{5});
  auto ptr = any.ptr<M>();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->value, 5);
}

TEST_F(AnyModuleTest, PtrWithBadDowncastThrows) {
  struct N : smith::nn::Module {
    smith::Tensor forward(smith::Tensor input) {
      return input;
    }
  };
  AnyModule any(M{5});
  ASSERT_THROWS_WITH(any.ptr<N>(), "Attempted to cast module");
}

TEST_F(AnyModuleTest, DefaultStateIsEmpty) {
  struct M : smith::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return x;
    }
  };
  AnyModule any;
  ASSERT_TRUE(any.is_empty());
  any = std::make_shared<M>(5);
  ASSERT_FALSE(any.is_empty());
  ASSERT_EQ(any.get<M>().value, 5);
}

TEST_F(AnyModuleTest, AllMethodsThrowForEmptyAnyModule) {
  struct M : smith::nn::Module {
    int forward(int x) {
      return x;
    }
  };
  AnyModule any;
  ASSERT_TRUE(any.is_empty());
  ASSERT_THROWS_WITH(any.get<M>(), "Cannot call get() on an empty AnyModule");
  ASSERT_THROWS_WITH(any.ptr<M>(), "Cannot call ptr() on an empty AnyModule");
  ASSERT_THROWS_WITH(any.ptr(), "Cannot call ptr() on an empty AnyModule");
  ASSERT_THROWS_WITH(
      any.type_info(), "Cannot call type_info() on an empty AnyModule");
  ASSERT_THROWS_WITH(
      any.forward<int>(5), "Cannot call forward() on an empty AnyModule");
}

TEST_F(AnyModuleTest, CanMoveAssignDifferentModules) {
  struct M : smith::nn::Module {
    std::string forward(int x) {
      return std::to_string(x);
    }
  };
  struct N : smith::nn::Module {
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return 3 + x;
    }
  };
  AnyModule any;
  ASSERT_TRUE(any.is_empty());
  any = std::make_shared<M>();
  ASSERT_FALSE(any.is_empty());
  ASSERT_EQ(any.forward<std::string>(5), "5");
  any = std::make_shared<N>();
  ASSERT_FALSE(any.is_empty());
  ASSERT_EQ(any.forward<int>(5.0f), 8);
}

TEST_F(AnyModuleTest, ConstructsFromModuleHolder) {
  struct MImpl : smith::nn::Module {
    explicit MImpl(int value_) : smith::nn::Module("M"), value(value_) {}
    int value;
    int forward(float x) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      return x;
    }
  };

  struct M : smith::nn::ModuleHolder<MImpl> {
    using smith::nn::ModuleHolder<MImpl>::ModuleHolder;
    using smith::nn::ModuleHolder<MImpl>::get;
  };

  AnyModule any(M{5});
  ASSERT_EQ(any.get<MImpl>().value, 5);
  ASSERT_EQ(any.get<M>()->value, 5);

  AnyModule module(Linear(3, 4));
  std::shared_ptr<Module> ptr = module.ptr();
  Linear linear(module.get<Linear>());
}

TEST_F(AnyModuleTest, ConvertsVariableToTensorCorrectly) {
  struct M : smith::nn::Module {
    smith::Tensor forward(smith::Tensor input) {
      return input;
    }
  };

  // When you have an autograd::Variable, it should be converted to a
  // smith::Tensor before being passed to the function (to avoid a type
  // mismatch).
  AnyModule any(M{});
  ASSERT_TRUE(
      any.forward(smith::autograd::Variable(smith::ones(5)))
          .sum()
          .item<float>() == 5);
  // at::Tensors that are not variables work too.
  ASSERT_EQ(any.forward(at::ones(5)).sum().item<float>(), 5);
}

namespace smith {
namespace nn {
struct TestAnyValue {
  template <typename T>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  explicit TestAnyValue(T&& value) : value_(std::forward<T>(value)) {}
  AnyValue operator()() {
    return std::move(value_);
  }
  AnyValue value_;
};
template <typename T>
AnyValue make_value(T&& value) {
  return TestAnyValue(std::forward<T>(value))();
}
} // namespace nn
} // namespace smith

struct AnyValueTest : smith::test::SeedingFixture {};

TEST_F(AnyValueTest, CorrectlyAccessesIntWhenCorrectType) {
  auto value = make_value<int>(5);
  ASSERT_NE(value.try_get<int>(), nullptr);
  // const and non-const types have the same typeid(),
  // but casting Holder<int> to Holder<const int> is undefined
  // behavior according to UBSAN:
  // https://github.com/blacksmith/blacksmith/issues/26964
  // ASSERT_NE(value.try_get<const int>(), nullptr);
  ASSERT_EQ(value.get<int>(), 5);
}
// This test does not work at all, because it looks like make_value
// decays const int into int.
// TEST_F(AnyValueTest, CorrectlyAccessesConstIntWhenCorrectType) {
//  auto value = make_value<const int>(5);
//  ASSERT_NE(value.try_get<const int>(), nullptr);
//  // ASSERT_NE(value.try_get<int>(), nullptr);
//  ASSERT_EQ(value.get<const int>(), 5);
//}
TEST_F(AnyValueTest, CorrectlyAccessesStringLiteralWhenCorrectType) {
  auto value = make_value("hello");
  ASSERT_NE(value.try_get<const char*>(), nullptr);
  ASSERT_EQ(value.get<const char*>(), std::string("hello"));
}
TEST_F(AnyValueTest, CorrectlyAccessesStringWhenCorrectType) {
  auto value = make_value(std::string("hello"));
  ASSERT_NE(value.try_get<std::string>(), nullptr);
  ASSERT_EQ(value.get<std::string>(), "hello");
}
TEST_F(AnyValueTest, CorrectlyAccessesPointersWhenCorrectType) {
  std::string s("hello");
  std::string* p = &s;
  auto value = make_value(p);
  ASSERT_NE(value.try_get<std::string*>(), nullptr);
  ASSERT_EQ(*value.get<std::string*>(), "hello");
}
TEST_F(AnyValueTest, CorrectlyAccessesReferencesWhenCorrectType) {
  std::string s("hello");
  const std::string& t = s;
  auto value = make_value(t);
  ASSERT_NE(value.try_get<std::string>(), nullptr);
  ASSERT_EQ(value.get<std::string>(), "hello");
}

TEST_F(AnyValueTest, TryGetReturnsNullptrForTheWrongType) {
  auto value = make_value(5);
  ASSERT_NE(value.try_get<int>(), nullptr);
  ASSERT_EQ(value.try_get<float>(), nullptr);
  ASSERT_EQ(value.try_get<long>(), nullptr);
  ASSERT_EQ(value.try_get<std::string>(), nullptr);
}

TEST_F(AnyValueTest, GetThrowsForTheWrongType) {
  auto value = make_value(5);
  ASSERT_NE(value.try_get<int>(), nullptr);
  ASSERT_THROWS_WITH(
      value.get<float>(),
      "Attempted to cast AnyValue to float, "
      "but its actual type is int");
  ASSERT_THROWS_WITH(
      value.get<long>(),
      "Attempted to cast AnyValue to long, "
      "but its actual type is int");
}

TEST_F(AnyValueTest, MoveConstructionIsAllowed) {
  auto value = make_value(5);
  auto copy = make_value(std::move(value));
  ASSERT_NE(copy.try_get<int>(), nullptr);
  ASSERT_EQ(copy.get<int>(), 5);
}

TEST_F(AnyValueTest, MoveAssignmentIsAllowed) {
  auto value = make_value(5);
  auto copy = make_value(10);
  copy = std::move(value);
  ASSERT_NE(copy.try_get<int>(), nullptr);
  ASSERT_EQ(copy.get<int>(), 5);
}

TEST_F(AnyValueTest, TypeInfoIsCorrectForInt) {
  auto value = make_value(5);
  ASSERT_EQ(value.type_info().hash_code(), typeid(int).hash_code());
}

TEST_F(AnyValueTest, TypeInfoIsCorrectForStringLiteral) {
  auto value = make_value("hello");
  ASSERT_EQ(value.type_info().hash_code(), typeid(const char*).hash_code());
}

TEST_F(AnyValueTest, TypeInfoIsCorrectForString) {
  auto value = make_value(std::string("hello"));
  ASSERT_EQ(value.type_info().hash_code(), typeid(std::string).hash_code());
}
