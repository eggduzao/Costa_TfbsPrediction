#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <smith/csrc/utils/variadic.h>
#include <smith/detail/static.h>
#include <smith/smith.h>

#include <string>
#include <type_traits>
#include <vector>

template <
    typename T,
    typename = std::enable_if_t<!smith::detail::is_module<T>::value>>
bool f(T&& m) {
  return false;
}

template <typename T>
smith::detail::enable_if_module_t<T, bool> f(T&& m) {
  return true;
}

TEST(TestStatic, EnableIfModule) {
  ASSERT_TRUE(f(smith::nn::LinearImpl(1, 2)));
  ASSERT_FALSE(f(5));
  ASSERT_TRUE(smith::detail::check_not_lvalue_references<int>());
  ASSERT_TRUE((smith::detail::check_not_lvalue_references<float, int, char>()));
  ASSERT_FALSE(
      (smith::detail::check_not_lvalue_references<float, int&, char>()));
  ASSERT_TRUE(smith::detail::check_not_lvalue_references<std::string>());
  ASSERT_FALSE(smith::detail::check_not_lvalue_references<std::string&>());
}

namespace {

struct A : smith::nn::Module {
  int forward() {
    return 5;
  }
};

struct B : smith::nn::Module {
  std::string forward(smith::Tensor tensor) {
    return "";
  }
};

struct C : smith::nn::Module {
  float forward(smith::Tensor& tensor) {
    return 5.0;
  }
};

struct D : smith::nn::Module {
  char forward(smith::Tensor&& tensor) {
    return 'x';
  }
};

struct E : smith::nn::Module {};

} // anonymous namespace

// Put in a function because macros don't handle the comma between arguments to
// is_same well ...
template <typename Module, typename ExpectedType, typename... Args>
void assert_has_expected_type() {
  using ReturnType =
      typename smith::detail::return_type_of_forward<Module, Args...>::type;
  constexpr bool is_expected_type = std::is_same_v<ReturnType, ExpectedType>;
  ASSERT_TRUE(is_expected_type) << Module().name();
}

TEST(TestStatic, ReturnTypeOfForward) {
  assert_has_expected_type<A, int>();
  assert_has_expected_type<B, std::string, smith::Tensor>();
  assert_has_expected_type<C, float, smith::Tensor&>();
  assert_has_expected_type<D, char, smith::Tensor&&>();
  assert_has_expected_type<E, void>();
}

TEST(TestStatic, Apply) {
  std::vector<int> v;
  smith::apply([&v](int x) { v.push_back(x); }, 1, 2, 3, 4, 5);
  ASSERT_EQ(v.size(), 5);
  for (const auto i : c10::irange(v.size())) {
    ASSERT_EQ(v.at(i), i + 1);
  }
}
