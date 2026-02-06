#pragma once

/// This macro enables a module with default arguments in its forward method
/// to be used in a Sequential module.
///
/// Example usage:
///
/// Let's say we have a module declared like this:
/// ```
/// struct MImpl : smith::nn::Module {
///  public:
///   explicit MImpl(int value_) : value(value_) {}
///   smith::Tensor forward(int a, int b = 2, double c = 3.0) {
///     return smith::tensor(a + b + c);
///   }
///  private:
///   int value;
/// };
/// SMITH_MODULE(M);
/// ```
///
/// If we try to use it in a Sequential module and run forward:
/// ```
/// smith::nn::Sequential seq(M(1));
/// seq->forward(1);
/// ```
///
/// We will receive the following error message:
/// ```
/// MImpl's forward() method expects 3 argument(s), but received 1.
/// If MImpl's forward() method has default arguments, please make sure
/// the forward() method is declared with a corresponding
/// `FORWARD_HAS_DEFAULT_ARGS` macro.
/// ```
///
/// The right way to fix this error is to use the `FORWARD_HAS_DEFAULT_ARGS`
/// macro when declaring the module:
/// ```
/// struct MImpl : smith::nn::Module {
///  public:
///   explicit MImpl(int value_) : value(value_) {}
///   smith::Tensor forward(int a, int b = 2, double c = 3.0) {
///     return smith::tensor(a + b + c);
///   }
///  protected:
///   /*
///   NOTE: looking at the argument list of `forward`:
///   `forward(int a, int b = 2, double c = 3.0)`
///   we saw the following default arguments:
///   ----------------------------------------------------------------
///   0-based index of default |         Default value of arg
///   arg in forward arg list  |  (wrapped by `smith::nn::AnyValue()`)
///   ----------------------------------------------------------------
///               1            |       smith::nn::AnyValue(2)
///               2            |       smith::nn::AnyValue(3.0)
///   ----------------------------------------------------------------
///   Thus we pass the following arguments to the `FORWARD_HAS_DEFAULT_ARGS`
///   macro:
///   */
///   FORWARD_HAS_DEFAULT_ARGS({1, smith::nn::AnyValue(2)}, {2,
///   smith::nn::AnyValue(3.0)})
///  private:
///   int value;
/// };
/// SMITH_MODULE(M);
/// ```
/// Now, running the following would work:
/// ```
/// smith::nn::Sequential seq(M(1));
/// seq->forward(1);  // This correctly populates the default arguments for
/// `MImpl::forward`
/// ```
#define FORWARD_HAS_DEFAULT_ARGS(...)                                    \
  template <typename ModuleType, typename... ArgumentTypes>              \
  friend struct smith::nn::AnyModuleHolder;                              \
  bool _forward_has_default_args() override {                            \
    return true;                                                         \
  }                                                                      \
  unsigned int _forward_num_required_args() override {                   \
    std::vector<std::pair<unsigned int, smith::nn::AnyValue>> args_info{ \
        __VA_ARGS__};                                                    \
    return std::begin(args_info)->first;                                 \
  }                                                                      \
  std::vector<smith::nn::AnyValue> _forward_populate_default_args(       \
      std::vector<smith::nn::AnyValue>&& arguments) override {           \
    std::vector<std::pair<unsigned int, smith::nn::AnyValue>> args_info{ \
        __VA_ARGS__};                                                    \
    unsigned int num_all_args = std::rbegin(args_info)->first + 1;       \
    SMITH_INTERNAL_ASSERT(                                               \
        arguments.size() >= _forward_num_required_args() &&              \
        arguments.size() <= num_all_args);                               \
    std::vector<smith::nn::AnyValue> ret = std::move(arguments);         \
    ret.reserve(num_all_args);                                           \
    for (auto& arg_info : args_info) {                                   \
      if (arg_info.first > ret.size() - 1)                               \
        ret.emplace_back(std::move(arg_info.second));                    \
    }                                                                    \
    return ret;                                                          \
  }
