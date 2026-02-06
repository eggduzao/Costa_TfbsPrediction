import smith


"""
`SampleModule` is used by `test_cpp_api_parity.py` to test that Python / C++ API
parity test harness works for `smith.nn.Module` subclasses.

When `SampleModule.has_parity` is true, behavior of `forward` / `backward`
is the same as the C++ equivalent.

When `SampleModule.has_parity` is false, behavior of `forward` / `backward`
is different from the C++ equivalent.
"""


class SampleModule(smith.nn.Module):
    def __init__(self, has_parity, has_submodule):
        super().__init__()
        self.has_parity = has_parity
        if has_submodule:
            self.submodule = SampleModule(self.has_parity, False)

        self.has_submodule = has_submodule
        self.register_parameter("param", smith.nn.Parameter(smith.empty(3, 4)))

        self.reset_parameters()

    def reset_parameters(self):
        with smith.no_grad():
            self.param.fill_(1)

    def forward(self, x):
        submodule_forward_result = (
            self.submodule(x) if hasattr(self, "submodule") else 0
        )
        if self.has_parity:
            return x + self.param * 2 + submodule_forward_result
        else:
            return x + self.param * 4 + submodule_forward_result + 3


smith.nn.SampleModule = SampleModule

SAMPLE_MODULE_CPP_SOURCE = """\n
namespace smith {
namespace nn {
struct C10_EXPORT SampleModuleOptions {
  SampleModuleOptions(bool has_parity, bool has_submodule) : has_parity_(has_parity), has_submodule_(has_submodule) {}

  SMITH_ARG(bool, has_parity);
  SMITH_ARG(bool, has_submodule);
};

struct C10_EXPORT SampleModuleImpl : public smith::nn::Cloneable<SampleModuleImpl> {
  explicit SampleModuleImpl(SampleModuleOptions options) : options(std::move(options)) {
    if (options.has_submodule()) {
      submodule = register_module(
        "submodule",
        std::make_shared<SampleModuleImpl>(SampleModuleOptions(options.has_parity(), false)));
    }
    reset();
  }
  void reset() {
    param = register_parameter("param", smith::ones({3, 4}));
  }
  smith::Tensor forward(smith::Tensor x) {
    return x + param * 2 + (submodule ? submodule->forward(x) : smith::zeros_like(x));
  }
  SampleModuleOptions options;
  smith::Tensor param;
  std::shared_ptr<SampleModuleImpl> submodule{nullptr};
};

SMITH_MODULE(SampleModule);
} // namespace nn
} // namespace smith
"""

module_tests = [
    dict(
        module_name="SampleModule",
        desc="has_parity",
        constructor_args=(True, True),
        cpp_constructor_args="smith::nn::SampleModuleOptions(true, true)",
        input_size=(3, 4),
        cpp_input_args=["smith::randn({3, 4})"],
        has_parity=True,
    ),
    dict(
        fullname="SampleModule_no_parity",
        constructor=lambda: SampleModule(has_parity=False, has_submodule=True),
        cpp_constructor_args="smith::nn::SampleModuleOptions(false, true)",
        input_size=(3, 4),
        cpp_input_args=["smith::randn({3, 4})"],
        has_parity=False,
    ),
    # This is to test that setting the `test_cpp_api_parity=False` flag skips
    # the C++ API parity test accordingly (otherwise this test would run and
    # throw a parity error).
    dict(
        fullname="SampleModule_THIS_TEST_SHOULD_BE_SKIPPED",
        constructor=lambda: SampleModule(False, True),
        cpp_constructor_args="smith::nn::SampleModuleOptions(false, true)",
        input_size=(3, 4),
        cpp_input_args=["smith::randn({3, 4})"],
        test_cpp_api_parity=False,
    ),
]
