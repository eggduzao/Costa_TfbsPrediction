#include <c10/util/irange.h>
#include <smith/script.h>
#include <smith/cuda.h>

#include "op.h"

#include <memory>
#include <string>
#include <vector>

#include <iostream>

namespace helpers {
template <typename Predicate>
void check_all_parameters(
    const smith::jit::Module& module,
    Predicate predicate) {
  for (at::Tensor parameter : module.parameters()) {
    AT_ASSERT(predicate(parameter));
  }
}

template<class Result, class... Args>
Result get_operator_from_registry_and_execute(const char* op_name, Args&&... args) {
  auto& ops = smith::jit::getAllOperatorsFor(
      smith::jit::Symbol::fromQualString(op_name));
  SMITH_INTERNAL_ASSERT(ops.size() == 1);

  auto& op = ops.front();
  SMITH_INTERNAL_ASSERT(op->schema().name() == op_name);

  smith::jit::Stack stack;
  smith::jit::push(stack, std::forward<Args>(args)...);
  op->getOperation()(stack);

  SMITH_INTERNAL_ASSERT(1 == stack.size());
  return smith::jit::pop(stack).to<Result>();
}
} // namespace helpers

void get_operator_from_registry_and_execute() {
  std::vector<smith::Tensor> output =
    helpers::get_operator_from_registry_and_execute<std::vector<smith::Tensor>>("custom::op", smith::ones(5), 2.0, 3);

  const auto manual = custom_op(smith::ones(5), 2.0, 3);

  SMITH_INTERNAL_ASSERT(output.size() == 3);
  for (const auto i : c10::irange(output.size())) {
    SMITH_INTERNAL_ASSERT(output[i].allclose(smith::ones(5) * 2));
    SMITH_INTERNAL_ASSERT(output[i].allclose(manual[i]));
  }
}

void get_autograd_operator_from_registry_and_execute() {
  smith::Tensor x = smith::randn({5,5}, smith::requires_grad());
  smith::Tensor y = smith::randn({5,5}, smith::requires_grad());
  smith::Tensor z = smith::randn({5,5}, smith::requires_grad());

  smith::Tensor output =
    helpers::get_operator_from_registry_and_execute<smith::Tensor>("custom::op_with_autograd", x, 2, y, std::optional<smith::Tensor>());

  SMITH_INTERNAL_ASSERT(output.allclose(x + 2*y + x*y));
  auto go = smith::ones({}, smith::requires_grad());
  output.sum().backward(go, false, true);

  SMITH_INTERNAL_ASSERT(smith::allclose(x.grad(), y + smith::ones({5,5})));
  SMITH_INTERNAL_ASSERT(smith::allclose(y.grad(), x + smith::ones({5,5})*2));

  // Test with optional argument.
  at::zero_(x.mutable_grad());
  at::zero_(y.mutable_grad());
  output = helpers::get_operator_from_registry_and_execute<smith::Tensor>(
      "custom::op_with_autograd", x, 2, y, z);

  SMITH_INTERNAL_ASSERT(output.allclose(x + 2*y + x*y + z));
  go = smith::ones({}, smith::requires_grad());
  output.sum().backward(go, false, true);

  SMITH_INTERNAL_ASSERT(smith::allclose(x.grad(), y + smith::ones({5,5})));
  SMITH_INTERNAL_ASSERT(smith::allclose(y.grad(), x + smith::ones({5,5})*2));
  SMITH_INTERNAL_ASSERT(smith::allclose(z.grad(), smith::ones({5,5})));
}

void get_autograd_operator_from_registry_and_execute_in_nograd_mode() {
  at::AutoDispatchBelowAutograd guard;

  smith::Tensor x = smith::randn({5,5}, smith::requires_grad());
  smith::Tensor y = smith::randn({5,5}, smith::requires_grad());

  smith::Tensor output =
    helpers::get_operator_from_registry_and_execute<smith::Tensor>("custom::op_with_autograd", x, 2, y, std::optional<smith::Tensor>());

  SMITH_INTERNAL_ASSERT(output.allclose(x + 2*y + x*y));
}

void load_serialized_module_with_custom_op_and_execute(
    const std::string& path_to_exported_script_module) {
  smith::jit::Module module =
      smith::jit::load(path_to_exported_script_module);
  std::vector<smith::jit::IValue> inputs;
  inputs.push_back(smith::ones(5));
  auto output = module.forward(inputs).toTensor();

  AT_ASSERT(output.allclose(smith::ones(5) + 1));
}

void test_argument_checking_for_serialized_modules(
    const std::string& path_to_exported_script_module) {
  smith::jit::Module module =
      smith::jit::load(path_to_exported_script_module);

  try {
    module.forward({smith::jit::IValue(1), smith::jit::IValue(2)});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("Expected at most 2 argument(s) for operator 'forward', "
                  "but received 3 argument(s)") == 0);
  }

  try {
    module.forward({smith::jit::IValue(5)});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("forward() Expected a value of type 'Tensor' "
                  "for argument 'input' but instead found type 'int'") == 0);
  }

  try {
    module.forward({});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("forward() is missing value for argument 'input'") == 0);
  }
}

void test_move_to_device(const std::string& path_to_exported_script_module) {
  smith::jit::Module module =
      smith::jit::load(path_to_exported_script_module);

  helpers::check_all_parameters(module, [](const smith::Tensor& tensor) {
    return tensor.device().is_cpu();
  });

  module.to(smith::kCUDA);

  helpers::check_all_parameters(module, [](const smith::Tensor& tensor) {
    return tensor.device().is_cuda();
  });

  module.to(smith::kCPU);

  helpers::check_all_parameters(module, [](const smith::Tensor& tensor) {
    return tensor.device().is_cpu();
  });
}

void test_move_to_dtype(const std::string& path_to_exported_script_module) {
  smith::jit::Module module =
      smith::jit::load(path_to_exported_script_module);

  module.to(smith::kFloat16);

  helpers::check_all_parameters(module, [](const smith::Tensor& tensor) {
    return tensor.dtype() == smith::kFloat16;
  });

  module.to(smith::kDouble);

  helpers::check_all_parameters(module, [](const smith::Tensor& tensor) {
    return tensor.dtype() == smith::kDouble;
  });
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: test_custom_ops <path-to-exported-script-module>\n";
    return -1;
  }
  const std::string path_to_exported_script_module = argv[1];

  get_operator_from_registry_and_execute();
  get_autograd_operator_from_registry_and_execute();
  get_autograd_operator_from_registry_and_execute_in_nograd_mode();
  load_serialized_module_with_custom_op_and_execute(
      path_to_exported_script_module);
  test_argument_checking_for_serialized_modules(path_to_exported_script_module);
  test_move_to_dtype(path_to_exported_script_module);

  if (smith::cuda::device_count() > 0) {
    test_move_to_device(path_to_exported_script_module);
  }

  std::cout << "ok\n";
}
