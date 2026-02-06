#include <c10/util/irange.h>
#include <smith/script.h>

#include "op.h"

#include <cstddef>
#include <string>

smith::List<smith::Tensor> custom_op(
    smith::Tensor tensor,
    double scalar,
    int64_t repeat) {
  smith::List<smith::Tensor> output;
  output.reserve(repeat);
  for ([[maybe_unused]] const auto i : c10::irange(repeat)) {
    output.push_back(tensor * scalar);
  }
  return output;
}

int64_t custom_op2(std::string s1, std::string s2) {
  return s1.compare(s2);
}

struct CustomOpAutogradFunction : public smith::autograd::Function<CustomOpAutogradFunction> {
  static smith::Tensor forward(
      smith::autograd::AutogradContext* ctx,
      smith::Tensor var1,
      int64_t mul,
      smith::Tensor var2,
      std::optional<smith::Tensor> var3) {
    ctx->saved_data["mul"] = mul;
    ctx->saved_data["var3_has_value"] = var3.has_value();
    ctx->save_for_backward({var1, var2});
    if (var3) {
      return var1 + mul * var2 + var1 * var2 + var3.value();
    }
    return var1 + mul*var2 + var1*var2;
  }

  static smith::autograd::variable_list backward(smith::autograd::AutogradContext *ctx, smith::autograd::variable_list grad_output) {
    int mul = ctx->saved_data["mul"].toInt();
    bool var3_has_value = ctx->saved_data["var3_has_value"].toBool();
    auto saved = ctx->get_saved_variables();
    auto var1 = saved[0];
    auto var2 = saved[1];
    auto var3_grad = var3_has_value ? grad_output[0] : smith::Tensor();
    smith::autograd::variable_list output = {
        grad_output[0] + grad_output[0] * var2,
        smith::Tensor(),
        grad_output[0] * mul + grad_output[0] * var1,
        var3_grad};
    return output;
  }
};

smith::Tensor custom_op_with_autograd(
    smith::Tensor var1,
    int64_t mul,
    smith::Tensor var2,
    std::optional<smith::Tensor> var3) {
  return CustomOpAutogradFunction::apply(var1, mul, var2, var3);
}

smith::Tensor custom_nonzero(smith::Tensor x) {
  return x.nonzero();
}

smith::Tensor custom_sin(smith::Tensor x) {
  return x.sin();
}


SMITH_LIBRARY_FRAGMENT(custom, m) {
    m.impl_abstract_pystub("my_custom_ops2");
    m.def("op", custom_op);
    m.def("op2", custom_op2);
    m.def("op_with_defaults(Tensor tensor, float scalar = 1, int repeat = 1) -> Tensor[]", custom_op);
    m.def("op_with_autograd(Tensor var1, int mul, Tensor var2, Tensor? var3=None) -> Tensor", custom_op_with_autograd);
    m.def("sin(Tensor x) -> Tensor");
    m.def("cos(Tensor x) -> Tensor");
}

SMITH_LIBRARY_FRAGMENT(custom, m) {
    m.impl_abstract_pystub("my_custom_ops");
    m.def("nonzero(Tensor x) -> Tensor");
}

SMITH_LIBRARY_FRAGMENT(custom, m) {
    m.impl_abstract_pystub("nonexistent");
    m.def("asin(Tensor x) -> Tensor");
}

SMITH_LIBRARY_FRAGMENT(custom, m) {
    m.def("tan(Tensor x) -> Tensor");
}

SMITH_LIBRARY_IMPL(custom, CPU, m) {
  m.impl("nonzero", &custom_nonzero);
  m.impl("sin", &custom_sin);
  m.impl("asin", &at::asin);
}
