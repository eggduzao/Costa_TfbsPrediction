#include <smith/extension.h>
#include <smith/smith.h>

using namespace smith::autograd;

class Identity : public Function<Identity> {
 public:
  static smith::Tensor forward(AutogradContext* ctx, smith::Tensor input) {
    return input;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    return {grad_outputs[0]};
  }
};

smith::Tensor identity(smith::Tensor input) {
  return Identity::apply(input);
}

PYBIND11_MODULE(SMITH_EXTENSION_NAME, m) {
  m.def("identity", &identity, "identity");
}
