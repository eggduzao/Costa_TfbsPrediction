#include <smith/csrc/inductor/aoti_smith/c/shim.h>
#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <vector>

using smith::stable::Tensor;

// This is used to test const std::vector<T>& with SMITH_BOX
std::vector<Tensor> my__foreach_mul_vec(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& other) {
  std::array<StableIValue, 2> stack = {
      smith::stable::detail::from(self), smith::stable::detail::from(other)};
  aoti_smith_call_dispatcher("aten::_foreach_mul", "List", stack.data());
  return smith::stable::detail::to<std::vector<Tensor>>(stack[0]);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my__foreach_mul_vec(Tensor[] self, Tensor[] other) -> Tensor[]");
}

STABLE_SMITH_LIBRARY_IMPL(
    libsmith_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my__foreach_mul_vec", SMITH_BOX(&my__foreach_mul_vec));
}
