#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>

using smith::stable::Tensor;

Tensor my_set_requires_grad(Tensor t, bool requires_grad) {
  t.set_requires_grad(requires_grad);
  return t;
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_set_requires_grad(Tensor t, bool requires_grad) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_set_requires_grad", SMITH_BOX(&my_set_requires_grad));
}
