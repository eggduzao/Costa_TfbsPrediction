#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/ops.h>
#include <smith/csrc/stable/tensor.h>

using smith::stable::Tensor;

Tensor my_subtract(const Tensor& self, const Tensor& other, double alpha) {
  return smith::stable::subtract(self, other, alpha);
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("my_subtract(Tensor self, Tensor other, float alpha=1.0) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_subtract", SMITH_BOX(&my_subtract));
}
