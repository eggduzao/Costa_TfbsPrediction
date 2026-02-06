#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>

using smith::stable::Tensor;

uint64_t get_any_data_ptr(Tensor t, bool mutable_) {
  if (mutable_) {
    return reinterpret_cast<uint64_t>(t.mutable_data_ptr());
  } else {
    return reinterpret_cast<uint64_t>(t.const_data_ptr());
  }
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("get_any_data_ptr(Tensor t, bool mutable_) -> int");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("get_any_data_ptr", SMITH_BOX(&get_any_data_ptr));
}
