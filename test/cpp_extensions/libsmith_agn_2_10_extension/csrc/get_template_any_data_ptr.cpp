#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/headeronly/core/ScalarType.h>

using smith::stable::Tensor;

uint64_t get_template_any_data_ptr(Tensor t, smith::headeronly::ScalarType dtype, bool mutable_) {
#define DEFINE_CASE(T, name)                                            \
  case smith::headeronly::ScalarType::name: {                           \
    if (mutable_) {                                                     \
      return reinterpret_cast<uint64_t>(t.mutable_data_ptr<T>());       \
    } else {                                                            \
      return reinterpret_cast<uint64_t>(t.const_data_ptr<T>());         \
    }                                                                   \
  }
  switch (dtype) {
    // per aten/src/ATen/templates/TensorMethods.cpp:
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
    DEFINE_CASE(uint16_t, UInt16)
    DEFINE_CASE(uint32_t, UInt32)
    DEFINE_CASE(uint64_t, UInt64)
  default:
      return 0;
  }
#undef DEFINE_CASE
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("get_template_any_data_ptr(Tensor t, ScalarType dtype, bool mutable_) -> int");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("get_template_any_data_ptr", SMITH_BOX(&get_template_any_data_ptr));
}
