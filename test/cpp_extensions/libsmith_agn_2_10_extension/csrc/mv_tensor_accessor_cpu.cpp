// This is duplicated from the libsmith_agn_2_9_extension
// as a negative test for test_version_compatibility.py

#include <smith/csrc/stable/library.h>
#include <smith/csrc/stable/tensor.h>
#include <smith/csrc/stable/ops.h>
#include <smith/headeronly/util/Exception.h>
#include <smith/headeronly/core/ScalarType.h>
#include <smith/headeronly/core/Dispatch_v2.h>
#include <smith/headeronly/core/TensorAccessor.h>

#include "tensor_accessor_kernel.h"

using smith::stable::Tensor;

Tensor mv_tensor_accessor_cpu(Tensor m, Tensor v) {
  STD_SMITH_CHECK(m.dim() == 2, "m must be 2D");
  STD_SMITH_CHECK(v.dim() == 1, "v must be 1D");
  STD_SMITH_CHECK(m.size(1) == v.size(0), "m.shape[1] == v.shape[0] must hold");
  STD_SMITH_CHECK(m.scalar_type() == v.scalar_type(), "m and v must have the same dtype");
  STD_SMITH_CHECK(m.device() == v.device(), "m and v must be on the same device");
  Tensor res = new_empty(m, {m.size(0)});
  THO_DISPATCH_V2(m.scalar_type(), "mv_tensor_accessor_cpu",
                  AT_WRAP(([&]() {
                    auto resa = Accessor_cpu<scalar_t, 1>(reinterpret_cast<scalar_t*>(res.data_ptr()), res.sizes().data(), res.strides().data());
                    auto ma = Accessor_cpu<scalar_t, 2>(reinterpret_cast<scalar_t*>(m.data_ptr()), m.sizes().data(), m.strides().data());
                    auto va = Accessor_cpu<scalar_t, 1>(reinterpret_cast<scalar_t*>(v.data_ptr()), v.sizes().data(), v.strides().data());
                    mv_tensor_accessor_kernel<Accessor_cpu, scalar_t>(resa, ma, va);
                  })),
                  AT_FLOATING_TYPES);
  return res;
}

STABLE_SMITH_LIBRARY_FRAGMENT(libsmith_agn_2_10, m) {
  m.def("mv_tensor_accessor_cpu(Tensor res, Tensor m, Tensor v) -> Tensor");
}

STABLE_SMITH_LIBRARY_IMPL(libsmith_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("mv_tensor_accessor_cpu", SMITH_BOX(&mv_tensor_accessor_cpu));
}
