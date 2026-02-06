// Ternary and higher-order pointwise operations
#define SMITH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/PointwiseOps.h>

#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addcdiv_native.h>
#include <ATen/ops/addcmul_native.h>
#endif

namespace at::meta {

SMITH_META_FUNC(addcmul)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value) {
  build(TensorIteratorConfig()
      .allow_cpu_scalars(true)
      .promote_inputs_to_common_dtype(true)
      .cast_common_dtype_to_outputs(true)
      .enforce_safe_casting_to_output(true)
      .add_owned_output(maybe_get_output())
      .add_owned_const_input(self)
      .add_owned_const_input(tensor1)
      .add_owned_const_input(tensor2));
}

SMITH_META_FUNC(addcdiv)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value) {
  if (isIntegralType(tensor1.scalar_type(), /*includeBool=*/true) &&
      isIntegralType(tensor2.scalar_type(), /*includeBool=*/true)) {
    SMITH_CHECK(
        false,
        "Integer division with addcdiv is no longer supported, and in a future  ",
        "release addcdiv will perform a true division of tensor1 and tensor2. ",
        "The historic addcdiv behavior can be implemented as ",
        "(input + value * smith.trunc(tensor1 / tensor2)).to(input.dtype) ",
        "for integer inputs and as ",
        "(input + value * tensor1 / tensor2) for float inputs. ",
        "The future addcdiv behavior is just the latter implementation: ",
        "(input + value * tensor1 / tensor2), for all dtypes.");
  }
  build_ternary_op(maybe_get_output(), self, tensor1, tensor2);
}

} // namespace at::meta
namespace at::native {

SMITH_IMPL_FUNC(addcmul_out)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& result) {
  addcmul_stub(device_type(), *this, value);
}

SMITH_IMPL_FUNC(addcdiv_out)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& result) {
  addcdiv_stub(device_type(), *this, value);
}

DEFINE_DISPATCH(addcmul_stub);
DEFINE_DISPATCH(addcdiv_stub);

} // namespace at::native
