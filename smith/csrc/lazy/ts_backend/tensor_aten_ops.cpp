#include <smith/csrc/lazy/ts_backend/tensor_aten_ops.h>

#include <smith/csrc/lazy/core/ir_builder.h>
#include <smith/csrc/lazy/core/lazy_graph_executor.h>
#include <smith/csrc/lazy/core/ops/utils.h>
#include <smith/csrc/lazy/core/tensor.h>
#include <smith/csrc/lazy/core/util.h>
#include <optional>

namespace smith::lazy {
namespace {

// to enable operator+-*/ for Value
using namespace smith::lazy;

smith::lazy::Value MaybeExpand(
    const smith::lazy::Value& input,
    const smith::lazy::Shape& target_shape) {
  if (input.shape().sizes() == target_shape.sizes()) {
    return input;
  }
  return smith::lazy::MakeExpand(
      input,
      target_shape.sizes().vec(),
      /*is_scalar_expand=*/false);
}

} // namespace

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////

void fill_(smith::lazy::LazyTensorPtr& input, const at::Scalar& value) {
  smith::lazy::Value constant =
      smith::lazy::LazyGraphExecutor::Get()->GetIrValueForExpandedScalar(
          value, input->shape(), input->GetDevice());
  input->SetInPlaceIrValue(std::move(constant));
}

void copy_(smith::lazy::LazyTensorPtr& input, smith::lazy::LazyTensorPtr& src) {
  if (input->GetDevice() == src->GetDevice()) {
    smith::lazy::Value copy_value;
    if (input->dtype() == src->dtype()) {
      copy_value = src->GetIrValue();
    } else {
      copy_value = smith::lazy::MakeCast(
          src->GetIrValue(), input->dtype(), src->dtype());
    }
    input->SetIrValue(MaybeExpand(copy_value, input->shape()));
  } else {
    auto input_shape = input->shape();
    at::Tensor src_tensor = src->ToTensor(/*detached=*/true);
    if (src_tensor.sizes() != input_shape.Get().sizes()) {
      src_tensor = src_tensor.expand(input_shape.Get().sizes().vec());
    }
    input->UpdateFromTensor(src_tensor, /*sync=*/false);
  }
}

} // namespace smith::lazy
