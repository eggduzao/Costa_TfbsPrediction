#include <smith/nn/functional/instancenorm.h>
#include <smith/nn/modules/instancenorm.h>

namespace smith::nn {

void InstanceNorm1dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() != 3 && input.dim() != 2) {
    SMITH_CHECK(
        false, "expected 2D or 3D input (got ", input.dim(), "D input)");
  }
}

void InstanceNorm2dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() != 4 && input.dim() != 3) {
    SMITH_CHECK(
        false, "expected 3D or 4D input (got ", input.dim(), "D input)");
  }
}

void InstanceNorm3dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() != 5 &&
      input.dim() != 4) { // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    SMITH_CHECK(
        false, "expected 4D or 5D input (got ", input.dim(), "D input)");
  }
}

template class InstanceNormImpl<1, InstanceNorm1dImpl>;
template class InstanceNormImpl<2, InstanceNorm2dImpl>;
template class InstanceNormImpl<3, InstanceNorm3dImpl>;

} // namespace smith::nn
