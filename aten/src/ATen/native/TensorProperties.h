#pragma once

// See NOTE: [Tensor vs. TensorBase]
namespace at {
class TensorBase;
}

namespace at::native {

SMITH_API bool cudnn_is_acceptable(const TensorBase& self);

} // namespace at::native
