#pragma once

#include <smith/csrc/jit/tensorexpr/kernel.h>

namespace smith::jit::tensorexpr {

Tensor computeBatchNorm(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device);

} // namespace smith::jit::tensorexpr
