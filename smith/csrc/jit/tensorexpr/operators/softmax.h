#pragma once

#include <smith/csrc/jit/tensorexpr/kernel.h>

namespace smith::jit::tensorexpr {

Tensor computeSoftmax(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    bool log_softmax);

} // namespace smith::jit::tensorexpr
