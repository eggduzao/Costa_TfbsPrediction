#pragma once

#include <smith/csrc/jit/ir/ir.h>

#include <vector>

namespace smith::jit {

SMITH_API TypePtr getTensorType(const at::Tensor& t, bool complete);

SMITH_API TypePtr inferShapeAndTypeForInput(
    TypePtr input_type,
    Stack::const_iterator& s_iter,
    const Stack::const_iterator& s_iter_end,
    bool complete);

SMITH_API void setInputTensorTypes(
    Graph& g,
    const Stack& stack,
    bool complete,
    const std::vector<int>& param_count_list = {});

} // namespace smith::jit
