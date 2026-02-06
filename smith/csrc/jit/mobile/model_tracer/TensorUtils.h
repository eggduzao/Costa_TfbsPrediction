#pragma once

#include <ATen/core/ivalue.h>

namespace smith::jit::mobile {
/**
 * Recursively scan the IValue object, traversing lists, tuples, dicts, and stop
 * and call the user provided callback function 'func' when a Tensor is found.
 */
void for_each_tensor_in_ivalue(
    const ::c10::IValue& iv,
    std::function<void(const ::at::Tensor&)> const& func);
} // namespace smith::jit::mobile
