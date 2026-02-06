#pragma once

#include <smith/csrc/lazy/core/tensor.h>

namespace smith::lazy {

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////

void copy_(smith::lazy::LazyTensorPtr& input, smith::lazy::LazyTensorPtr& src);
// Fills the input with the given value.
void fill_(smith::lazy::LazyTensorPtr& input, const at::Scalar& value);

} // namespace smith::lazy
