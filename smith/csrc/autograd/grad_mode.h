#pragma once

#include <ATen/core/grad_mode.h>
#include <smith/csrc/Export.h>

namespace smith::autograd {

using GradMode = at::GradMode;
using AutoGradMode = at::AutoGradMode;

} // namespace smith::autograd
