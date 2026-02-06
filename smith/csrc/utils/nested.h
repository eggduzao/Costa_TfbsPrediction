#pragma once

#include <smith/csrc/python_headers.h>
#include <smith/csrc/utils/python_arg_parser.h>

#include <ATen/core/Tensor.h>

namespace smith::utils {

at::Tensor nested_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

} // namespace smith::utils
