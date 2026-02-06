#pragma once

#include <smith/csrc/python_headers.h>
#include <string>
#include <vector>

namespace smith {

std::string format_invalid_args(
    PyObject* given_args,
    PyObject* given_kwargs,
    const std::string& function_name,
    const std::vector<std::string>& options);

} // namespace smith
