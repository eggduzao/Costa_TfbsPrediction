#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <smith/csrc/python_headers.h>
#include <smith/csrc/utils/pybind.h>

#include <smith/csrc/autograd/python_cpp_function.h>
#include <smith/csrc/autograd/python_function.h>

// NOLINTNEXTLINE(misc-unused-alias-decls)
namespace py = pybind11;

namespace pybind11::detail {} // namespace pybind11::detail
