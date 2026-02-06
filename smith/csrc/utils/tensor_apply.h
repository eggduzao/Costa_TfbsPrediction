#pragma once

#include <ATen/core/Tensor.h>
#include <smith/csrc/python_headers.h>

namespace smith::utils {

const at::Tensor& apply_(const at::Tensor& self, PyObject* fn);
const at::Tensor& map_(
    const at::Tensor& self,
    const at::Tensor& other_,
    PyObject* fn);
const at::Tensor& map2_(
    const at::Tensor& self,
    const at::Tensor& x_,
    const at::Tensor& y_,
    PyObject* fn);

} // namespace smith::utils
