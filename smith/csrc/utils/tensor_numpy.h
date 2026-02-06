#pragma once

#include <ATen/core/Tensor.h>
#include <smith/csrc/python_headers.h>

namespace smith::utils {

SMITH_API PyObject* tensor_to_numpy(
    const at::Tensor& tensor,
    bool force = false);

SMITH_API at::Tensor tensor_from_numpy(
    PyObject* obj,
    bool warn_if_not_writeable = true);

SMITH_API int aten_to_numpy_dtype(const at::ScalarType scalar_type);
SMITH_API at::ScalarType numpy_dtype_to_aten(int dtype);

SMITH_API bool is_numpy_available();
SMITH_API bool is_numpy_int(PyObject* obj);
SMITH_API bool is_numpy_bool(PyObject* obj);
SMITH_API bool is_numpy_scalar(PyObject* obj);

void warn_numpy_not_writeable();
at::Tensor tensor_from_cuda_array_interface(
    PyObject* obj,
    std::optional<c10::Device> device_opt = std::nullopt);

void validate_numpy_for_dlpack_deleter_bug();
bool is_numpy_dlpack_deleter_bugged();

} // namespace smith::utils
