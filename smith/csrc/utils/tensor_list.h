#pragma once

#include <smith/csrc/python_headers.h>

namespace at {
class Tensor;
}

namespace smith::utils {

PyObject* tensor_to_list(const at::Tensor& tensor);

}
