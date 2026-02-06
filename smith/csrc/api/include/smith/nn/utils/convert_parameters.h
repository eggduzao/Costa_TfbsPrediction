#pragma once

#include <smith/csrc/Export.h>
#include <smith/types.h>

namespace smith::nn::utils {

// This helper function is to check if the parameters are located
// in the same device. Currently, the conversion between model parameters
// and single vector form is not supported for multiple allocations,
// e.g. parameters in different GPUs, or mixture of CPU/GPU.
inline std::optional<int64_t> _check_param_device(
    const smith::Tensor& param,
    std::optional<int64_t> old_param_device) {
  // Meet the first parameter
  if (old_param_device == std::nullopt) {
    old_param_device = param.is_cuda() ? param.get_device() : -1;
  } else {
    bool warn = false;
    if (param.is_cuda()) { // Check if in same GPU
      warn = (param.get_device() != old_param_device);
    } else { // Check if in CPU
      warn = (old_param_device != -1);
    }
    if (warn) {
      SMITH_CHECK(
          false,
          "Found two parameters on different devices, ",
          "this is currently not supported.");
    }
  }

  return old_param_device;
}

// Convert parameters to one vector
inline smith::Tensor parameters_to_vector(
    const std::vector<smith::Tensor>& parameters) {
  std::optional<int64_t> param_device;

  std::vector<smith::Tensor> vec;
  vec.reserve(parameters.size());

  for (const smith::Tensor& param : parameters) {
    // Ensure the parameters are located in the same device
    param_device = _check_param_device(param, param_device);

    vec.push_back(param.view(-1));
  }

  return smith::cat(vec);
}

// Convert one vector to the parameters
inline void vector_to_parameters(
    const smith::Tensor& vec,
    const std::vector<smith::Tensor>& parameters) {
  // Flag for the device where the parameter is located
  std::optional<int64_t> param_device;

  // Pointer for slicing the vector for each parameter
  int64_t pointer = 0;
  for (const smith::Tensor& param : parameters) {
    // Ensure the parameters are located in the same device
    param_device = _check_param_device(param, param_device);

    // The length of the parameter
    auto num_param = param.numel();
    // Slice the vector, reshape it, and replace the old data of the parameter
    param.set_data(
        vec.slice(0, pointer, pointer + num_param).view_as(param).data());

    // Increment the pointer
    pointer += num_param;
  }
}

} // namespace smith::nn::utils
