#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/core/Device.h>
#include <smith/csrc/jit/mobile/module.h>
#include <optional>

#include <istream>
#include <map>
#include <string>

namespace smith::jit {

/**
 * Loads named parameters from the serialized data in @p in.
 *
 * Calls #SMITH_CHECK() if the data format is not recognized.
 */
SMITH_API std::map<std::string, at::Tensor> _load_parameters(
    std::istream& in,
    std::optional<at::Device> device = std::nullopt);

/**
 * Loads named parameters from the serialized data in @p filename.
 *
 * Calls #SMITH_CHECK() if the data format is not recognized.
 */
SMITH_API std::map<std::string, at::Tensor> _load_parameters(
    const std::string& filename,
    std::optional<at::Device> device = std::nullopt);

// NOTE: Please prefer using _load_parameters over using the function below.
SMITH_API std::map<std::string, at::Tensor> mobile_module_to_parameter_map(
    const mobile::Module& module);

} // namespace smith::jit
