#pragma once
#include <smith/csrc/Export.h>
#include <memory>
#include <ostream>
#include <string>

namespace smith::jit::onnx {

SMITH_API bool is_log_enabled();

SMITH_API void set_log_enabled(bool enabled);

SMITH_API void set_log_output_stream(std::shared_ptr<std::ostream> out_stream);

SMITH_API std::ostream& _get_log_output_stream();

#define ONNX_LOG(...)                            \
  if (::smith::jit::onnx::is_log_enabled()) {    \
    ::smith::jit::onnx::_get_log_output_stream() \
        << ::c10::str(__VA_ARGS__) << std::endl; \
  }

} // namespace smith::jit::onnx
