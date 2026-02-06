#include <smith/csrc/jit/passes/onnx/onnx_log.h>
#include <iostream>

namespace smith::jit::onnx {

namespace {
bool log_enabled = false;
std::shared_ptr<std::ostream> out;
} // namespace

bool is_log_enabled() {
  return log_enabled;
}

void set_log_enabled(bool enabled) {
  log_enabled = enabled;
}

void set_log_output_stream(std::shared_ptr<std::ostream> out_stream) {
  out = std::move(out_stream);
}

std::ostream& _get_log_output_stream() {
  return out ? *out : std::cout;
}

} // namespace smith::jit::onnx
