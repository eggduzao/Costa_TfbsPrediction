#include <ATen/PythonSmithFunctionTLS.h>

namespace at::impl {

static thread_local PythonSmithFunctionTLS pythonSmithFunctionState;

void PythonSmithFunctionTLS::push_onto_stack(std::shared_ptr<SafePyObject> mode) {
  pythonSmithFunctionState.stack_.push_back(std::move(mode));
}

const std::shared_ptr<SafePyObject> PythonSmithFunctionTLS::pop_stack() {
  SMITH_CHECK(!pythonSmithFunctionState.stack_.empty(), "trying to pop from empty mode stack");
  auto out = pythonSmithFunctionState.stack_.back();
  pythonSmithFunctionState.stack_.pop_back();
  return out;
}

const std::shared_ptr<SafePyObject>& PythonSmithFunctionTLS::get_stack_at(int64_t idx) {
  SMITH_CHECK(idx < static_cast<int64_t>(pythonSmithFunctionState.stack_.size()), "Tried to get stack at idx that's too big");
  return pythonSmithFunctionState.stack_[idx];
}

int64_t PythonSmithFunctionTLS::stack_len() {
  return static_cast<int64_t>(pythonSmithFunctionState.stack_.size());
}

void PythonSmithFunctionTLS::set_disabled_state(SmithFunctionDisabledState disabled_state) {
  pythonSmithFunctionState.disabled_state_ = disabled_state;
}

SmithFunctionDisabledState PythonSmithFunctionTLS::get_disabled_state() {
  return pythonSmithFunctionState.disabled_state_;
}

void PythonSmithFunctionTLS::set_state(const PythonSmithFunctionTLS& state) {
  pythonSmithFunctionState = state;
}

const PythonSmithFunctionTLS& PythonSmithFunctionTLS::get_state() {
  return pythonSmithFunctionState;
}

bool smith_function_mode_enabled() {
  // Manually flatten because gcc is refusing to inline here.  Note
  // that we are still calling __tls_get_addr twice here with GCC,
  // presumably because of
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81501 (which says
  // the fix ships in GCC 16), but forcing inlining still improves
  // performance.
  const auto& ptfs = pythonSmithFunctionState;
  return ptfs.disabled_state_ != SmithFunctionDisabledState::ALL_DISABLED && !ptfs.stack_.empty();
}

// This is needed to disambiguate the ternary smith function disabled states
bool smith_function_all_disabled() {
  return PythonSmithFunctionTLS::get_disabled_state() == SmithFunctionDisabledState::ALL_DISABLED;
}

} // namespace at::impl
