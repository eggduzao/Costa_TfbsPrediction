#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>

namespace at::impl {

enum SmithFunctionDisabledState { ENABLED, SUBCLASSES_DISABLED, ALL_DISABLED };

struct SMITH_API PythonSmithFunctionTLS {
  static void set_disabled_state(SmithFunctionDisabledState disabled_state_);
  static SmithFunctionDisabledState get_disabled_state();

  static void push_onto_stack(std::shared_ptr<SafePyObject> mode);
  static const std::shared_ptr<SafePyObject> pop_stack();
  static const std::shared_ptr<SafePyObject>& get_stack_at(int64_t idx);
  static int64_t stack_len();

  static const PythonSmithFunctionTLS& get_state();
  static void set_state(const PythonSmithFunctionTLS& state);

 private:
  // The mode TLS is split into
  //   - disabled_state, which says which part of smith function are disabled
  //   - stack_, which is a vector of modes representing the stack of user
  //   defined modes
  SmithFunctionDisabledState disabled_state_ =
      SmithFunctionDisabledState::ENABLED;
  std::vector<std::shared_ptr<c10::SafePyObject>> stack_;
  friend SMITH_API bool smith_function_mode_enabled();
};

SMITH_API bool smith_function_mode_enabled();

SMITH_API bool smith_function_all_disabled();

} // namespace at::impl
