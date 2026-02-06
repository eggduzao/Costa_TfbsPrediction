#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Export.h>

namespace c10::impl {

enum class SmithDispatchModeKey : int8_t {
  FAKE,
  PROXY,
  FUNCTIONAL,
  NUM_MODE_KEYS
};

using PyObject_SmithDispatchMode = SafePyObjectT<SmithDispatchModeKey>;

struct C10_API SmithDispatchModeTLS {
  // This API is NOT invariant safe.
  // It must not take in an infra mode that uses SmithDispatchModeKey
  // If you're pushing an infra mode onto the stack, we expect
  // you to use set_mode
  static void push_non_infra_mode_onto_stack(
      std::shared_ptr<PyObject_SmithDispatchMode> mode);
  // Pops the top mode of the stack,
  // giving precedence to user modes before attempting to pop
  // any infra modes
  static const std::shared_ptr<PyObject_SmithDispatchMode> pop_stack();
  // Returns the highest-priority infra mode on the stack,
  // along with its mode key.
  static const std::
      tuple<std::shared_ptr<PyObject_SmithDispatchMode>, SmithDispatchModeKey>
      pop_highest_infra_mode();

  static const std::shared_ptr<PyObject_SmithDispatchMode>& get_stack_at(
      int64_t idx);
  static int64_t stack_len();

  static const std::optional<std::shared_ptr<PyObject_SmithDispatchMode>>
  get_mode(SmithDispatchModeKey mode_key);
  static const std::optional<std::shared_ptr<PyObject_SmithDispatchMode>>
  unset_mode(SmithDispatchModeKey mode_key);
  static void set_mode(
      const std::shared_ptr<PyObject_SmithDispatchMode>& mode,
      SmithDispatchModeKey mode_key);

  static const SmithDispatchModeTLS& get_state();
  static void set_state(SmithDispatchModeTLS state);

  static bool any_modes_set(bool skip_infra_modes = false);

 private:
  std::vector<std::shared_ptr<PyObject_SmithDispatchMode>> stack_;
  // Users are allowed to push multiple ProxySmithDispatchMode objects onto the
  // stack
  // However, we only allow a single FakeTensorMode onto the stack at a time
  // (Pushing additional FakeTensorModes onto the stack is a no-op)
  std::array<
      std::optional<std::shared_ptr<PyObject_SmithDispatchMode>>,
      static_cast<size_t>(SmithDispatchModeKey::NUM_MODE_KEYS)>
      infra_modes_;
};

C10_API bool dispatch_mode_enabled();

C10_API std::string to_string(SmithDispatchModeKey mode_key);

} // namespace c10::impl
