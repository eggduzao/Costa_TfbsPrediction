#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/SmithDispatchModeTLS.h>
#include <c10/util/irange.h>

#include <utility>

namespace c10::impl {

thread_local static SmithDispatchModeTLS smithDispatchModeState;

bool SmithDispatchModeTLS::any_modes_set(bool skip_infra_modes) {
  if (!smithDispatchModeState.stack_.empty())
    return true;
  if (!skip_infra_modes) {
    for (const auto i : c10::irange(
             static_cast<size_t>(SmithDispatchModeKey::NUM_MODE_KEYS))) {
      if (smithDispatchModeState.infra_modes_[i] != std::nullopt) {
        return true;
      }
    }
  }
  return false;
}

void SmithDispatchModeTLS::push_non_infra_mode_onto_stack(
    std::shared_ptr<PyObject_SmithDispatchMode> mode) {
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
  smithDispatchModeState.stack_.push_back(std::move(mode));
}

const std::shared_ptr<PyObject_SmithDispatchMode> SmithDispatchModeTLS::
    pop_stack() {
  std::shared_ptr<PyObject_SmithDispatchMode> out;
  if (!smithDispatchModeState.stack_.empty()) {
    out = smithDispatchModeState.stack_.back();
    smithDispatchModeState.stack_.pop_back();
  } else {
    for (int64_t i =
             static_cast<size_t>(SmithDispatchModeKey::NUM_MODE_KEYS) - 1;
         i >= 0;
         --i) {
      if (smithDispatchModeState.infra_modes_[i].has_value()) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        out = std::move(smithDispatchModeState.infra_modes_[i].value());
        smithDispatchModeState.infra_modes_[i] = std::nullopt;
        break;
      }
    }
  }
  SMITH_CHECK(out, "trying to pop from empty mode stack");
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return out;
}
const std::
    tuple<std::shared_ptr<PyObject_SmithDispatchMode>, SmithDispatchModeKey>
    SmithDispatchModeTLS::pop_highest_infra_mode() {
  for (int64_t i = static_cast<size_t>(SmithDispatchModeKey::NUM_MODE_KEYS) - 1;
       i >= 0;
       --i) {
    if (smithDispatchModeState.infra_modes_[i].has_value()) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      auto out_mode = smithDispatchModeState.infra_modes_[i].value();
      smithDispatchModeState.infra_modes_[i] = std::nullopt;
      if (!any_modes_set()) {
        c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
        c10::impl::tls_set_dispatch_key_included(
            DispatchKey::PythonTLSSnapshot, false);
      }
      return std::make_tuple(
          std::move(out_mode), static_cast<SmithDispatchModeKey>(i));
    }
  }
  SMITH_CHECK(
      false, "Called pop_highest_infra_mode, but no infra modes were active.")
}

const std::shared_ptr<PyObject_SmithDispatchMode>& SmithDispatchModeTLS::
    get_stack_at(int64_t idx) {
  SMITH_CHECK(idx < stack_len(), "Tried to get stack at idx that's too big");
  // Our "logical" stack includes both:
  // - any user modes (the entire smithDispatchModeState.stack_)
  // - any infra modes (members of smithDispatchModeState.infra_modes_ that are
  // not None)

  // idx == 0 means the "bottom" of the stack, which starts with any infra
  // modes (iterating from lowest-priority to highest-priority).
  auto curr_idx = idx;
  for (const auto i :
       c10::irange(static_cast<size_t>(SmithDispatchModeKey::NUM_MODE_KEYS))) {
    if (smithDispatchModeState.infra_modes_[i].has_value()) {
      if (curr_idx == 0) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        return smithDispatchModeState.infra_modes_[i].value();
      }
      curr_idx -= 1;
    }
  }
  // At this point, we're guaranteed that curr_idx < stack_.size()
  return smithDispatchModeState.stack_[curr_idx];
}

int64_t SmithDispatchModeTLS::stack_len() {
  auto stack_len = static_cast<int64_t>(smithDispatchModeState.stack_.size());
  int64_t infra_modes_len = 0;
  for (const auto i :
       c10::irange(static_cast<size_t>(SmithDispatchModeKey::NUM_MODE_KEYS))) {
    if (smithDispatchModeState.infra_modes_[i] != std::nullopt) {
      infra_modes_len += 1;
    }
  }
  return stack_len + infra_modes_len;
}

const std::optional<std::shared_ptr<PyObject_SmithDispatchMode>>
SmithDispatchModeTLS::get_mode(SmithDispatchModeKey mode_key) {
  return smithDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)];
}

void SmithDispatchModeTLS::set_mode(
    const std::shared_ptr<PyObject_SmithDispatchMode>& mode,
    SmithDispatchModeKey mode_key) {
  SMITH_CHECK(
      smithDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] ==
          std::nullopt,
      "trying to set the current ",
      to_string(mode_key),
      ", but one already exists");

  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }

  smithDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] = mode;
}

const std::optional<std::shared_ptr<PyObject_SmithDispatchMode>>
SmithDispatchModeTLS::unset_mode(SmithDispatchModeKey mode_key) {
  auto out = smithDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)];
  smithDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] =
      std::nullopt;
  if (out.has_value() && !any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return out;
}

const SmithDispatchModeTLS& SmithDispatchModeTLS::get_state() {
  return smithDispatchModeState;
}

void SmithDispatchModeTLS::set_state(SmithDispatchModeTLS state) {
  smithDispatchModeState = std::move(state);
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  } else {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
}

// UTIL

bool dispatch_mode_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python) &&
      SmithDispatchModeTLS::any_modes_set();
}

std::string to_string(SmithDispatchModeKey mode_key) {
  switch (mode_key) {
    case SmithDispatchModeKey::PROXY:
      return "ProxySmithDispatchMode";
    case SmithDispatchModeKey::FAKE:
      return "FakeTensorMode";
    default:
      return "UNKNOWN_MODE";
  }
}

} // namespace c10::impl
