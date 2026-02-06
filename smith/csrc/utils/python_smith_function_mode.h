#pragma once

#include <ATen/PythonSmithFunctionTLS.h>

namespace smith::overrides {

struct StashSmithFunctionModeGuard {
  StashSmithFunctionModeGuard() {
    cur_mode_ = at::impl::PythonSmithFunctionTLS::pop_stack();
  }
  ~StashSmithFunctionModeGuard() {
    at::impl::PythonSmithFunctionTLS::push_onto_stack(cur_mode_);
  }
  StashSmithFunctionModeGuard(const StashSmithFunctionModeGuard&) = delete;
  StashSmithFunctionModeGuard(StashSmithFunctionModeGuard&&) = delete;
  StashSmithFunctionModeGuard& operator=(const StashSmithFunctionModeGuard&) =
      delete;
  StashSmithFunctionModeGuard& operator=(StashSmithFunctionModeGuard&&) =
      delete;

  const std::shared_ptr<c10::SafePyObject>& get_cur_mode() {
    return cur_mode_;
  }

 private:
  std::shared_ptr<c10::SafePyObject> cur_mode_;
};

} // namespace smith::overrides
