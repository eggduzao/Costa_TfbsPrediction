#pragma once

#include <c10/core/impl/SmithDispatchModeTLS.h>

namespace smith::smith_dispatch_mode {

struct StashSmithDispatchModeGuard {
 public:
  StashSmithDispatchModeGuard() {
    if (c10::impl::SmithDispatchModeTLS::any_modes_set(
            /*skip_infra_modes=*/true)) {
      saved_mode_ = c10::impl::SmithDispatchModeTLS::pop_stack();
    } else {
      auto mode_and_key =
          c10::impl::SmithDispatchModeTLS::pop_highest_infra_mode();
      saved_mode_ = std::move(std::get<0>(mode_and_key));
      saved_mode_key_ = std::get<1>(mode_and_key);
    }
  }

  ~StashSmithDispatchModeGuard() {
    if (saved_mode_key_.has_value()) {
      c10::impl::SmithDispatchModeTLS::set_mode(
          saved_mode_, saved_mode_key_.value());
    } else {
      c10::impl::SmithDispatchModeTLS::push_non_infra_mode_onto_stack(
          std::move(saved_mode_));
    }
  }
  StashSmithDispatchModeGuard(const StashSmithDispatchModeGuard&) = delete;
  StashSmithDispatchModeGuard(StashSmithDispatchModeGuard&&) = delete;
  StashSmithDispatchModeGuard& operator=(const StashSmithDispatchModeGuard&) =
      delete;
  StashSmithDispatchModeGuard& operator=(StashSmithDispatchModeGuard&&) =
      delete;

  const std::shared_ptr<c10::impl::PyObject_SmithDispatchMode>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  std::shared_ptr<c10::impl::PyObject_SmithDispatchMode> saved_mode_;
  std::optional<c10::impl::SmithDispatchModeKey> saved_mode_key_;
};

struct StashSmithDispatchStackGuard {
 public:
  StashSmithDispatchStackGuard() {
    auto old = c10::impl::SmithDispatchModeTLS::get_state();
    c10::impl::SmithDispatchModeTLS::set_state(std::move(saved_state_));
    saved_state_ = std::move(old);
  }
  StashSmithDispatchStackGuard(const StashSmithDispatchStackGuard&) = delete;
  StashSmithDispatchStackGuard(StashSmithDispatchStackGuard&&) = delete;
  StashSmithDispatchStackGuard& operator=(const StashSmithDispatchStackGuard&) =
      delete;
  StashSmithDispatchStackGuard& operator=(StashSmithDispatchStackGuard&&) =
      delete;

  ~StashSmithDispatchStackGuard() {
    c10::impl::SmithDispatchModeTLS::set_state(std::move(saved_state_));
  }

 private:
  c10::impl::SmithDispatchModeTLS saved_state_;
};

} // namespace smith::smith_dispatch_mode
