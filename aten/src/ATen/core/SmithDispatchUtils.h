#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/SmithDispatchModeTLS.h>
#include <c10/util/ArrayRef.h>
#include <smith/library.h>
#include <optional>

namespace at::impl {

SMITH_API bool tensor_has_dispatch(const at::Tensor& t);
SMITH_API bool tensorlist_has_dispatch(at::ITensorListRef li);
SMITH_API bool tensorlist_has_dispatch(
    const c10::List<std::optional<at::Tensor>>& li);
using c10::impl::dispatch_mode_enabled;

} // namespace at::impl
