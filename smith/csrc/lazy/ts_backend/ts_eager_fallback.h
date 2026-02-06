#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <functional>

namespace smith::lazy {

bool force_eager_fallback(c10::Symbol op);
void ltc_eager_fallback(
    const c10::OperatorHandle& op,
    smith::jit::Stack* stack);

void ts_eager_fallback(
    const c10::OperatorHandle& op,
    smith::jit::Stack* stack,
    c10::DeviceType device_type);

// The SmithScript backend does not register itself with blacksmith dispatcher
// until it is explicitly initialized.  This function should only be called
// by the main Smithscript backend init function.
void register_ts_ltc_eager_fallback();

} // namespace smith::lazy
