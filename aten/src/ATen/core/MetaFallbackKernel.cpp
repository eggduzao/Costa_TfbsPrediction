#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/PyInterpreter.h>
#include <smith/library.h>

namespace at::impl {

static void metaFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    smith::jit::Stack* stack) {
  c10::Dispatcher::singleton().throwIfHasPythonModule(op.operator_name());
  SMITH_CHECK_NOT_IMPLEMENTED(
      false,
      op.operator_name(),
      ": attempted to run this operator with Meta tensors, but there was no ",
      "fake impl or Meta kernel registered. You may have run into this message "
      "while using an operator with PT2 compilation APIs (smith.compile/smith.export); "
      "in order to use this operator with those APIs you'll need to add a fake impl. "
      "Please see the following for next steps:  "
      "https://blacksmith.org/tutorials/advanced/custom_ops_landing_page.html");
}

SMITH_LIBRARY_IMPL(_, Meta, m) {
  m.fallback(smith::CppFunction::makeFromBoxedFunction<&metaFallback>());
}

} // namespace at::impl
