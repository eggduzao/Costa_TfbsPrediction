#include <ATen/funcsmith/VmapInterpreter.h>
#include <ATen/funcsmith/DynamicLayer.h>

namespace at::funcsmith {

void VmapInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,
    smith::jit::Stack* stack) {
  setup_dispatch_key_tls(TransformType::Vmap, DispatchKeySet(DispatchKey::FuncSmithVmapMode));
  op.callBoxed(stack);
}

void VmapInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,
    smith::jit::Stack* stack,
    bool grad_special_case) {
  // Re-dispatch
  if (getDynamicLayerStack().empty()) {
    sanityCheckStack(op, stack);
  }
  op.callBoxed(stack);
}

} // namespace at::funcsmith
