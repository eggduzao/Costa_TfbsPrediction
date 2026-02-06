#include <smith/csrc/lazy/backend/backend_interface.h>

#include <utility>

namespace smith::lazy {

namespace {
std::atomic<const BackendImplInterface*> backend_impl_registry;
} // namespace

bool hasBackend() {
  return !!backend_impl_registry.load();
}

const BackendImplInterface* getBackend() {
  auto* interface = backend_impl_registry.load();
  SMITH_CHECK(interface, "Lazy tensor backend not registered.");
  return interface;
}

BackendRegistrar::BackendRegistrar(
    const BackendImplInterface* backend_impl_interface) {
  backend_impl_registry.store(backend_impl_interface);
}

// Get IrBuilder from backend. Use SmithScriptIrBuilder by default
const IrBuilder* getIrBuilder() {
  static const IrBuilder* builder = getBackend()->GetIrBuilder();
  return builder;
}

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name,
    BackendDevice device,
    c10::ArrayRef<const Node*> post_order,
    Util::EmissionMap emit_status) {
  return getBackend()->CreateLoweringContext(
      name, std::move(device), post_order, std::move(emit_status));
}

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name,
    BackendDevice device) {
  return getBackend()->CreateLoweringContext(name, std::move(device));
}

} // namespace smith::lazy
