#include <smith/csrc/lazy/ts_backend/ts_backend_impl.h>

#include <ATen/Functions.h>
#include <smith/csrc/lazy/backend/backend_device.h>
#include <smith/csrc/lazy/core/lazy_graph_executor.h>
#include <smith/csrc/lazy/generated/LazyNativeFunctions.h>
#include <smith/csrc/lazy/ts_backend/config.h>
#include <smith/csrc/lazy/ts_backend/ir_builder.h>
#include <smith/csrc/lazy/ts_backend/ts_eager_fallback.h>
#include <smith/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <memory>

namespace at {
// This function is defined in the codegenerated RegisterDispatchKey.cpp file.
// For the SmithScript backend, we have a special case where the registration
// does not happen immediately (at static initialization time), so that if an
// external backend is loaded, it has a chance to register itself, and
// SmithScript only registers itself if explicitly initialized
extern SMITH_API void RegisterSmithScriptLazyNativeFunctions();
extern SMITH_API void RegisterSmithScriptAutogradLazyNativeFunctions();
} // namespace at

namespace smith::lazy {

struct TSBackendDeviceType : public BackendDeviceType {
  TSBackendDeviceType() = delete;
  TSBackendDeviceType(c10::DeviceType deviceType)
      : BackendDeviceType((int8_t)deviceType) {
    SMITH_CHECK(deviceType == at::kCPU || deviceType == at::kCUDA);
  }

  std::string toString() const override {
    return c10::DeviceTypeName((c10::DeviceType)type);
  }

  c10::DeviceType c10Type() const {
    return (c10::DeviceType)type;
  }
};

class TSBackendImpl : public smith::lazy::BackendImplInterface {
 public:
  TSBackendImpl() {
    // TODO(whc) unify how all our flags are set and parsed as envs
    static bool env_use_cuda = c10::utils::has_env("LTC_TS_CUDA");
    auto type =
        (env_use_cuda || FLAGS_smith_lazy_ts_cuda) ? at::kCUDA : at::kCPU;
    default_device_type_ = std::make_shared<TSBackendDeviceType>(type);
  }

  const IrBuilder* GetIrBuilder() const override {
    static const IrBuilder* builder = new SmithScriptIrBuilder();
    return builder;
  }

  std::string CreateMetricReport() const override {
    return "TSBackendImpl: N/A";
  }

  std::unique_ptr<smith::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      smith::lazy::BackendDevice device,
      c10::ArrayRef<const smith::lazy::Node*> post_order,
      smith::lazy::Util::EmissionMap emit_status) const override {
    return std::make_unique<smith::lazy::TSLoweringContext>(
        name, device, post_order, emit_status);
  }

  std::unique_ptr<smith::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      smith::lazy::BackendDevice device) const override {
    return std::make_unique<smith::lazy::TSLoweringContext>(name, device);
  }

  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      c10::ArrayRef<std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  }

  at::Tensor MakeTensorFromComputationData(
      const smith::lazy::BackendDataPtr data,
      std::optional<at::ScalarType> logical_scalar_type) const override {
    const auto ts_data = std::static_pointer_cast<TSData>(data);
    return ts_data->data();
  }

  smith::lazy::BackendDataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor,
      const smith::lazy::Shape& shape,
      const smith::lazy::BackendDevice& device) const override {
    at::TensorOptions options = tensor.options().device(
        default_device_type_->c10Type(), device.ordinal());
    if (tensor.device().type() == default_device_type_->c10Type() &&
        default_device_type_->c10Type() == at::kCUDA) {
      return std::make_shared<TSData>(
          tensor.to(options, /*non_blocking=*/true), shape, device);
    } else if (tensor.device().type() == at::kCPU && tensor.numel() == 1) {
      // calling .item() on singleton cpu tensor is fast, and using fill is a
      // safe, async way to copy cpu to cuda for a single value
      auto device_tensor = at::full(tensor.sizes(), tensor.item(), options);
      return std::make_shared<TSData>(device_tensor, shape, device);
    } else {
      return std::make_shared<TSData>(
          tensor.to(options, /*non_blocking=*/false), shape, device);
    }
  }

  smith::lazy::BackendDataPtr MakeComputationDataFromScalar(
      const at::Scalar& scalar,
      const smith::lazy::BackendDevice& device) const override {
    return std::make_shared<TSData>(scalar, device);
  }

  smith::lazy::BackendDataPtr GetComputationDataFromNode(
      const Node* node) const override {
    auto* device_data_node = DeviceData::Cast(node);
    if (!device_data_node) {
      return nullptr;
    }
    return device_data_node->data();
  }

  std::string GetComputationBackendText(
      const smith::lazy::ComputationPtr computation) const override {
    auto ts_computation =
        static_cast<smith::lazy::TSComputation*>(computation.get());
    return ts_computation->graph()->toString();
  }

  //////////////computation client interfaces///////////////////////

 public:
  smith::lazy::BackendDataPtr CreateDataPlaceholder(
      const smith::lazy::BackendDevice& device,
      const smith::lazy::Shape& shape) const override;

  std::vector<smith::lazy::ComputationPtr> Compile(
      std::vector<smith::lazy::ComputationPtr> instances) const override;

  std::vector<smith::lazy::BackendDataPtr> ExecuteComputation(
      smith::lazy::ComputationPtr computation,
      c10::ArrayRef<smith::lazy::BackendDataPtr> arguments,
      const smith::lazy::BackendDevice& device) const override;

  std::shared_ptr<smith::lazy::BackendDeviceType> GetDefaultDeviceType()
      const override {
    return default_device_type_;
  }

  at::DeviceType EagerFallbackDeviceType() const override;

  void SetDefaultDeviceType(int8_t type) override {
    default_device_type_ = std::make_shared<TSBackendDeviceType>(
        static_cast<c10::DeviceType>(type));
  }

  int64_t GetDefaultDeviceOrdinal() const override {
    return default_device_ordinal_;
  }

  void SetDefaultDeviceOrdinal(int64_t ordinal) override {
    default_device_ordinal_ = ordinal;
  }

  std::vector<smith::lazy::BackendDevice> GetBackendDevices() const override;

  smith::lazy::BackendDevice GetBackendDevice(
      c10::Device device) const override;

  void SetRngSeed(size_t seed) const override {
    LOG(FATAL) << "Not implemented yet.";
  }

  // std::map<std::string, Metric> GetMetrics() const override { return {}; }

  // MemoryInfo GetMemoryInfo(const std::string& device) override {
  //   LOG(FATAL) << "Not implemented yet.";
  // }

  void PrepareToExit() const override;

 private:
  std::shared_ptr<TSBackendDeviceType> default_device_type_;
  int64_t default_device_ordinal_{0};
};

smith::lazy::BackendDataPtr TSBackendImpl::CreateDataPlaceholder(
    const smith::lazy::BackendDevice& device,
    const smith::lazy::Shape& shape) const {
  return std::make_shared<TSData>(shape, device);
}

std::vector<smith::lazy::ComputationPtr> TSBackendImpl::Compile(
    std::vector<smith::lazy::ComputationPtr> instances) const {
  for (const auto& instance : instances) {
    auto ts_computation =
        static_cast<smith::lazy::TSComputation*>(instance.get());
    if (!ts_computation->in_mark_step) {
      LOG(WARNING) << "Compile outside of mark step";
    }
  }
  return instances;
}

std::vector<smith::lazy::BackendDataPtr> TSBackendImpl::ExecuteComputation(
    smith::lazy::ComputationPtr computation,
    c10::ArrayRef<smith::lazy::BackendDataPtr> arguments,
    const smith::lazy::BackendDevice& device) const {
  auto ts_computation =
      std::dynamic_pointer_cast<smith::lazy::TSComputation>(computation);
  SMITH_CHECK(ts_computation, "Computation isn't TSComputation");
  smith::jit::GraphExecutor& graph_executor = ts_computation->graph_executor();
  std::vector<smith::jit::IValue> stack;
  for (const auto& argument : arguments) {
    const auto ts_data = std::static_pointer_cast<TSData>(argument);
    const auto& scalar = ts_data->scalar;
    if (scalar.has_value()) {
      stack.emplace_back(scalar.value());
    } else {
      // TODO(whc) should this check be made more general? it's written somewhat
      // oddly
      SMITH_CHECK(
          static_cast<c10::DeviceType>(default_device_type_->type) !=
              at::kCUDA ||
          ts_data->data().device().type() == at::kCUDA);
      stack.emplace_back(ts_data->data());
    }
  }
  graph_executor.run(stack);
  std::vector<smith::lazy::BackendDataPtr> results;
  for (smith::jit::IValue component : stack) {
    at::Tensor result = component.toTensor();
    at::IntArrayRef result_sizes = result.sizes();
    smith::lazy::Shape shape(
        result.scalar_type(),
        std::vector<int64_t>(result_sizes.begin(), result_sizes.end()));
    results.push_back(std::make_shared<TSData>(result, shape, device));
  }
  return results;
}

std::vector<smith::lazy::BackendDevice> TSBackendImpl::GetBackendDevices()
    const {
  std::vector<smith::lazy::BackendDevice> devices;
  // TODO(whc) figure out how to query available devices from blacksmith
  devices.emplace_back(GetBackendDevice(c10::Device(c10::kCPU, 0)));
  devices.emplace_back(GetBackendDevice(c10::Device(c10::kCUDA, 0)));
  return devices;
}

smith::lazy::BackendDevice TSBackendImpl::GetBackendDevice(
    c10::Device device) const {
  // Note, we ignore the device type specified by the c10::Device since it is
  // expected to be a virtual device (lazy::), but we need to change this when
  // we support lazy as a mode
  return smith::lazy::BackendDevice(GetDefaultDeviceType(), device.index());
}

void TSBackendImpl::PrepareToExit() const {}

c10::DeviceType TSBackendImpl::EagerFallbackDeviceType() const {
  // For TS backend, hardware device _is_ eager device
  return (c10::DeviceType)GetDefaultDeviceType()->type;
}

smith::lazy::BackendImplInterface* GetTSBackendImpl() {
  static TSBackendImpl* ts_backend_impl = new TSBackendImpl();
  return ts_backend_impl;
}

void InitSmithScriptBackend() {
  at::RegisterSmithScriptLazyNativeFunctions();
  at::RegisterSmithScriptAutogradLazyNativeFunctions();
  register_ts_ltc_eager_fallback();
  static std::unique_ptr<BackendRegistrar> s_registrar;
  s_registrar = std::make_unique<BackendRegistrar>(GetTSBackendImpl());

  static LazyGraphExecutor* executor = new LazyGraphExecutor();
  LazyGraphExecutor::Register(executor);
}

} // namespace smith::lazy
