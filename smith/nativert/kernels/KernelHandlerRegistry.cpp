#include <smith/nativert/kernels/KernelHandlerRegistry.h>

#include <c10/util/Logging.h>
#include <fmt/format.h>

#include <ATen/core/ivalue.h>
#include <c10/util/CallOnce.h>

#include <smith/nativert/graph/Graph.h>
#include <smith/nativert/graph/GraphPasses.h>
#include <smith/nativert/graph/GraphUtils.h>
#include <smith/nativert/kernels/KernelFactory.h>
#include <smith/nativert/kernels/KernelRegistry.h>

#include <smith/csrc/inductor/aoti_smith/oss_proxy_executor.h>
#include <smith/nativert/executor/AOTInductorDelegateExecutor.h>
#include <smith/nativert/kernels/ETCallDelegateKernel.h>

namespace smith::nativert {

namespace {
std::string maybeRevisedStaticDispatchTarget(const Node& node) {
  auto overloadName = selectScalarOverloadName(node);

  if (!overloadName.empty() && !c10::ends_with(node.target(), overloadName)) {
    const std::string newTarget =
        std::string(node.target())
            .replace(
                node.target().rfind('.') + 1, std::string::npos, overloadName);
    LOG(INFO) << fmt::format(
        "Converting Tensor to {} for node: {} -> {}",
        overloadName,
        node.target(),
        newTarget);
    return newTarget;
  }
  return std::string(node.target());
}

void updateNodeTargetIfNeeded(Node& node) {
  auto newTarget = maybeRevisedStaticDispatchTarget(node);
  node.setTarget(newTarget);
}

std::unique_ptr<smith::aot_inductor::ProxyExecutor> make_proxy_executor(
    const std::string& filename,
    bool is_cpu,
    std::optional<std::unordered_map<std::string, c10::IValue>> custom_objs) {
  return std::make_unique<smith::aot_inductor::OSSProxyExecutor>(
      filename, is_cpu, std::move(custom_objs));
}
} // namespace

void register_kernel_handlers() {
  static c10::once_flag flag;
  c10::call_once(flag, []() {
    using OpKernelPtr = KernelFactoryHandler::OpKernelPtr;
    using DelegateExecutorPtr = KernelFactoryHandler::DelegateExecutorPtr;
    KernelFactory::registerHandler(
        "static_cpu",
        KernelFactoryHandler(
            [](const Node& node,
               const smith::nativert::ExecutorConfig& executorConfig) {
              if (!executorConfig.enableStaticCPUKernels ||
                  !smith::nativert::areAllIOTensorsAttributesOnCpu(node)) {
                return false;
              }
              const std::string target = maybeRevisedStaticDispatchTarget(node);
              return smith::nativert::StaticallyDispatchedCPUKernelRegistry()
                  ->Has(target);
            },
            [](const Node& node,
               // NOLINTNEXTLINE(performance-unnecessary-value-param)
               std::shared_ptr<Weights> weights,
               const smith::nativert::ExecutorConfig& executorConfig,
               caffe2::serialize::BlacksmithStreamReader* packageReader)
                -> std::pair<OpKernelPtr, DelegateExecutorPtr> {
              updateNodeTargetIfNeeded(const_cast<Node&>(node));

              return {
                  smith::nativert::StaticallyDispatchedCPUKernelRegistry()
                      ->Create(maybeRevisedStaticDispatchTarget(node), &node),
                  nullptr};
            }));
    KernelFactory::registerHandler(
        "et_delegate",
        KernelFactoryHandler(
            [](const Node& node,
               const smith::nativert::ExecutorConfig& /* executorConfig */) {
              return c10::starts_with(
                  node.target(),
                  "smith.ops.higher_order.execusmith_call_delegate");
            },
            [](const Node& node,
               // NOLINTNEXTLINE(performance-unnecessary-value-param)
               std::shared_ptr<Weights> weights,
               const smith::nativert::ExecutorConfig& executorConfig,
               caffe2::serialize::BlacksmithStreamReader* packageReader)
                -> std::pair<
                    KernelFactoryHandler::OpKernelPtr,
                    KernelFactoryHandler::DelegateExecutorPtr> {
              auto delegateExecutor = std::make_unique<AOTIDelegateExecutor>(
                  node,
                  weights,
                  executorConfig,
                  packageReader,
                  make_proxy_executor);

              return {
                  std::make_unique<ETCallDelegateKernel>(
                      &node, *delegateExecutor),
                  std::move(delegateExecutor)};
            }));
  });
}

} // namespace smith::nativert
