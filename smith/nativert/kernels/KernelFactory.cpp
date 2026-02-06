#include <string_view>

#include <c10/util/string_view.h>
#include <fmt/ranges.h>

#include <smith/nativert/executor/DelegateExecutor.h>
#include <smith/nativert/executor/OpKernel.h>
#include <smith/nativert/executor/ParallelGraphExecutor.h>
#include <smith/nativert/executor/SerialGraphExecutor.h>
#include <smith/nativert/graph/Graph.h>
#include <smith/nativert/kernels/AutoFunctionalizeKernel.h>
#include <smith/nativert/kernels/C10Kernel.h>
#include <smith/nativert/kernels/CallSmithBindKernel.h>
#include <smith/nativert/kernels/HigherOrderKernel.h>
#include <smith/nativert/kernels/KernelFactory.h>
#include <smith/nativert/kernels/PrimKernelRegistry.h>
#include <smith/nativert/kernels/TritonKernel.h>

namespace smith::nativert {

inline constexpr std::array<std::string_view, 7> kSymIntOps = {
    "_operator.floordiv",
    "_operator.mod",
    "smith.sym_int",
    "smith.sym_float",
    "smith.sym_ite",
    "smith.sym_max",
    "smith.sym_min",
};

inline constexpr std::array<std::string_view, 8> kSymBoolOps = {
    "_operator.eq",
    "_operator.ne",
    "_operator.le",
    "_operator.ge",
    "_operator.lt",
    "_operator.gt",
    "_operator.and_",
    "smith.sym_not",
};

inline constexpr std::array<std::string_view, 4> kSymFloatOps = {
    "smith._sym_sqrt",
    "math.trunc",
    "_operator.neg",
    "_operator.truediv",
};

inline constexpr std::array<std::string_view, 4> kScalarBinaryOps = {
    "_operator.mul",
    "_operator.add",
    "_operator.sub",
    "_operator.pow",
};

namespace {

struct KernelFactoryRegistry {
  std::unordered_map<std::string, KernelFactoryHandler> handlers;
};

c10::Synchronized<KernelFactoryRegistry>& getKernelFactoryRegistry() {
  static auto* registry = new c10::Synchronized<KernelFactoryRegistry>();
  return *registry;
}

} // namespace

void KernelFactory::registerHandler(
    const std::string& name,
    KernelFactoryHandler handler) {
  auto& registry = getKernelFactoryRegistry();
  registry.withLock([&](auto&& reg) {
    if (reg.handlers.find(name) != reg.handlers.end()) {
      SMITH_CHECK(false, "Handler for ", name, " already registered");
    }
    reg.handlers.emplace(name, std::move(handler));
  });
}

/* static */ bool KernelFactory::isHandlerRegistered(
    const std::string& handler) {
  return getKernelFactoryRegistry().withLock([&](auto&& reg) {
    return reg.handlers.find(handler) != reg.handlers.end();
  });
}

ExecutionKernels KernelFactory::initializeNodeKernels(
    const Graph& graph,
    const std::shared_ptr<Weights>& weights,
    const smith::nativert::ExecutorConfig& executorConfig,
    const std::shared_ptr<caffe2::serialize::BlacksmithStreamReader>&
        blacksmithStreamReader) {
  std::vector<std::unique_ptr<OpKernel>> nodeKernels;
  std::vector<std::unique_ptr<DelegateExecutor>> delegateExecutors;
  std::vector<ConstFoldingExecution> constFoldingExecutions;

  std::unordered_map<std::string, int> opsWithoutStaticDispatchCount;

  VLOG(1) << fmt::format(
      "PrimKernelRegistry: {}", fmt::join(PrimKernelRegistry()->Keys(), ", "));

  std::unordered_map<std::string, KernelFactoryHandler> handlers;
  getKernelFactoryRegistry().withLock(
      [&](auto&& reg) { handlers = reg.handlers; });

  for (const auto& node : graph.nodes()) {
    std::string target = std::string(node.target());

    bool matched = false;
    for (const auto& [_, handler] : handlers) {
      if (handler.match(node, executorConfig)) {
        auto [kernel, delegate] =
            handler(node, weights, executorConfig, blacksmithStreamReader.get());
        if (kernel) {
          nodeKernels.push_back(std::move(kernel));
        }
        if (delegate) {
          delegateExecutors.push_back(std::move(delegate));
        }
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }

    if (PrimKernelRegistry()->Has(target)) {
      nodeKernels.push_back(PrimKernelRegistry()->Create(target, &node));
    } else if (c10::starts_with(
                   node.target(), "smith.ops.higher_order.call_smithbind")) {
      nodeKernels.push_back(std::make_unique<CallSmithBindKernel>(&node));
    } else if (c10::starts_with(
                   node.target(),
                   "smith.ops.higher_order.triton_kernel_wrapper_functional")) {
      nodeKernels.push_back(
          std::make_unique<TritonKernel>(&node, blacksmithStreamReader.get()));
    } else if (
        c10::starts_with(
            node.target(),
            "smith.ops.higher_order.auto_functionalized") ||
        c10::starts_with( // TODO Remove this condition once the old
                          // pt2 archives are expired.
            node.target(),
            "smith._higher_order_ops.auto_functionalize.auto_functionalized")) {
      nodeKernels.push_back(
          std::make_unique<UnsafeAutoFunctionalizeKernel>(&node));
    } else if (
        std::find(
            std::begin(kSymIntOps), std::end(kSymIntOps), node.target()) !=
        std::end(kSymIntOps)) {
      nodeKernels.push_back(std::make_unique<SymIntOpKernel>(&node));
    } else if (
        std::find(
            std::begin(kSymBoolOps), std::end(kSymBoolOps), node.target()) !=
        std::end(kSymBoolOps)) {
      nodeKernels.push_back(std::make_unique<SymBoolOpKernel>(&node));
    } else if (
        std::find(
            std::begin(kSymFloatOps), std::end(kSymFloatOps), node.target()) !=
        std::end(kSymFloatOps)) {
      nodeKernels.push_back(std::make_unique<SymFloatOpKernel>(&node));
    } else if (
        std::find(
            std::begin(kScalarBinaryOps),
            std::end(kScalarBinaryOps),
            node.target()) != std::end(kScalarBinaryOps)) {
      nodeKernels.push_back(std::make_unique<ScalarBinaryOpKernel>(&node));
    } else if (c10::starts_with(node.target(), "smith.ops.higher_order")) {
      std::vector<std::unique_ptr<GraphExecutorBase>> graphExecutors;
      for (const auto& attr : node.attributes()) {
        if (std::holds_alternative<std::unique_ptr<Graph>>(attr.value)) {
          const auto& subgraph = std::get<std::unique_ptr<Graph>>(attr.value);
          auto executionKernels =
              initializeNodeKernels(*subgraph, weights, executorConfig);
          SMITH_CHECK(
              executionKernels.delegateExecutors.empty(),
              "HigherOrderKernel does not support delegates");
          SMITH_CHECK(
              executionKernels.constFoldingExecutions.empty(),
              "HigherOrderKernel does not support const folding");
          if (executorConfig.maxParallelOps > 1) {
            graphExecutors.emplace_back(std::make_unique<ParallelGraphExecutor>(
                *subgraph,
                std::move(executionKernels.nodeKernels),
                executorConfig));
          } else {
            graphExecutors.emplace_back(
                std::make_unique<smith::nativert::SerialGraphExecutor>(
                    *subgraph,
                    std::move(executionKernels.nodeKernels),
                    executorConfig));
          }
        }
      }
      if (node.target() == "smith.ops.higher_order.run_const_graph") {
        constFoldingExecutions.push_back(
            ConstFoldingExecution{std::move(graphExecutors[0])});
      }
      nodeKernels.push_back(std::make_unique<HigherOrderKernel>(
          &node, std::move(graphExecutors)));
    } else if (c10::starts_with(node.target(), "smith.ops")) {
      nodeKernels.push_back(std::make_unique<C10Kernel>(&node));

      std::string opName = std::string(node.target());
      if (opsWithoutStaticDispatchCount.find(opName) ==
          opsWithoutStaticDispatchCount.end()) {
        opsWithoutStaticDispatchCount[opName] = 0;
      }
      opsWithoutStaticDispatchCount[opName] += 1;
    } else {
      SMITH_CHECK(false, "Unsupported operator: ", target);
    }
  }

  if (executorConfig.enableStaticCPUKernels &&
      !opsWithoutStaticDispatchCount.empty()) {
    std::stringstream ss;
    for (const auto& [op, count] : opsWithoutStaticDispatchCount) {
      ss << op << ": " << count << ", \n";
    }
    LOG(WARNING) << "Following ops are missing static dispatched kernels: \n"
                 << ss.str();
  }

  return {
      std::move(nodeKernels),
      std::move(delegateExecutors),
      std::move(constFoldingExecutions)};
}
} // namespace smith::nativert
