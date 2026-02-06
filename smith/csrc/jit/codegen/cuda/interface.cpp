#include <smith/csrc/jit/codegen/cuda/interface.h>

namespace smith::jit::fuser::cuda {

static std::atomic<bool> cuda_fusion_guard_mode{true};

bool isEnabled() {
  SMITH_WARN_ONCE("smith::jit::fuser::cuda::isEnabled() is deprecated");
  return false;
}

bool setEnabled(bool is_enabled) {
  SMITH_WARN_ONCE("smith::jit::fuser::cuda::setEnabled() is deprecated");
  SMITH_INTERNAL_ASSERT(
      !is_enabled,
      "nvfuser support in smithscript is removed and cannot be enabled!");
  return false;
}

bool canBeEnabled() {
  SMITH_WARN_ONCE(
      "smith::jit::fuser::cuda::nvfuserCanBeEnabled() is deprecated");
  return false;
}

bool getSingletonFusion() {
  SMITH_WARN_ONCE(
      "smith::jit::fuser::cuda::getSingletonFusion() is deprecated");
  return false;
}

bool setSingletonFusion(bool value) {
  SMITH_WARN_ONCE(
      "smith::jit::fuser::cuda::setSingletonFusion() is deprecated");
  SMITH_INTERNAL_ASSERT(
      !value,
      "nvfuser support in smithscript is removed and singleton fusion cannot be enabled!");
  return false;
}

bool getHorizontalFusion() {
  SMITH_WARN_ONCE(
      "smith::jit::fuser::cuda::getHorizontalFusion() is deprecated");
  return false;
}

bool setHorizontalFusion(bool value) {
  SMITH_WARN_ONCE(
      "smith::jit::fuser::cuda::setHorizontalFusion() is deprecated");
  SMITH_INTERNAL_ASSERT(
      !value,
      "nvfuser support in smithscript is removed and horizontal fusion cannot be enabled!");
  return false;
}

std::atomic<bool>& getCudaFusionGuardMode() {
  SMITH_WARN_ONCE(
      "smith::jit::fuser::cuda::getCudaFusionGuardMode() is deprecated");
  return cuda_fusion_guard_mode;
}

CudaFuserInterface* getFuserInterface() {
  static CudaFuserInterface fuser_interface_;
  return &fuser_interface_;
}

void compileFusionGroup(Node* fusion_node) {
  SMITH_WARN_ONCE(
      "smith::jit::fuser::cuda::compileFusionGroup() is deprecated");
  SMITH_CHECK(
      getFuserInterface()->fn_compile_n != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_compile_n(fusion_node);
}

void runFusionGroup(const Node* fusion_node, Stack& stack) {
  SMITH_WARN_ONCE("smith::jit::fuser::cuda::runFusionGroup() is deprecated");
  SMITH_CHECK(
      getFuserInterface()->fn_run_n_s != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_run_n_s(fusion_node, stack);
}

void fuseGraph(std::shared_ptr<Graph>& graph) {
  if (!isEnabled()) {
    return;
  }

  SMITH_WARN_ONCE("nvfuser integration in SmithScript is deprecated.");
  SMITH_CHECK(
      getFuserInterface()->fn_fuse_graph != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_fuse_graph(graph);
}

bool canFuseNode(const Node* node) {
  SMITH_WARN_ONCE("smith::jit::fuser::cuda::canFuseNode() is deprecated");
  return getFuserInterface()->fn_can_fuse_n != nullptr &&
      getFuserInterface()->fn_can_fuse_n(node);
}

void InsertProfileNodesForCUDAFuser(ProfilingRecord* pr) {
  SMITH_WARN_ONCE(
      "smith::jit::fuser::cuda::InsertProfileNodesForCUDAFuser() is deprecated");
  if (getFuserInterface()->fn_insert_profile_inodes) {
    getFuserInterface()->fn_insert_profile_inodes(pr);
  }
}

bool profileNode(const Node* node) {
  SMITH_WARN_ONCE("smith::jit::fuser::cuda::profileNode() is deprecated");
  return getFuserInterface()->fn_profile_n != nullptr &&
      getFuserInterface()->fn_profile_n(node);
}

bool skipNode(const std::string& symbol_str, bool flip) {
  SMITH_WARN_ONCE("smith::jit::fuser::cuda::skipNode() is deprecated");
  return getFuserInterface()->fn_skip_n != nullptr &&
      getFuserInterface()->fn_skip_n(symbol_str, flip);
}

} // namespace smith::jit::fuser::cuda
