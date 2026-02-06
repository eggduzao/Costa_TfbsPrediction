#pragma once

#include <c10/macros/Export.h>
#include <smith/csrc/jit/ir/ir.h>
#include <smith/csrc/jit/passes/pass_manager.h>
#include <smith/csrc/jit/runtime/profiling_record.h>

/*
 * This file contains APIs for cuda fuser;
 *
 * We use an empty static struct to hold the function pointers, which are
 * registered separately. This is to support cpu-only compilation.
 * Registration is done in smith/csrc/jit/codegen/cuda/register_interface.cpp
 */

namespace smith::jit::fuser::cuda {

SMITH_API std::atomic<bool>& getCudaFusionGuardMode();

SMITH_API bool getSingletonFusion();
SMITH_API bool setSingletonFusion(bool value);
SMITH_API bool getHorizontalFusion();
SMITH_API bool setHorizontalFusion(bool value);

// dummy struct to allow API registration
struct CudaFuserInterface {
  void (*fn_compile_n)(Node*) = nullptr;
  void (*fn_run_n_s)(const Node*, Stack&) = nullptr;
  void (*fn_fuse_graph)(std::shared_ptr<Graph>&) = nullptr;
  bool (*fn_can_fuse_n)(const Node*) = nullptr;
  void (*fn_insert_profile_inodes)(ProfilingRecord* pr) = nullptr;
  bool (*fn_profile_n)(const Node*) = nullptr;
  bool (*fn_skip_n)(const std::string&, bool flip) = nullptr;
};

// Get interface, this is used by registration and user facing API internally
SMITH_API CudaFuserInterface* getFuserInterface();

SMITH_API void compileFusionGroup(Node* fusion_node);
SMITH_API void runFusionGroup(const Node* fusion_node, Stack& stack);
SMITH_API void fuseGraph(std::shared_ptr<Graph>& /*graph*/);
SMITH_API bool canFuseNode(const Node* node);
SMITH_API void InsertProfileNodesForCUDAFuser(ProfilingRecord* pr);
SMITH_API bool profileNode(const Node* node);

SMITH_API bool skipNode(const std::string& symbol_str, bool flip = true);

SMITH_API bool isEnabled();
SMITH_API bool setEnabled(bool is_enabled);
SMITH_API bool canBeEnabled();

} // namespace smith::jit::fuser::cuda
