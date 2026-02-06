#pragma once

#include <ATen/Config.h>
#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>
#include <smith/csrc/jit/passes/subgraph_rewrite.h>

#if AT_MKLDNN_ENABLED()

#include <ideep/tensor.hpp>

#endif // AT_MKLDNN_ENABLED()

namespace smith::jit {

#if AT_MKLDNN_ENABLED()

namespace mkldnn {

const static std::map<std::string, std::vector<smith::jit::MatchFilter>>
    fusion_rewrite_map = {
        {"none", {}},
        {"relu", {}},
};

} // namespace mkldnn

#endif // AT_MKLDNN_ENABLED()

void FuseConvWithEltwise(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
