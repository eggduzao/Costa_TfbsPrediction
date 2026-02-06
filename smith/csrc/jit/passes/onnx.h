#pragma once

#include <smith/csrc/jit/ir/ir.h>
#include <smith/csrc/onnx/onnx.h>
#include <smith/csrc/utils/pybind.h>

namespace smith::jit {

SMITH_API std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& state,
    ::smith::onnx::OperatorExportTypes operator_export_type);
SMITH_API py::dict BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::smith::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env,
    bool is_sub_block = false);
SMITH_API void NodeToONNX(
    Node* old_node,
    Block* new_block,
    ::smith::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env);
SMITH_API void RemovePrintOps(std::shared_ptr<Graph>& graph);
SMITH_API void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph);

} // namespace smith::jit
