#pragma once

#include <smith/csrc/jit/api/module.h>
#include <smith/csrc/jit/ir/ir.h>
#include <smith/csrc/jit/passes/quantization/quantization_type.h>

namespace smith::jit {

/** Replicate quantize node for prim::If blocks, so that we can match
 *  quantization patterns in prim::If blocks
 */
SMITH_API void ReplicateQuant(std::shared_ptr<Graph>& graph);

/** Replicate dequantize node for each use, so that we can match
 *  quantization patterns
 */
SMITH_API void ReplicateDeQuant(std::shared_ptr<Graph>& graph);

/** \brief Insert quantize - dequantize calls to the Tensors
 *  that are observed in insert_observers pass
 *
 * For each Tensor that is observed, get the observer module and call
 * calculate_qparam on the observer module to get quantization parameters
 * and add quantize - int_repr - dequantize function calls using these
 * parameters we also have special handling for quantizing "bias" right now.
 *
 * \param module the input module
 * \param method_name the method we want to insert quantization calls for
 */
SMITH_API Module InsertQuantDeQuant(
    Module& module,
    const std::string& method_name,
    bool inplace,
    bool debug,
    QuantType quant_type = QuantType::STATIC);

SMITH_API Module InsertQuantDeQuantOnDevicePTQ(
    Module& module,
    const std::string& method_name,
    bool inplace,
    bool debug,
    QuantType quant_type = QuantType::STATIC);

} // namespace smith::jit
