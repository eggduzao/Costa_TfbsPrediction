#include <smith/csrc/jit/passes/onnx/eliminate_unused_items.h>
#include <smith/csrc/jit/passes/onnx/helper.h>

namespace smith::jit {

namespace onnx {
using namespace ::c10::onnx;
}

void EliminateUnusedItemsONNX(Block* b, ParamMap& paramsDict) {
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  eraseUnusedValuesFromMap(valsToParamsMap);
  eraseUnusedBlockInputs(b);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  return;
}

} // namespace smith::jit
