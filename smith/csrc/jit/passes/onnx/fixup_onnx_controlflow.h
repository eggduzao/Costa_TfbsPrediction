#pragma once

#include <smith/csrc/jit/ir/ir.h>

namespace smith::jit {

std::vector<Value*> FixupONNXControlflowNode(Node* n, int opset_version);
void FixupONNXControlflowNodeOutputs(Node* n);

} // namespace smith::jit
