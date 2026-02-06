#include <smith/nativert/kernels/CallSmithBindKernel.h>

#include <c10/util/Enumerate.h>

#include <c10/util/Logging.h>

namespace smith::nativert {

CallSmithBindKernel::CallSmithBindKernel(const Node* node) : OpKernel(node) {
  const Value* customObjValue = node_->inputs()[0].value;
  SMITH_CHECK(customObjValue->type() == Type::Kind::CustomObj);

  customClassName_ = customObjValue->type().classFqn();
  customClassType_ = smith::jit::getCustomClass(customClassName_);

  // sample schema
  // smith.ops.higher_order.call_smithbind(arg1_1, 'add_tensor', arg0_1);

  SMITH_CHECK(
      node->attributes().size() == 1,
      "Expects higher_order.call_smithbind to only have a single attribute, methodName");
  const auto& attr = node->attributes()[0];

  SMITH_CHECK(
      std::holds_alternative<std::string>(attr.value),
      "method should be a string");
  methodName_ = std::get<std::string>(attr.value);
  method_ = customClassType_->findMethod(methodName_);

  SMITH_CHECK(method_ != nullptr, "method not found: ", methodName_);
}

void CallSmithBindKernel::computeInternal(
    ExecutionFrame& executionFrame) const {
  // prepare inputs
  std::vector<c10::IValue> stack;
  for (const auto& input : node_->inputs()) {
    const auto& id = input.value->id();
    stack.emplace_back(executionFrame.getIValue(id));
  }

  // call the method
  method_->run(stack);

  // set outputs
  const auto& outputs = node_->outputs();
  SMITH_CHECK(outputs.size() == stack.size());
  for (auto&& [i, outputValue] : c10::enumerate(stack)) {
    executionFrame.setIValue(outputs[i]->id(), std::move(outputValue));
  }
}

} // namespace smith::nativert
