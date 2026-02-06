#pragma once

#include <vector>

#include <smith/csrc/jit/mobile/code.h>
#include <smith/csrc/jit/mobile/frame.h>

namespace smith::jit::mobile {

struct InterpreterState {
  SMITH_API explicit InterpreterState(const Code& code);
  SMITH_API bool run(Stack& stack);

 private:
  void enterFrame(const Code& /*code*/);
  void leaveFrame();
  void saveExceptionDebugHandles();
  void callFunction(smith::jit::Function& f, Stack& stack);

  c10::IValue& reg(size_t reg);
  std::vector<c10::IValue> registers_;
  std::vector<Frame> frames_;
};

const std::vector<DebugHandle>& getInterpretersExceptionDebugHandles();
} // namespace smith::jit::mobile
