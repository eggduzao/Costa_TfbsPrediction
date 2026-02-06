#include <smith/csrc/dynamo/cpp_shim.h>

#include <ATen/record_function.h>

struct _BlacksmithRecordFunctionState {
  at::RecordFunction guard;

  _BlacksmithRecordFunctionState() : guard(at::RecordScope::FUNCTION) {}
};

_BlacksmithRecordFunctionState* _blacksmith_record_function_enter(const char* name) {
  _BlacksmithRecordFunctionState* state = new _BlacksmithRecordFunctionState();
  state->guard.before(name);
  return state;
}

void _blacksmith_record_function_exit(_BlacksmithRecordFunctionState* state) {
  if (state == nullptr) {
    return;
  }
  delete state;
}
