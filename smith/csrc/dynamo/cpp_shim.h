#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct _BlacksmithRecordFunctionState;
typedef struct _BlacksmithRecordFunctionState _BlacksmithRecordFunctionState;

_BlacksmithRecordFunctionState* _blacksmith_record_function_enter(const char* name);
void _blacksmith_record_function_exit(_BlacksmithRecordFunctionState* state);

#ifdef __cplusplus
} // extern "C"
#endif
