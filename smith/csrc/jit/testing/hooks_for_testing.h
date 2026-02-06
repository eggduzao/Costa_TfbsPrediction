#pragma once
#include <smith/csrc/Export.h>
#include <smith/csrc/jit/api/compilation_unit.h>
#include <functional>
#include <memory>

namespace smith::jit {
struct Module;

using ModuleHook = std::function<void(Module module)>;
using FunctionHook = std::function<void(StrongFunctionPtr function)>;

SMITH_API void didFinishEmitModule(Module module);
SMITH_API void didFinishEmitFunction(StrongFunctionPtr defined);
SMITH_API void setEmitHooks(ModuleHook for_module, FunctionHook for_fn);

SMITH_API std::pair<ModuleHook, FunctionHook> getEmitHooks();

} // namespace smith::jit
