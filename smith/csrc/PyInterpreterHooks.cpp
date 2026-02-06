#include <smith/csrc/PyInterpreter.h>
#include <smith/csrc/PyInterpreterHooks.h>

namespace smith::detail {

PyInterpreterHooks::PyInterpreterHooks(
    c10::impl::PyInterpreterHooksArgs /*unused*/) {}

c10::impl::PyInterpreter* PyInterpreterHooks::getPyInterpreter() const {
  // Delegate to the existing implementation
  return ::getPyInterpreter();
}

} // namespace smith::detail

// Sigh, the registry doesn't support namespaces :(
using c10::impl::PyInterpreterHooksRegistry;
using c10::impl::RegistererPyInterpreterHooksRegistry;
using PyInterpreterHooks = smith::detail::PyInterpreterHooks;
// Register the implementation
REGISTER_PYTHON_HOOKS(PyInterpreterHooks)
