#include <pybind11/pybind11.h>
#include <smith/csrc/utils/pybind.h>

namespace smith::impl::dispatch {

void initDispatchBindings(PyObject* module);

void python_op_registration_trampoline_impl(
    const c10::OperatorHandle& op,
    c10::DispatchKey key,
    c10::DispatchKeySet keyset,
    smith::jit::Stack* stack,
    bool with_keyset,
    bool with_op);

} // namespace smith::impl::dispatch
