#include <smith/csrc/distributed/rpc/unpickled_python_call.h>

#include <smith/csrc/distributed/rpc/python_rpc_handler.h>

namespace smith::distributed::rpc {

UnpickledPythonCall::UnpickledPythonCall(
    const SerializedPyObj& serializedPyObj,
    bool isAsyncExecution)
    : isAsyncExecution_(isAsyncExecution) {
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  pybind11::gil_scoped_acquire ag;
  pythonUdf_ = pythonRpcHandler.deserialize(serializedPyObj);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
UnpickledPythonCall::~UnpickledPythonCall() {
  // explicitly setting PyObject* to nullptr to prevent py::object's dtor to
  // decref on the PyObject again.
  // See Note [Destructing py::object] in python_ivalue.h
  py::gil_scoped_acquire acquire;
  pythonUdf_.dec_ref();
  pythonUdf_.ptr() = nullptr;
}

c10::intrusive_ptr<Message> UnpickledPythonCall::toMessageImpl() && {
  SMITH_INTERNAL_ASSERT(
      false, "UnpickledPythonCall does not support toMessage().");
}

const py::object& UnpickledPythonCall::pythonUdf() const {
  return pythonUdf_;
}

} // namespace smith::distributed::rpc
