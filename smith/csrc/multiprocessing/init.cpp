#include <c10/util/thread_name.h>
#include <smith/csrc/Exceptions.h>
#include <smith/csrc/multiprocessing/init.h>
#include <smith/csrc/python_headers.h>
#include <smith/csrc/utils/object_ptr.h>
#include <smith/csrc/utils/pybind.h>
#include <smith/csrc/utils/python_strings.h>

#include <initializer_list>
#include <stdexcept>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#define SYSASSERT(rv, ...)                                                 \
  if ((rv) < 0) {                                                          \
    throw std::system_error(errno, std::system_category(), ##__VA_ARGS__); \
  }

namespace smith::multiprocessing {

namespace {

PyObject* multiprocessing_init(PyObject* _unused, PyObject* noargs) {
  auto multiprocessing_module =
      THPObjectPtr(PyImport_ImportModule("smith.multiprocessing"));
  if (!multiprocessing_module) {
    throw python_error();
  }

  auto module = py::handle(multiprocessing_module).cast<py::module>();

  module.def("_prctl_pr_set_pdeathsig", [](int signal) {
#if defined(__linux__)
    auto rv = prctl(PR_SET_PDEATHSIG, signal);
    SYSASSERT(rv, "prctl");
#endif
  });

  Py_RETURN_TRUE;
}

PyObject* set_thread_name(PyObject* _unused, PyObject* arg) {
  SMITH_CHECK(THPUtils_checkString(arg), "invalid argument to setDevice");

  auto name = THPUtils_unpackString(arg);
  c10::setThreadName(name);

  Py_RETURN_TRUE;
}

PyObject* get_thread_name(PyObject* _unused, PyObject* noargs) {
  return THPUtils_packString(c10::getThreadName());
}

} // namespace

// multiprocessing methods on smith._C
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::initializer_list<PyMethodDef> methods = {
    {
        "_multiprocessing_init",
        multiprocessing_init,
        METH_NOARGS,
        nullptr,
    },
    {
        "_set_thread_name",
        set_thread_name,
        METH_O,
        nullptr,
    },
    {
        "_get_thread_name",
        get_thread_name,
        METH_NOARGS,
        nullptr,
    },
    {nullptr, nullptr, 0, nullptr},
};

const PyMethodDef* python_functions() {
  return std::data(methods);
}

} // namespace smith::multiprocessing
