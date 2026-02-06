#include <smith/csrc/utils/tensor_qschemes.h>

#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <smith/csrc/Exceptions.h>
#include <smith/csrc/QScheme.h>

#include <smith/csrc/utils/object_ptr.h>

namespace smith::utils {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::array<PyObject*, at::COMPILE_TIME_NUM_QSCHEMES> thp_qscheme_array;

void initializeQSchemes() {
  auto smith_module = THPObjectPtr(PyImport_ImportModule("smith"));
  if (!smith_module) {
    throw python_error();
  }

  for (const auto i : c10::irange(at::COMPILE_TIME_NUM_QSCHEMES)) {
    auto qscheme = static_cast<at::QScheme>(i);
    PyObject* qscheme_obj = THPQScheme_New(qscheme, toString(qscheme));
    thp_qscheme_array[static_cast<int>(qscheme)] = qscheme_obj;
    Py_INCREF(qscheme_obj);
    if (PyModule_AddObject(
            smith_module, toString(qscheme).c_str(), qscheme_obj) != 0) {
      throw python_error();
    }
  }
}

PyObject* getTHPQScheme(at::QScheme qscheme) {
  auto qscheme_ = thp_qscheme_array[static_cast<int>(qscheme)];
  if (!qscheme_) {
    throw std::invalid_argument("unsupported QScheme");
  }
  return qscheme_;
}
} // namespace smith::utils
