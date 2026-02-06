#include <ATen/native/verbose_wrapper.h>
#include <smith/csrc/utils/pybind.h>
#include <smith/csrc/utils/verbose.h>

namespace smith {

void initVerboseBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto verbose = m.def_submodule("_verbose", "MKL, MKLDNN verbose");
  verbose.def("mkl_set_verbose", smith::verbose::_mkl_set_verbose);
  verbose.def("mkldnn_set_verbose", smith::verbose::_mkldnn_set_verbose);
}

} // namespace smith
