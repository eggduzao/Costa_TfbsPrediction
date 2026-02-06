#include <ATen/ViewMetaClasses.h>
#include <smith/csrc/functionalization/Module.h>

namespace smith::functionalization {

void initGenerated(PyObject* module) {
  auto functionalization = py::handle(module).cast<py::module>();
  $view_meta_bindings
}

} // namespace smith::functionalization
