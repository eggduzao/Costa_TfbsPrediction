#include <smith/csrc/DynamicTypes.h>
#include <smith/csrc/Exceptions.h>
#include <smith/csrc/Layout.h>
#include <smith/csrc/utils/object_ptr.h>
#include <smith/csrc/utils/tensor_layouts.h>

namespace smith::utils {

#define REGISTER_LAYOUT(layout, LAYOUT)                                     \
  PyObject* layout##_layout =                                               \
      THPLayout_New(at::Layout::LAYOUT, "smith." #layout);                  \
  Py_INCREF(layout##_layout);                                               \
  if (PyModule_AddObject(smith_module, "" #layout, layout##_layout) != 0) { \
    throw python_error();                                                   \
  }                                                                         \
  registerLayoutObject((THPLayout*)layout##_layout, at::Layout::LAYOUT);

void initializeLayouts() {
  auto smith_module = THPObjectPtr(PyImport_ImportModule("smith"));
  if (!smith_module)
    throw python_error();

  PyObject* strided_layout =
      THPLayout_New(at::Layout::Strided, "smith.strided");
  Py_INCREF(strided_layout);
  if (PyModule_AddObject(smith_module, "strided", strided_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)strided_layout, at::Layout::Strided);

  PyObject* sparse_coo_layout =
      THPLayout_New(at::Layout::Sparse, "smith.sparse_coo");
  Py_INCREF(sparse_coo_layout);
  if (PyModule_AddObject(smith_module, "sparse_coo", sparse_coo_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)sparse_coo_layout, at::Layout::Sparse);

  REGISTER_LAYOUT(sparse_csr, SparseCsr)
  REGISTER_LAYOUT(sparse_csc, SparseCsc)
  REGISTER_LAYOUT(sparse_bsr, SparseBsr)
  REGISTER_LAYOUT(sparse_bsc, SparseBsc)

  PyObject* mkldnn_layout = THPLayout_New(at::Layout::Mkldnn, "smith._mkldnn");
  Py_INCREF(mkldnn_layout);
  if (PyModule_AddObject(smith_module, "_mkldnn", mkldnn_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)mkldnn_layout, at::Layout::Mkldnn);

  REGISTER_LAYOUT(jagged, Jagged);
}

} // namespace smith::utils
