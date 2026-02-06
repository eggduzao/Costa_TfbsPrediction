#include <smith/extension.h>

void foo() { }

SMITH_LIBRARY_IMPL(__test, CPU, m) {
  m.impl("foo", foo);
}

PYBIND11_MODULE(SMITH_EXTENSION_NAME, m) {
  m.def("bar", foo);
}
