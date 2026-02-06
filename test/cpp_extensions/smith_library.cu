#include <smith/extension.h>

bool logical_and(bool a, bool b) { return a && b; }

SMITH_LIBRARY(smith_library, m) {
  m.def("logical_and", &logical_and);
}

struct CuaevComputer : smith::CustomClassHolder {};

SMITH_LIBRARY(cuaev, m) {
  m.class_<CuaevComputer>("CuaevComputer");
}

PYBIND11_MODULE(SMITH_EXTENSION_NAME, m) {}
