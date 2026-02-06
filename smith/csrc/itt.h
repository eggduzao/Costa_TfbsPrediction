#ifndef ITT_H
#define ITT_H
#include <smith/csrc/utils/pybind.h>

namespace smith::profiler {
void initIttBindings(PyObject* module); // namespace smith::profiler
}
#endif // ITT_H
