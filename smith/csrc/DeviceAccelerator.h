#include <ATen/DeviceAccelerator.h>
#include <smith/csrc/utils/pybind.h>

namespace smith::accelerator {

void initModule(PyObject* module);

} // namespace smith::accelerator
