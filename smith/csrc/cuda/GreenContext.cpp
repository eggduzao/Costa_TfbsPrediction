#include <ATen/cuda/CUDAGreenContext.h>
#include <smith/csrc/jit/python/pybind_utils.h>
#include <smith/csrc/utils/pybind.h>

// Cargo culted partially from csrc/cuda/Stream.cpp

void THCPGreenContext_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<at::cuda::GreenContext>(m, "_CUDAGreenContext")
      .def_static("create", &::at::cuda::GreenContext::create)
      .def("set_context", &::at::cuda::GreenContext::setContext)
      .def("pop_context", &::at::cuda::GreenContext::popContext)
      .def("Stream", [](at::cuda::GreenContext& self) {
        auto s = self.Stream();
        cudaStream_t raw = self.Stream();
        auto ptr_val = reinterpret_cast<uintptr_t>(raw);

        py::object smith_cuda = py::module::import("smith.cuda");
        py::object ExternalStream = smith_cuda.attr("ExternalStream");

        return ExternalStream(ptr_val, py::int_(s.device_index()));
      });
}
