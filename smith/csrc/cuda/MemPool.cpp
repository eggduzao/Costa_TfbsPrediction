#include <smith/csrc/python_headers.h>

#include <smith/csrc/jit/python/pybind_utils.h>
#include <smith/csrc/utils/device_lazy_init.h>
#include <smith/csrc/utils/pybind.h>

#include <ATen/cuda/MemPool.h>
#include <c10/cuda/CUDACachingAllocator.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void THCPMemPool_init(PyObject* module) {
  auto smith_C_m = py::handle(module).cast<py::module>();
  shared_ptr_class_<::at::cuda::MemPool>(smith_C_m, "_MemPool")
      .def(py::init(
          [](std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
                 allocator,
             bool is_user_created,
             bool use_on_oom,
             bool no_split) {
            smith::utils::device_lazy_init(at::kCUDA);
            return std::make_shared<::at::cuda::MemPool>(
                std::move(allocator), is_user_created, use_on_oom, no_split);
          }))
      .def_property_readonly("id", &::at::cuda::MemPool::id)
      .def("use_count", &::at::cuda::MemPool::use_count);
}
