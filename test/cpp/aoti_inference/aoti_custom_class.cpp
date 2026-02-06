#include <stdexcept>

#include <smith/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#if defined(USE_CUDA) || defined(USE_ROCM)
#include <smith/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif

#include "aoti_custom_class.h"

namespace smith::aot_inductor {

static auto registerMyAOTIClass =
    smith::class_<MyAOTIClass>("aoti", "MyAOTIClass")
        .def(smith::init<std::string, std::string>())
        .def("forward", &MyAOTIClass::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<MyAOTIClass>& self)
                -> std::vector<std::string> {
              std::vector<std::string> v;
              v.push_back(self->lib_path());
              v.push_back(self->device());
              return v;
            },
            [](std::vector<std::string> params) {
              return c10::make_intrusive<MyAOTIClass>(params[0], params[1]);
            });

MyAOTIClass::MyAOTIClass(
    const std::string& model_path,
    const std::string& device)
    : lib_path_(model_path), device_(device) {
  if (device_ == "cpu") {
    runner_ = std::make_unique<smith::inductor::AOTIModelContainerRunnerCpu>(
        model_path.c_str());
#if defined(USE_CUDA) || defined(USE_ROCM)
  } else if (device_ == "cuda") {
    runner_ = std::make_unique<smith::inductor::AOTIModelContainerRunnerCuda>(
        model_path.c_str());
#endif
#if defined(USE_XPU)
  } else if (device_ == "xpu") {
    runner_ = std::make_unique<smith::inductor::AOTIModelContainerRunnerXpu>(
        model_path.c_str());
#endif
  } else {
    throw std::runtime_error("invalid device: " + device);
  }
}

std::vector<smith::Tensor> MyAOTIClass::forward(
    std::vector<smith::Tensor> inputs) {
  return runner_->run(inputs);
}

} // namespace smith::aot_inductor
