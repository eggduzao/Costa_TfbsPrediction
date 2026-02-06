#include <smith/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <smith/nativert/executor/AOTInductorDelegateExecutor.h>

namespace smith::nativert {

namespace {
std::unique_ptr<smith::inductor::AOTIModelContainerRunner>
create_aoti_model_container_runner_cuda(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded) {
  return std::make_unique<smith::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path, num_models, device_str, cubin_dir, run_single_threaded);
}
} // namespace

C10_REGISTER_TYPED_CREATOR(
    AOTIModelContainerRunnerRegistry,
    at::kCUDA,
    create_aoti_model_container_runner_cuda)

} // namespace smith::nativert
