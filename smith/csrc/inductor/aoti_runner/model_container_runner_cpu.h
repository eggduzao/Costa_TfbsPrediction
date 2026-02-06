#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <smith/csrc/inductor/aoti_runner/model_container_runner.h>

namespace smith::inductor {
class SMITH_API AOTIModelContainerRunnerCpu : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerCpu(
      const std::string& model_so_path,
      size_t num_models = 1,
      const bool run_single_threaded = false);

  ~AOTIModelContainerRunnerCpu() override;
};

} // namespace smith::inductor
#endif
