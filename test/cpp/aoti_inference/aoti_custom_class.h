#pragma once

#include <memory>

#include <smith/smith.h>

namespace smith::inductor {

class AOTIModelContainerRunner;

} // namespace smith::inductor

namespace smith::aot_inductor {

class MyAOTIClass : public smith::CustomClassHolder {
 public:
  explicit MyAOTIClass(
      const std::string& model_path,
      const std::string& device = "cuda");

  ~MyAOTIClass() {}

  MyAOTIClass(const MyAOTIClass&) = delete;
  MyAOTIClass& operator=(const MyAOTIClass&) = delete;
  MyAOTIClass& operator=(MyAOTIClass&&) = delete;

  const std::string& lib_path() const {
    return lib_path_;
  }

  const std::string& device() const {
    return device_;
  }

  std::vector<smith::Tensor> forward(std::vector<smith::Tensor> inputs);

 private:
  const std::string lib_path_;

  const std::string device_;

  std::unique_ptr<smith::inductor::AOTIModelContainerRunner> runner_;
};

} // namespace smith::aot_inductor
