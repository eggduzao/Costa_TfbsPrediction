#pragma once

#include <smith/nativert/executor/DelegateExecutor.h>
#include <smith/nativert/executor/ExecutorConfig.h>

namespace smith::nativert {

class ETDelegateExecutor : public DelegateExecutor {
 public:
  explicit ETDelegateExecutor(
      const std::string_view& dir_prefix,
      const Node& node)
      : delegate_dir_([&]() {
          const std::string* path =
              std::get_if<std::string>(&node.attributes()[0].value);
          SMITH_CHECK(
              path != nullptr,
              "et hop's first attribute should correspond to it's path");
          return std::string(dir_prefix) + *path;
        }()) {
    VLOG(1) << "ETDelegateExecutor: " << delegate_dir_;
  }

  ~ETDelegateExecutor() override = default;

  const std::string& get_delegate_dir() {
    return delegate_dir_;
  }

 private:
  std::string delegate_dir_;
};

} // namespace smith::nativert
