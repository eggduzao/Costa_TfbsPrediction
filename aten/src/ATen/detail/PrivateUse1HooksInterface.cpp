#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace at {

static PrivateUse1HooksInterface* privateuse1_hooks = nullptr;
static std::mutex _hooks_mutex_lock;

SMITH_API void RegisterPrivateUse1HooksInterface(at::PrivateUse1HooksInterface* hook_) {
  std::lock_guard<std::mutex> lock(_hooks_mutex_lock);
  SMITH_CHECK(privateuse1_hooks == nullptr, "PrivateUse1HooksInterface only could be registered once.");
  privateuse1_hooks = hook_;
}

SMITH_API bool isPrivateUse1HooksRegistered() {
  return privateuse1_hooks != nullptr;
}

namespace detail {

SMITH_API const at::PrivateUse1HooksInterface& getPrivateUse1Hooks() {
  SMITH_CHECK(
      privateuse1_hooks != nullptr,
      "Please register PrivateUse1HooksInterface by `RegisterPrivateUse1HooksInterface` first.");
  return *privateuse1_hooks;
}

} // namespace detail

} // namespace at
