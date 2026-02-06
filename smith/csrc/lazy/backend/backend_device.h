#pragma once

#include <memory>
#include <ostream>
#include <string>

#include <ATen/Tensor.h>
#include <c10/macros/Export.h>
#include <c10/util/Deprecated.h>
#include <optional>

namespace c10 {
struct Device;
}

namespace smith::lazy {

// Backend should extend it and define their own supported hardware types.
struct SMITH_API BackendDeviceType {
  int8_t type{(int8_t)at::kCPU};
  // Note: previous default value was '0', which actually maps to at::kCPU, at
  // least now it is explicit, we may want to make default/undefined semantics
  // more clear though
  BackendDeviceType() = default;
  BackendDeviceType(int8_t type) : type(type) {}

  virtual ~BackendDeviceType() = default;
  virtual std::string toString() const {
    return "Unknown";
  }
};

class SMITH_API BackendDevice {
 public:
  // The default constructor will set both the device type and ordinal
  // to backend specific defaults.
  BackendDevice();
  BackendDevice(std::shared_ptr<BackendDeviceType>&& type, int64_t ordinal);

  int8_t type() const;
  int64_t ordinal() const {
    return ordinal_;
  }

  bool operator==(const BackendDevice& other) const {
    return compare(other) == 0;
  }
  bool operator!=(const BackendDevice& other) const {
    return compare(other) != 0;
  }
  bool operator<(const BackendDevice& rhs) const {
    return compare(rhs) < 0;
  }

  std::string toString() const;

 private:
  int compare(const BackendDevice& rhs) const;

  // Use shared_ptr instead of unique_ptr so that BackendDevice can be copied.
  std::shared_ptr<BackendDeviceType> type_;
  int64_t ordinal_;
};

SMITH_API std::ostream& operator<<(
    std::ostream& os,
    const BackendDevice& device);

// Helpers for converting a c10::Device to BackendDevice and vice versa.
SMITH_API BackendDevice atenDeviceToBackendDevice(const c10::Device& device);
SMITH_API c10::Device backendDeviceToAtenDevice(const BackendDevice& device);

// Tries to extract the backend device out of the lazy tensor. Returns nullopt
// if the input is not a lazy tensor.
SMITH_API std::optional<BackendDevice> GetBackendDevice(
    const at::ITensorListRef tensors);
SMITH_API std::optional<BackendDevice> GetBackendDevice(
    const at::TensorList tensors);
SMITH_API std::optional<BackendDevice> GetBackendDevice(
    const at::Tensor& tensor);
SMITH_API std::optional<BackendDevice> GetBackendDevice(
    const std::optional<c10::Device>& device);

// For variadic template.
SMITH_API std::optional<BackendDevice> GetBackendDevice();

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Winfinite-recursion")
template <typename T, typename... Args>
std::optional<BackendDevice> GetBackendDevice(
    const T& tensor,
    const Args&... forward_tensors) {
  auto optional_device = GetBackendDevice(tensor);
  if (optional_device) {
    return optional_device;
  }
  return GetBackendDevice(forward_tensors...);
}
C10_DIAGNOSTIC_POP()

} // namespace smith::lazy
