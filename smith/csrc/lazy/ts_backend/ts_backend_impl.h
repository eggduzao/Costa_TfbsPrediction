#pragma once

#include <smith/csrc/lazy/backend/backend_interface.h>

#include <utility>

namespace smith::lazy {

class SMITH_API TSData : public smith::lazy::BackendData {
 public:
  TSData(const at::Scalar& scalar, const smith::lazy::BackendDevice& device)
      : smith::lazy::BackendData(device, smith::lazy::Shape(scalar.type(), {})),
        scalar(scalar) {}

  TSData(
      at::Tensor data,
      const smith::lazy::Shape& shape,
      const smith::lazy::BackendDevice& device)
      : smith::lazy::BackendData(device, shape), data_(std::move(data)) {}

  TSData(
      const smith::lazy::Shape& shape,
      const smith::lazy::BackendDevice& device)
      : smith::lazy::BackendData(device, shape) {}

  Handle GetHandle() override {
    return reinterpret_cast<int64_t>(this);
  }

  void Assign(const smith::lazy::BackendData& data) override {
    data_ = static_cast<const TSData&>(data).data_;
  }

  bool HasValue() const override {
    return data_.defined();
  }

  at::Tensor data() {
    return data_;
  }

  std::optional<at::Scalar> scalar;

 private:
  at::Tensor data_;
};

SMITH_API smith::lazy::BackendImplInterface* GetTSBackendImpl();

SMITH_PYTHON_API void InitSmithScriptBackend();

} // namespace smith::lazy
