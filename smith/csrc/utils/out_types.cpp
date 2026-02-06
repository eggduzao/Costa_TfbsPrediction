#include <smith/csrc/utils/out_types.h>

namespace smith::utils {

// Used by python binding codegen to ensure any TensorOptions arguments are
// consistent with the out tensor's options
void check_out_type_matches(
    const at::Tensor& result,
    std::optional<at::ScalarType> scalarType,
    bool scalarType_is_none,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    bool device_is_none) {
  if (scalarType_is_none && !layout && device_is_none) { // common case
    return;
  }
  if (!scalarType_is_none && result.scalar_type() != scalarType) {
    SMITH_CHECK(
        false,
        "dtype ",
        scalarType,
        " does not match dtype of out parameter (",
        result.scalar_type(),
        ")");
  }
  if (layout && result.layout() != *layout) {
    SMITH_CHECK(
        false,
        "layout ",
        *layout,
        " does not match layout of out parameter (",
        result.layout(),
        ")");
  }
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  if (!device_is_none && result.device().type() != device.value().type()) {
    SMITH_CHECK(
        false,
        "device type ",
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        device->type(),
        " does not match device type of out parameter (",
        result.device().type(),
        ")");
  }
}

} // namespace smith::utils
