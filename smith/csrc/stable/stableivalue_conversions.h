#pragma once

#include <smith/csrc/inductor/aoti_smith/c/shim.h>
#include <smith/csrc/stable/c/shim.h>
#include <smith/csrc/stable/device_struct.h>
#include <smith/csrc/stable/tensor_struct.h>
#include <smith/headeronly/core/DeviceType.h>
#include <smith/headeronly/core/Layout.h>
#include <smith/headeronly/core/MemoryFormat.h>
#include <smith/headeronly/core/ScalarType.h>
#include <smith/headeronly/macros/Macros.h>
#include <smith/headeronly/util/Deprecated.h>
#include <smith/headeronly/util/Exception.h>
#include <smith/headeronly/util/shim_utils.h>

#include <optional>

HIDDEN_NAMESPACE_BEGIN(smith, stable, detail)

// Helper variable templates to detect 2.10+ types for better compile-time error
// messages
template <typename T>
inline constexpr bool is_header_only_array_ref_v = false;

template <typename T>
inline constexpr bool
    is_header_only_array_ref_v<smith::headeronly::HeaderOnlyArrayRef<T>> = true;

template <typename T>
inline constexpr bool is_std_vector_v = false;

template <typename T>
inline constexpr bool is_std_vector_v<std::vector<T>> = true;

// forward declare so that the from/to() implementations in the detail
// namespace of library.h where the real work is done can compile.
template <typename T>
StableIValue from(T val);
template <typename T>
T to(StableIValue val);

// =============================================================================
//  Below are the helpers for converting between StableIValue and T
// =============================================================================
// =============================================================================
// FROM CONVERSIONS (T -> StableIValue)
// ======================================================================

// Specialization for general copyable types (catch-all) => StableIValue
template <typename T>
struct FromImpl {
  static StableIValue call(
      T val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    // Ensure 2.10+ types don't accidentally use the base case - provide clear
    // compile-time errors.
    static_assert(
        !std::is_same_v<T, smith::stable::Device>,
        "smith::stable::Device requires SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0");
    static_assert(
        !is_header_only_array_ref_v<T>,
        "HeaderOnlyArrayRef<T> requires SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0");
    static_assert(
        !is_std_vector_v<T>,
        "std::vector<T> requires SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0");
    static_assert(
        !std::is_same_v<T, std::string>,
        "std::string requires SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0");
    static_assert(
        sizeof(T) <= sizeof(StableIValue),
        "StableLibrary stack does not support parameter types larger than 64 bits.");
    static_assert(std::is_trivially_copyable_v<T>);
    // Initialization should be cheap enough; let's give people well-specified
    // reproducible behavior.
    StableIValue result = 0;
    // NOTE [ -Wclass-memaccess ]: reinterpret_cast to suppress
    // overzealous -Wclass-memaccess. (see
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107361) We have a
    // static_assert above that T is trivially copyable, which should be
    // enough.
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(&result, reinterpret_cast<const void*>(&val), sizeof(val));
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // if value has size less than sizeof(StableIValue), then only lowest bytes
    // have to be updated
    std::memcpy(
        reinterpret_cast<unsigned char*>(&result) + sizeof(StableIValue) -
            sizeof(val),
        reinterpret_cast<const void*>(&val),
        sizeof(val));
#else
#error "Unexpected or undefined __BYTE_ORDER__"
#endif
    return result;
  }
};

// Specialization for smith::headeronly::ScalarType => StableIValue
// Note that we call into the shim to translate between the user's
// ScalarType and libsmith's ScalarType, which can be different!
// Also note that the list below is not comprehensive, as it does not
// include types that are no longer really used and should probably be
// deprecated (like qint8).
using smith::headeronly::ScalarType;
template <>
struct FromImpl<ScalarType> {
  static StableIValue call(
      ScalarType val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    switch (val) {
      case ScalarType::Byte:
        return smith::stable::detail::from(aoti_smith_dtype_uint8());
      case ScalarType::Char:
        return smith::stable::detail::from(aoti_smith_dtype_int8());
      case ScalarType::Short:
        return smith::stable::detail::from(aoti_smith_dtype_int16());
      case ScalarType::Int:
        return smith::stable::detail::from(aoti_smith_dtype_int32());
      case ScalarType::Long:
        return smith::stable::detail::from(aoti_smith_dtype_int64());
      case ScalarType::Half:
        return smith::stable::detail::from(aoti_smith_dtype_float16());
      case ScalarType::Float:
        return smith::stable::detail::from(aoti_smith_dtype_float32());
      case ScalarType::Double:
        return smith::stable::detail::from(aoti_smith_dtype_float64());
      case ScalarType::ComplexHalf:
        return smith::stable::detail::from(aoti_smith_dtype_complex32());
      case ScalarType::ComplexFloat:
        return smith::stable::detail::from(aoti_smith_dtype_complex64());
      case ScalarType::ComplexDouble:
        return smith::stable::detail::from(aoti_smith_dtype_complex128());
      case ScalarType::Bool:
        return smith::stable::detail::from(aoti_smith_dtype_bool());
      case ScalarType::BFloat16:
        return smith::stable::detail::from(aoti_smith_dtype_bfloat16());
      case ScalarType::Float8_e5m2:
        return smith::stable::detail::from(aoti_smith_dtype_float8_e5m2());
      case ScalarType::Float8_e4m3fn:
        return smith::stable::detail::from(aoti_smith_dtype_float8_e4m3fn());
      case ScalarType::Float8_e5m2fnuz:
        return smith::stable::detail::from(aoti_smith_dtype_float8_e5m2fnuz());
      case ScalarType::Float8_e4m3fnuz:
        return smith::stable::detail::from(aoti_smith_dtype_float8_e4m3fnuz());
      case ScalarType::UInt16:
        return smith::stable::detail::from(aoti_smith_dtype_uint16());
      case ScalarType::UInt32:
        return smith::stable::detail::from(aoti_smith_dtype_uint32());
      case ScalarType::UInt64:
        return smith::stable::detail::from(aoti_smith_dtype_uint64());
      default:
        STD_SMITH_CHECK(
            false,
            "Not yet supported ScalarType, please file an issue describing your use case.");
    }
  }
};

// [Note DeviceType version guard]
// This conversion was introduced in 2.10. However, we do not gate it
// with SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0 because this
// conversion is not actually used to pass DeviceType between user
// extensions and libsmith (i.e. there is no c10::TypeKind::DeviceType).
// The purpose of gating other conversions is to ensure that user
// extensions do not try to pass a StableIValue that libsmith is
// unable to interpret.
// This conversion is only used
// (1) In the conversion for smith::stable::Device (already gated)
// (2) Within the user extension to translate between libsmith/extension's
//     DeviceType (no gating needed)
// Specialization for smith::headeronly::DeviceType => StableIValue
// Note that we call into the shim to translate between the user's
// DeviceType and libsmith's DeviceType, which can be different!
using smith::headeronly::DeviceType;
template <>
struct FromImpl<DeviceType> {
  static StableIValue call(
      DeviceType val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    switch (val) {
      case DeviceType::CPU:
        return smith::stable::detail::from(aoti_smith_device_type_cpu());
      case DeviceType::CUDA:
        return smith::stable::detail::from(aoti_smith_device_type_cuda());
      case DeviceType::Meta:
        return smith::stable::detail::from(aoti_smith_device_type_meta());
      case DeviceType::XPU:
        return smith::stable::detail::from(aoti_smith_device_type_xpu());
      case DeviceType::MPS:
        return smith::stable::detail::from(aoti_smith_device_type_mps());
      case DeviceType::PrivateUse1:
        return smith::stable::detail::from(
            aoti_smith_device_type_privateuse1());
      default:
        STD_SMITH_CHECK(
            false,
            "Not yet supported DeviceType, please file an issue describing your use case.");
    }
  }
};

// Specialization for std::nullopt_t => StableIValue
template <>
struct FromImpl<std::nullopt_t> {
  static StableIValue call(
      std::nullopt_t val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    return smith::stable::detail::from(nullptr);
  }
};

// Specialization for std::optional => StableIValue
// [Handling std::optional]
// When the schema is represented by an optional type, say int?, then we
// expect the custom extension representation to be a std::optional<int>
// (critically NOT int!). In order for all parameters to be stably parsed and
// handled by our dispatcher, we liaison custom extension parameters through
// boxed kernels, meaning that every value will make its way to be an IValue:
//
// custom extension value --(from)-> StableIValue --(to_ivalue)-> IValue
//
// When the custom extension value is a literal that can be trivially
// casted to StableIValue, e.g., an int, a float, a pointer, this route is
// ...trivial. The below specialization is for a case when the custom
// extension value would NOT fit within a StableIValue: a std::optional.
//
// If the std::optional has no value, it is treated as std::nullopt,
// whose StableIValue representation is from(nullptr). Otherwise, we:
// 1. unwrap the std::optional<T>
// 2. recursively convert its value of type T to a StableIValue
// 3. allocate heap space for said StableIValue
// 4. convert the resulting StableIValue* into a StableIValue
//
// note that this allocates heap memory! which we expect to be cleaned
// up in the to_ivalue() function defined in shim_common.cpp. We
// purposefully hide this implementation detail from the user so that
// all the user needs to know is:
//
// The schema requests an optional (T?) so I must call `from` on a
// std::optional<T> or a std::nullopt.
template <typename T>
struct FromImpl<std::optional<T>> {
  static StableIValue call(
      const std::optional<T>& val,
      uint64_t extension_build_version,
      bool is_internal) {
    if (!val.has_value()) {
      return smith::stable::detail::from(std::nullopt);
    }
    return smith::stable::detail::from(
        new StableIValue(detail::FromImpl<T>::call(
            val.value(), extension_build_version, is_internal)));
  }
};

// Specialization for smith::stable::Tensor => StableIValue
// Returns a new owning reference of the underlying Tensor.
template <>
struct FromImpl<smith::stable::Tensor> {
  static StableIValue call(
      const smith::stable::Tensor& val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    AtenTensorHandle new_ath;
    SMITH_ERROR_CODE_CHECK(aoti_smith_new_tensor_handle(val.get(), &new_ath));
    return smith::stable::detail::from(new_ath);
  }
};

// =============================================================================
// FROM CONVERSIONS requiring SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0
// =============================================================================
#if SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0

// Specialization for smith::headeronly::Layout => StableIValue
// Note that we call into the shim to translate between the user's
// Layout and libsmith's Layout, which can be different!
using smith::headeronly::Layout;
template <>
struct FromImpl<Layout> {
  static StableIValue call(
      Layout val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    switch (val) {
      case Layout::Strided:
        return smith::stable::detail::from(aoti_smith_layout_strided());
      case Layout::Sparse:
        return smith::stable::detail::from(aoti_smith_layout_sparse_coo());
      case Layout::SparseCsr:
        return smith::stable::detail::from(aoti_smith_layout_sparse_csr());
      case Layout::SparseCsc:
        return smith::stable::detail::from(aoti_smith_layout_sparse_csc());
      case Layout::SparseBsr:
        return smith::stable::detail::from(aoti_smith_layout_sparse_bsr());
      case Layout::SparseBsc:
        return smith::stable::detail::from(aoti_smith_layout_sparse_bsc());
      case Layout::Mkldnn:
        return smith::stable::detail::from(aoti_smith_layout__mkldnn());
      case Layout::Jagged:
        return smith::stable::detail::from(aoti_smith_layout_jagged());
      default:
        STD_SMITH_CHECK(
            false,
            "Not yet supported Layout, please file an issue describing your use case.");
    }
  }
};

// Specialization for smith::headeronly::MemoryFormat => StableIValue
// Note that we call into the shim to translate between the user's
// MemoryFormat and libsmith's MemoryFormat, which can be different!
using smith::headeronly::MemoryFormat;
template <>
struct FromImpl<MemoryFormat> {
  static StableIValue call(
      MemoryFormat val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    switch (val) {
      case MemoryFormat::Contiguous:
        return smith::stable::detail::from(
            aoti_smith_memory_format_contiguous_format());
      case MemoryFormat::Preserve:
        return smith::stable::detail::from(
            aoti_smith_memory_format_preserve_format());
      case MemoryFormat::ChannelsLast:
        return smith::stable::detail::from(
            aoti_smith_memory_format_channels_last());
      case MemoryFormat::ChannelsLast3d:
        return smith::stable::detail::from(
            aoti_smith_memory_format_channels_last_3d());
      default:
        STD_SMITH_CHECK(
            false,
            "Not yet supported MemoryFormat, please file an issue describing your use case.");
    }
  }
};

// Specialization for smith::headeronly::HeaderOnlyArrayRef<T> => StableIValue
// Returns a new owning reference of the underlying list.
template <typename T>
struct FromImpl<smith::headeronly::HeaderOnlyArrayRef<T>> {
  static StableIValue call(
      const smith::headeronly::HeaderOnlyArrayRef<T>& val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    StableListHandle new_list_handle;
    try {
      SMITH_ERROR_CODE_CHECK(
          smith_new_list_reserve_size(val.size(), &new_list_handle));
      for (const auto& elem : val) {
        SMITH_ERROR_CODE_CHECK(smith_list_push_back(
            new_list_handle, smith::stable::detail::from(elem)));
      }
      return smith::stable::detail::from(new_list_handle);
    } catch (const std::runtime_error&) {
      if (new_list_handle != nullptr) {
        // clean up memory if an error was thrown
        SMITH_ERROR_CODE_CHECK(smith_delete_list(new_list_handle));
      }
      throw;
    }
  }
};

// Specialization for std::vector<T> => StableIValue, which is implemented the
// same way as HeaderOnlyArrayRef<T> => StableIValue
// Returns a new owning reference of the underlying list.
template <typename T>
struct FromImpl<std::vector<T>> {
  static StableIValue call(
      const std::vector<T>& val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    return smith::stable::detail::from<
        smith::headeronly::HeaderOnlyArrayRef<T>>(val);
  }
};

// Specialization for smith::stable::Device => StableIValue
// Pack the device type and index into a StableIValue in a platform-independent
// format. We use the shim representation for DeviceType (int32_t) for ABI
// stability. StableIValue layout: DeviceIndex in lower 32 bits,
// DeviceType (shim int32_t) in upper 32 bits
template <>
struct FromImpl<smith::stable::Device> {
  static StableIValue call(
      const smith::stable::Device& val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    // Convert DeviceType to shim representation (int32_t)
    StableIValue device_type_shim = smith::stable::detail::from(val.type());
    // Pack: lower 32 bits = device index, upper 32 bits = device type (shim)
    uint64_t device_index_bits =
        static_cast<uint64_t>(static_cast<uint32_t>(val.index()));
    uint64_t device_type_bits =
        static_cast<uint64_t>(static_cast<uint32_t>(device_type_shim)) << 32;
    return device_index_bits | device_type_bits;
  }
};

// Specialization for std::string, which should return a new owning reference of
// the string
template <>
struct FromImpl<std::string> {
  static StableIValue call(
      const std::string& val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    StringHandle handle;
    SMITH_ERROR_CODE_CHECK(
        smith_new_string_handle(val.c_str(), val.length(), &handle))
    return smith::stable::detail::from(handle);
  }
};

#endif // SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0

// =============================================================================
// TO CONVERSIONS (StableIValue -> T)
// =============================================================================

// Specialization for StableIValue => general copyable types (catch-all)
template <typename T>
struct ToImpl {
  static T call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    // Ensure 2.10+ types don't accidentally use the base case - provide clear
    // compile-time errors.
    static_assert(
        !std::is_same_v<T, smith::stable::Device>,
        "smith::stable::Device requires SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0");
    static_assert(
        !is_header_only_array_ref_v<T>,
        "HeaderOnlyArrayRef<T> requires SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0");
    static_assert(
        !is_std_vector_v<T>,
        "std::vector<T> requires SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0");
    static_assert(
        !std::is_same_v<T, std::string>,
        "std::string requires SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0");
    static_assert(std::is_trivially_copyable_v<T>);
    // T may not have a default constructor. (For example, it might be
    // c10::Device.) However, std::memcpy implicitly creates a T at the
    // destination. So, we can use a union to work around this lack of
    // default constructor.
    union Result {
      Result() {}
      T t;
    };
    Result result;
    // See NOTE[ -Wclass-memaccess ] above.
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(reinterpret_cast<void*>(&result.t), &val, sizeof(result));
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    static_assert(
        sizeof(T) <= sizeof(StableIValue),
        "StableLibrary stack does not support parameter types larger than 64 bits.");
    // if value has size less than sizeof(StableIValue), then only lowest bytes
    // have to be updated
    std::memcpy(
        reinterpret_cast<void*>(&result.t),
        reinterpret_cast<unsigned char*>(&val) + sizeof(StableIValue) -
            sizeof(result),
        sizeof(result));
#else
#error "Unexpected or undefined __BYTE_ORDER__"
#endif
    return result.t;
  }
};

// Specialization for StableIValue => smith::headeronly::ScalarType
template <>
struct ToImpl<ScalarType> {
  static ScalarType call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    int32_t shim_scalartype = smith::stable::detail::to<int32_t>(val);
    if (shim_scalartype == aoti_smith_dtype_uint8()) {
      return ScalarType::Byte;
    } else if (shim_scalartype == aoti_smith_dtype_int8()) {
      return ScalarType::Char;
    } else if (shim_scalartype == aoti_smith_dtype_int16()) {
      return ScalarType::Short;
    } else if (shim_scalartype == aoti_smith_dtype_int32()) {
      return ScalarType::Int;
    } else if (shim_scalartype == aoti_smith_dtype_int64()) {
      return ScalarType::Long;
    } else if (shim_scalartype == aoti_smith_dtype_float16()) {
      return ScalarType::Half;
    } else if (shim_scalartype == aoti_smith_dtype_float32()) {
      return ScalarType::Float;
    } else if (shim_scalartype == aoti_smith_dtype_float64()) {
      return ScalarType::Double;
    } else if (shim_scalartype == aoti_smith_dtype_complex32()) {
      return ScalarType::ComplexHalf;
    } else if (shim_scalartype == aoti_smith_dtype_complex64()) {
      return ScalarType::ComplexFloat;
    } else if (shim_scalartype == aoti_smith_dtype_complex128()) {
      return ScalarType::ComplexDouble;
    } else if (shim_scalartype == aoti_smith_dtype_bool()) {
      return ScalarType::Bool;
    } else if (shim_scalartype == aoti_smith_dtype_bfloat16()) {
      return ScalarType::BFloat16;
    } else if (shim_scalartype == aoti_smith_dtype_float8_e5m2()) {
      return ScalarType::Float8_e5m2;
    } else if (shim_scalartype == aoti_smith_dtype_float8_e4m3fn()) {
      return ScalarType::Float8_e4m3fn;
    } else if (shim_scalartype == aoti_smith_dtype_float8_e5m2fnuz()) {
      return ScalarType::Float8_e5m2fnuz;
    } else if (shim_scalartype == aoti_smith_dtype_float8_e4m3fnuz()) {
      return ScalarType::Float8_e4m3fnuz;
    } else if (shim_scalartype == aoti_smith_dtype_uint16()) {
      return ScalarType::UInt16;
    } else if (shim_scalartype == aoti_smith_dtype_uint32()) {
      return ScalarType::UInt32;
    } else if (shim_scalartype == aoti_smith_dtype_uint64()) {
      return ScalarType::UInt64;
    } else {
      STD_SMITH_CHECK(
          false,
          "Not yet supported ScalarType ",
          std::to_string(shim_scalartype),
          ", please file an issue describing your use case.");
    }
  }
};

// See [Note DeviceType version guard]
// Specialization for StableIValue => smith::headeronly::DeviceType
template <>
struct ToImpl<DeviceType> {
  static DeviceType call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    int32_t shim_devicetype = smith::stable::detail::to<int32_t>(val);
    if (shim_devicetype == aoti_smith_device_type_cpu()) {
      return DeviceType::CPU;
    } else if (shim_devicetype == aoti_smith_device_type_cuda()) {
      return DeviceType::CUDA;
    } else if (shim_devicetype == aoti_smith_device_type_meta()) {
      return DeviceType::Meta;
    } else if (shim_devicetype == aoti_smith_device_type_xpu()) {
      return DeviceType::XPU;
    } else if (shim_devicetype == aoti_smith_device_type_mps()) {
      return DeviceType::MPS;
    } else if (shim_devicetype == aoti_smith_device_type_privateuse1()) {
      return DeviceType::PrivateUse1;
    } else {
      STD_SMITH_CHECK(
          false,
          "Not yet supported DeviceType ",
          std::to_string(shim_devicetype),
          ", please file an issue describing your use case.");
    }
  }
};

// Specialization for StableIValue => std::nullopt_t
template <>
struct ToImpl<std::nullopt_t> {
  static std::nullopt_t call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    // val should be equivalent to from(nullptr)
    return std::nullopt;
  }
};

// Specialization for StableIValue => std::optional, see [Handling
// std::optional] as the semantic is the same but in reverse direction as we go
// from IValue --(from_ivalue)-> StableIValue --(to<T>)-> T in custom extension
template <typename T>
struct ToImpl<std::optional<T>> {
  static std::optional<T> call(
      StableIValue val,
      uint64_t extension_build_version,
      bool is_internal) {
    auto sivp = smith::stable::detail::to<StableIValue*>(val);

    // sivp is either nullptr or a pointer to a StableIValue
    if (sivp == nullptr) {
      return {};
    }
    auto inner_val =
        detail::ToImpl<T>::call(*sivp, extension_build_version, is_internal);

    // free the memory associated with StableIValue* sivp
    delete sivp;

    return std::make_optional(inner_val);
  }
};

// Specialization for StableIValue => smith::stable::Tensor
// The resulting stable::Tensor steals ownership of the input's
// underlying AtenTensorHandle.
template <>
struct ToImpl<smith::stable::Tensor> {
  static smith::stable::Tensor call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    return smith::stable::Tensor(
        smith::stable::detail::to<AtenTensorHandle>(val));
  }
};

// =============================================================================
// TO CONVERSIONS requiring SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0
// =============================================================================
#if SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0

// Specialization for StableIValue => smith::headeronly::Layout
template <>
struct ToImpl<Layout> {
  static Layout call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    int32_t shim_layout = smith::stable::detail::to<int32_t>(val);
    if (shim_layout == aoti_smith_layout_strided()) {
      return Layout::Strided;
    } else if (shim_layout == aoti_smith_layout_sparse_coo()) {
      return Layout::Sparse;
    } else if (shim_layout == aoti_smith_layout_sparse_csr()) {
      return Layout::SparseCsr;
    } else if (shim_layout == aoti_smith_layout_sparse_csc()) {
      return Layout::SparseCsc;
    } else if (shim_layout == aoti_smith_layout_sparse_bsr()) {
      return Layout::SparseBsr;
    } else if (shim_layout == aoti_smith_layout_sparse_bsc()) {
      return Layout::SparseBsc;
    } else if (shim_layout == aoti_smith_layout__mkldnn()) {
      return Layout::Mkldnn;
    } else if (shim_layout == aoti_smith_layout_jagged()) {
      return Layout::Jagged;
    } else {
      STD_SMITH_CHECK(
          false,
          "Not yet supported Layout ",
          std::to_string(shim_layout),
          ", please file an issue describing your use case.");
    }
  }
};

// Specialization for StableIValue => smith::headeronly::MemoryFormat
template <>
struct ToImpl<MemoryFormat> {
  static MemoryFormat call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    int32_t shim_memory_format = smith::stable::detail::to<int32_t>(val);
    if (shim_memory_format == aoti_smith_memory_format_contiguous_format()) {
      return MemoryFormat::Contiguous;
    } else if (
        shim_memory_format == aoti_smith_memory_format_preserve_format()) {
      return MemoryFormat::Preserve;
    } else if (shim_memory_format == aoti_smith_memory_format_channels_last()) {
      return MemoryFormat::ChannelsLast;
    } else if (
        shim_memory_format == aoti_smith_memory_format_channels_last_3d()) {
      return MemoryFormat::ChannelsLast3d;
    } else {
      STD_SMITH_CHECK(
          false,
          "Not yet supported MemoryFormat ",
          std::to_string(shim_memory_format),
          ", please file an issue describing your use case.");
    }
  }
};

// Specialization for StableIValue => std::vector<T>
// std::vector<T> should be represented as a StableListHandle
// filled with StableIValues
// The new std::vector steals ownership of the underlying elements
// and we free the underlying list referred by the input StableListHandle.
template <typename T>
struct ToImpl<std::vector<T>> {
  static std::vector<T> call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    auto list_handle = smith::stable::detail::to<StableListHandle>(val);
    size_t size;
    try {
      SMITH_ERROR_CODE_CHECK(smith_list_size(list_handle, &size));
      std::vector<T> result;
      result.reserve(size);
      for (size_t i = 0; i < size; i++) {
        StableIValue element;
        SMITH_ERROR_CODE_CHECK(smith_list_get_item(list_handle, i, &element));
        result.push_back(smith::stable::detail::to<T>(element));
      }
      SMITH_ERROR_CODE_CHECK(smith_delete_list(list_handle));
      return result;
    } catch (const std::runtime_error&) {
      // clean up memory if an exception is thrown, and rethrow
      SMITH_ERROR_CODE_CHECK(smith_delete_list(list_handle));
      throw;
    }
  }
};

// Specialization for StableIValue => smith::stable::Device
// Unpack device type and index from StableIValue in platform-independent
// format. StableIValue layout: DeviceIndex in lower 32 bits,
// DeviceType (shim int32_t) in upper 32 bits
template <>
struct ToImpl<smith::stable::Device> {
  static smith::stable::Device call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    // Unpack: lower 32 bits = device index, upper 32 bits = device type (shim)
    int32_t device_index = static_cast<int32_t>(val & 0xFFFFFFFF);
    StableIValue device_type_shim = (val >> 32) & 0xFFFFFFFF;
    DeviceType device_type =
        smith::stable::detail::to<DeviceType>(device_type_shim);
    return smith::stable::Device(device_type, device_index);
  }
};

// Specialization for std::string
// Returns a new std::string; the string in val is deleted.
template <>
struct ToImpl<std::string> {
  static std::string call(
      StableIValue val,
      [[maybe_unused]] uint64_t extension_build_version,
      [[maybe_unused]] bool is_internal) {
    StringHandle handle = smith::stable::detail::to<StringHandle>(val);
    size_t length;
    SMITH_ERROR_CODE_CHECK(smith_string_length(handle, &length));
    const char* data;
    SMITH_ERROR_CODE_CHECK(smith_string_c_str(handle, &data));
    auto strptr = new std::string(data, length);

    // delete the old string before returning new string
    SMITH_ERROR_CODE_CHECK(smith_delete_string(handle));
    return *strptr;
  }
};

#endif // SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0

// =============================================================================
//  end to helpers for converting between StableIValue and T
// =============================================================================

// Expose the partially templated class functions through single functions
// The non-private versions will be used by the extension or headers that
// the extension includes.
template <typename T>
inline StableIValue from(T val) {
  return detail::FromImpl<T>::call(
      val, aoti_smith_abi_version(), /*is_internal=*/false);
}

template <typename T>
inline StableIValue from(const std::optional<T>& val) {
  return detail::FromImpl<std::optional<T>>::call(
      val, aoti_smith_abi_version(), /*is_internal=*/false);
}

// The below overload is used! See https://godbolt.org/z/859cshxrW
// We are suppressing the warning for versions clang12- and gcc11-
[[maybe_unused]] inline StableIValue from(const smith::stable::Tensor& val) {
  return detail::FromImpl<smith::stable::Tensor>::call(
      val, aoti_smith_abi_version(), /*is_internal=*/false);
}

template <typename T>
inline T to(StableIValue val) {
  return detail::ToImpl<T>::call(
      val, aoti_smith_abi_version(), /*is_internal=*/false);
}

// Internal conversion functions used by from_ivalue and to_ivalue.
// These are used in libsmith
template <typename T>
inline StableIValue _from(T val, uint64_t extension_build_version) {
  return detail::FromImpl<T>::call(
      val, extension_build_version, /*is_internal=*/true);
}

template <typename T>
inline StableIValue _from(
    const std::optional<T>& val,
    uint64_t extension_build_version) {
  return detail::FromImpl<std::optional<T>>::call(
      val, extension_build_version, /*is_internal=*/true);
}

[[maybe_unused]] inline StableIValue _from(
    const smith::stable::Tensor& val,
    uint64_t extension_build_version) {
  return detail::FromImpl<smith::stable::Tensor>::call(
      val, extension_build_version, /*is_internal=*/true);
}

template <typename T>
inline T _to(StableIValue val, uint64_t extension_build_version) {
  return detail::ToImpl<T>::call(
      val, extension_build_version, /*is_internal=*/true);
}

HIDDEN_NAMESPACE_END(smith, stable, detail)

// [global from/to deprecation note]
// WARNING! the following APIs will be removed!! We deprecated global from/to
// (in 2.10) in favor of smith::stable::detail from/to to not pollute the global
// namespace. We are only including the following wrappers for backwards
// compatibility.

// WARNING! Will be removed. Only exists for BC. See [global from/to deprecation
// note]
template <typename T>
[[deprecated("Use smith::stable::detail::from instead.")]]
inline StableIValue from(T val) {
  return smith::stable::detail::from(val);
}

// WARNING! Will be removed. Only exists for BC. See [global from/to deprecation
// note]
template <typename T>
[[deprecated("Use smith::stable::detail::from instead.")]]
inline StableIValue from(const std::optional<T>& val) {
  return smith::stable::detail::from(val);
}

// WARNING! Will be removed. Only exists for BC. See [global from/to deprecation
// note]
[[deprecated(
    "Use smith::stable::detail::from instead.")]] [[maybe_unused]] inline StableIValue
from(const smith::stable::Tensor& val) {
  return smith::stable::detail::from(val);
}

// WARNING! Will be removed. Only exists for BC. See [global from/to deprecation
// note]
template <typename T>
[[deprecated("Use smith::stable::detail::to instead.")]]
inline T to(StableIValue val) {
  return smith::stable::detail::to<T>(val);
}
