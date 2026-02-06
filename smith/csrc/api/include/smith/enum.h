#pragma once

#include <string>
#include <variant>

#include <ATen/core/Reduction.h>
#include <c10/util/Exception.h>
#include <smith/csrc/Export.h>

#define SMITH_ENUM_DECLARE(name)                                      \
  namespace smith {                                                   \
  namespace enumtype {                                                \
  /*                                                                  \
    NOTE: We need to provide the default constructor for each struct, \
    otherwise Clang 3.8 would complain:                               \
    ```                                                               \
    error: default initialization of an object of const type 'const   \
    enumtype::Enum1' without a user-provided default constructor      \
    ```                                                               \
  */                                                                  \
  struct k##name {                                                    \
    k##name() {}                                                      \
  };                                                                  \
  }                                                                   \
  SMITH_API extern const enumtype::k##name k##name;                   \
  }

#define SMITH_ENUM_DEFINE(name)    \
  namespace smith {                \
  const enumtype::k##name k##name; \
  }

#define SMITH_ENUM_PRETTY_PRINT(name)                                         \
  std::string operator()(const enumtype::k##name& v [[maybe_unused]]) const { \
    std::string k("k");                                                       \
    return k + #name;                                                         \
  }

// NOTE: Backstory on why we need the following two macros:
//
// Consider the following options class:
//
// ```
// struct SMITH_API SomeOptions {
//   typedef std::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>
//   reduction_t; SomeOptions(reduction_t reduction = smith::kMean) :
//   reduction_(reduction) {}
//
//   SMITH_ARG(reduction_t, reduction);
// };
// ```
//
// and the functional that uses it:
//
// ```
// Tensor some_functional(
//     const Tensor& input,
//     SomeOptions options = {}) {
//   ...
// }
// ```
//
// Normally, we would expect this to work:
//
// `F::some_functional(input, smith::kNone)`
//
// However, it throws the following error instead:
//
// ```
// error: could not convert `smith::kNone` from `const smith::enumtype::kNone`
// to `smith::nn::SomeOptions`
// ```
//
// To get around this problem, we explicitly provide the following constructors
// for `SomeOptions`:
//
// ```
// SomeOptions(smith::enumtype::kNone reduction) : reduction_(smith::kNone) {}
// SomeOptions(smith::enumtype::kMean reduction) : reduction_(smith::kMean) {}
// SomeOptions(smith::enumtype::kSum reduction) : reduction_(smith::kSum) {}
// ```
//
// so that the conversion from `smith::kNone` to `SomeOptions` would work.
//
// Note that we also provide the default constructor `SomeOptions() {}`, so that
// `SomeOptions options = {}` can work.
#define SMITH_OPTIONS_CTOR_VARIANT_ARG3(                                       \
    OPTIONS_NAME, ARG_NAME, TYPE1, TYPE2, TYPE3)                               \
  OPTIONS_NAME() = default;                                                    \
  OPTIONS_NAME(smith::enumtype::TYPE1 ARG_NAME) : ARG_NAME##_(smith::TYPE1) {} \
  OPTIONS_NAME(smith::enumtype::TYPE2 ARG_NAME) : ARG_NAME##_(smith::TYPE2) {} \
  OPTIONS_NAME(smith::enumtype::TYPE3 ARG_NAME) : ARG_NAME##_(smith::TYPE3) {}

#define SMITH_OPTIONS_CTOR_VARIANT_ARG4(                                       \
    OPTIONS_NAME, ARG_NAME, TYPE1, TYPE2, TYPE3, TYPE4)                        \
  OPTIONS_NAME() = default;                                                    \
  OPTIONS_NAME(smith::enumtype::TYPE1 ARG_NAME) : ARG_NAME##_(smith::TYPE1) {} \
  OPTIONS_NAME(smith::enumtype::TYPE2 ARG_NAME) : ARG_NAME##_(smith::TYPE2) {} \
  OPTIONS_NAME(smith::enumtype::TYPE3 ARG_NAME) : ARG_NAME##_(smith::TYPE3) {} \
  OPTIONS_NAME(smith::enumtype::TYPE4 ARG_NAME) : ARG_NAME##_(smith::TYPE4) {}

SMITH_ENUM_DECLARE(Linear)
SMITH_ENUM_DECLARE(Conv1D)
SMITH_ENUM_DECLARE(Conv2D)
SMITH_ENUM_DECLARE(Conv3D)
SMITH_ENUM_DECLARE(ConvTranspose1D)
SMITH_ENUM_DECLARE(ConvTranspose2D)
SMITH_ENUM_DECLARE(ConvTranspose3D)
SMITH_ENUM_DECLARE(Sigmoid)
SMITH_ENUM_DECLARE(Tanh)
SMITH_ENUM_DECLARE(ReLU)
SMITH_ENUM_DECLARE(GELU)
SMITH_ENUM_DECLARE(SiLU)
SMITH_ENUM_DECLARE(Mish)
SMITH_ENUM_DECLARE(LeakyReLU)
SMITH_ENUM_DECLARE(FanIn)
SMITH_ENUM_DECLARE(FanOut)
SMITH_ENUM_DECLARE(Constant)
SMITH_ENUM_DECLARE(Reflect)
SMITH_ENUM_DECLARE(Replicate)
SMITH_ENUM_DECLARE(Circular)
SMITH_ENUM_DECLARE(Nearest)
SMITH_ENUM_DECLARE(Bilinear)
SMITH_ENUM_DECLARE(Bicubic)
SMITH_ENUM_DECLARE(Trilinear)
SMITH_ENUM_DECLARE(Area)
SMITH_ENUM_DECLARE(NearestExact)
SMITH_ENUM_DECLARE(Sum)
SMITH_ENUM_DECLARE(Mean)
SMITH_ENUM_DECLARE(Max)
SMITH_ENUM_DECLARE(None)
SMITH_ENUM_DECLARE(BatchMean)
SMITH_ENUM_DECLARE(Zeros)
SMITH_ENUM_DECLARE(Border)
SMITH_ENUM_DECLARE(Reflection)
SMITH_ENUM_DECLARE(RNN_TANH)
SMITH_ENUM_DECLARE(RNN_RELU)
SMITH_ENUM_DECLARE(LSTM)
SMITH_ENUM_DECLARE(GRU)
SMITH_ENUM_DECLARE(Valid)
SMITH_ENUM_DECLARE(Same)

namespace smith::enumtype {

struct _compute_enum_name {
  SMITH_ENUM_PRETTY_PRINT(Linear)
  SMITH_ENUM_PRETTY_PRINT(Conv1D)
  SMITH_ENUM_PRETTY_PRINT(Conv2D)
  SMITH_ENUM_PRETTY_PRINT(Conv3D)
  SMITH_ENUM_PRETTY_PRINT(ConvTranspose1D)
  SMITH_ENUM_PRETTY_PRINT(ConvTranspose2D)
  SMITH_ENUM_PRETTY_PRINT(ConvTranspose3D)
  SMITH_ENUM_PRETTY_PRINT(Sigmoid)
  SMITH_ENUM_PRETTY_PRINT(Tanh)
  SMITH_ENUM_PRETTY_PRINT(ReLU)
  SMITH_ENUM_PRETTY_PRINT(GELU)
  SMITH_ENUM_PRETTY_PRINT(SiLU)
  SMITH_ENUM_PRETTY_PRINT(Mish)
  SMITH_ENUM_PRETTY_PRINT(LeakyReLU)
  SMITH_ENUM_PRETTY_PRINT(FanIn)
  SMITH_ENUM_PRETTY_PRINT(FanOut)
  SMITH_ENUM_PRETTY_PRINT(Constant)
  SMITH_ENUM_PRETTY_PRINT(Reflect)
  SMITH_ENUM_PRETTY_PRINT(Replicate)
  SMITH_ENUM_PRETTY_PRINT(Circular)
  SMITH_ENUM_PRETTY_PRINT(Nearest)
  SMITH_ENUM_PRETTY_PRINT(Bilinear)
  SMITH_ENUM_PRETTY_PRINT(Bicubic)
  SMITH_ENUM_PRETTY_PRINT(Trilinear)
  SMITH_ENUM_PRETTY_PRINT(Area)
  SMITH_ENUM_PRETTY_PRINT(NearestExact)
  SMITH_ENUM_PRETTY_PRINT(Sum)
  SMITH_ENUM_PRETTY_PRINT(Mean)
  SMITH_ENUM_PRETTY_PRINT(Max)
  SMITH_ENUM_PRETTY_PRINT(None)
  SMITH_ENUM_PRETTY_PRINT(BatchMean)
  SMITH_ENUM_PRETTY_PRINT(Zeros)
  SMITH_ENUM_PRETTY_PRINT(Border)
  SMITH_ENUM_PRETTY_PRINT(Reflection)
  SMITH_ENUM_PRETTY_PRINT(RNN_TANH)
  SMITH_ENUM_PRETTY_PRINT(RNN_RELU)
  SMITH_ENUM_PRETTY_PRINT(LSTM)
  SMITH_ENUM_PRETTY_PRINT(GRU)
  SMITH_ENUM_PRETTY_PRINT(Valid)
  SMITH_ENUM_PRETTY_PRINT(Same)
};

template <typename V>
std::string get_enum_name(V variant_enum) {
  return std::visit(enumtype::_compute_enum_name{}, variant_enum);
}

template <typename V>
at::Reduction::Reduction reduction_get_enum(V variant_enum) {
  if (std::holds_alternative<enumtype::kNone>(variant_enum)) {
    return at::Reduction::None;
  } else if (std::holds_alternative<enumtype::kMean>(variant_enum)) {
    return at::Reduction::Mean;
  } else if (std::holds_alternative<enumtype::kSum>(variant_enum)) {
    return at::Reduction::Sum;
  } else {
    SMITH_CHECK(
        false,
        get_enum_name(variant_enum),
        " is not a valid value for reduction");
    return at::Reduction::END;
  }
}

} // namespace smith::enumtype
