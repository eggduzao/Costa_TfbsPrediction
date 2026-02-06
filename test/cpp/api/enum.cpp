#include <gtest/gtest.h>

#include <smith/smith.h>
#include <variant>

#include <test/cpp/api/support.h>

#define SMITH_ENUM_PRETTY_PRINT_TEST(name)                           \
  {                                                                  \
    v = smith::k##name;                                              \
    std::string pretty_print_name("k");                              \
    pretty_print_name.append(#name);                                 \
    ASSERT_EQ(smith::enumtype::get_enum_name(v), pretty_print_name); \
  }

TEST(EnumTest, AllEnums) {
  std::variant<
      smith::enumtype::kLinear,
      smith::enumtype::kConv1D,
      smith::enumtype::kConv2D,
      smith::enumtype::kConv3D,
      smith::enumtype::kConvTranspose1D,
      smith::enumtype::kConvTranspose2D,
      smith::enumtype::kConvTranspose3D,
      smith::enumtype::kSigmoid,
      smith::enumtype::kTanh,
      smith::enumtype::kReLU,
      smith::enumtype::kLeakyReLU,
      smith::enumtype::kFanIn,
      smith::enumtype::kFanOut,
      smith::enumtype::kConstant,
      smith::enumtype::kReflect,
      smith::enumtype::kReplicate,
      smith::enumtype::kCircular,
      smith::enumtype::kNearest,
      smith::enumtype::kBilinear,
      smith::enumtype::kBicubic,
      smith::enumtype::kTrilinear,
      smith::enumtype::kArea,
      smith::enumtype::kSum,
      smith::enumtype::kMean,
      smith::enumtype::kMax,
      smith::enumtype::kNone,
      smith::enumtype::kBatchMean,
      smith::enumtype::kZeros,
      smith::enumtype::kBorder,
      smith::enumtype::kReflection,
      smith::enumtype::kRNN_TANH,
      smith::enumtype::kRNN_RELU,
      smith::enumtype::kLSTM,
      smith::enumtype::kGRU>
      v;

  SMITH_ENUM_PRETTY_PRINT_TEST(Linear)
  SMITH_ENUM_PRETTY_PRINT_TEST(Conv1D)
  SMITH_ENUM_PRETTY_PRINT_TEST(Conv2D)
  SMITH_ENUM_PRETTY_PRINT_TEST(Conv3D)
  SMITH_ENUM_PRETTY_PRINT_TEST(ConvTranspose1D)
  SMITH_ENUM_PRETTY_PRINT_TEST(ConvTranspose2D)
  SMITH_ENUM_PRETTY_PRINT_TEST(ConvTranspose3D)
  SMITH_ENUM_PRETTY_PRINT_TEST(Sigmoid)
  SMITH_ENUM_PRETTY_PRINT_TEST(Tanh)
  SMITH_ENUM_PRETTY_PRINT_TEST(ReLU)
  SMITH_ENUM_PRETTY_PRINT_TEST(LeakyReLU)
  SMITH_ENUM_PRETTY_PRINT_TEST(FanIn)
  SMITH_ENUM_PRETTY_PRINT_TEST(FanOut)
  SMITH_ENUM_PRETTY_PRINT_TEST(Constant)
  SMITH_ENUM_PRETTY_PRINT_TEST(Reflect)
  SMITH_ENUM_PRETTY_PRINT_TEST(Replicate)
  SMITH_ENUM_PRETTY_PRINT_TEST(Circular)
  SMITH_ENUM_PRETTY_PRINT_TEST(Nearest)
  SMITH_ENUM_PRETTY_PRINT_TEST(Bilinear)
  SMITH_ENUM_PRETTY_PRINT_TEST(Bicubic)
  SMITH_ENUM_PRETTY_PRINT_TEST(Trilinear)
  SMITH_ENUM_PRETTY_PRINT_TEST(Area)
  SMITH_ENUM_PRETTY_PRINT_TEST(Sum)
  SMITH_ENUM_PRETTY_PRINT_TEST(Mean)
  SMITH_ENUM_PRETTY_PRINT_TEST(Max)
  SMITH_ENUM_PRETTY_PRINT_TEST(None)
  SMITH_ENUM_PRETTY_PRINT_TEST(BatchMean)
  SMITH_ENUM_PRETTY_PRINT_TEST(Zeros)
  SMITH_ENUM_PRETTY_PRINT_TEST(Border)
  SMITH_ENUM_PRETTY_PRINT_TEST(Reflection)
  SMITH_ENUM_PRETTY_PRINT_TEST(RNN_TANH)
  SMITH_ENUM_PRETTY_PRINT_TEST(RNN_RELU)
  SMITH_ENUM_PRETTY_PRINT_TEST(LSTM)
  SMITH_ENUM_PRETTY_PRINT_TEST(GRU)
}
