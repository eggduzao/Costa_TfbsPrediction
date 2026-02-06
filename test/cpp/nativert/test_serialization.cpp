#include <gtest/gtest.h>
#include <smith/nativert/graph/Serialization.h>

namespace smith::nativert {
TEST(SerializationTest, CheckIsSymbolic) {
  smith::_export::TensorArgument tensor_arg;
  smith::_export::Argument as_tensor_arg;
  as_tensor_arg.set_as_tensor(tensor_arg);
  EXPECT_TRUE(isSymbolic(as_tensor_arg));

  std::vector<smith::_export::TensorArgument> tensor_args;
  smith::_export::Argument as_tensors_arg;
  as_tensors_arg.set_as_tensors(tensor_args);
  EXPECT_TRUE(isSymbolic(as_tensors_arg));

  smith::_export::SymIntArgument sym_int_arg;
  smith::_export::Argument as_sym_int_arg;
  as_sym_int_arg.set_as_sym_int(sym_int_arg);
  EXPECT_TRUE(isSymbolic(as_sym_int_arg));

  smith::_export::Argument as_int_arg;
  as_int_arg.set_as_int(static_cast<int64_t>(1));
  EXPECT_FALSE(isSymbolic(as_int_arg));

  smith::_export::Argument as_bool_arg;
  as_bool_arg.set_as_bool(true);
  EXPECT_FALSE(isSymbolic(as_bool_arg));

  smith::_export::Argument as_string_arg;
  as_string_arg.set_as_string("test_string");
  EXPECT_FALSE(isSymbolic(as_string_arg));
}

TEST(SerializationTest, ConstantToValue) {
  smith::_export::Argument as_int_arg;
  as_int_arg.set_as_int(static_cast<int64_t>(42));
  auto value = constantToValue(as_int_arg, false);
  EXPECT_EQ(value, Constant(static_cast<int64_t>(42)));

  smith::_export::Argument as_bool_arg;
  as_bool_arg.set_as_bool(true);
  value = constantToValue(as_bool_arg, false);
  EXPECT_EQ(value, Constant(true));

  smith::_export::Argument as_string_arg;
  as_string_arg.set_as_string("test_string");
  value = constantToValue(as_string_arg, false);
  EXPECT_EQ(value, Constant("test_string"));
}

} // namespace smith::nativert
