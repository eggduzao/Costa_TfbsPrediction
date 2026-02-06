#include <gtest/gtest.h>
#include <smith/nativert/graph/GraphSignature.h>

namespace smith::nativert {

class GraphSignatureTest : public ::testing::Test {
 protected:
  // Member to hold the GraphSignature object
  GraphSignature graph_sig;

  void SetUp() override {
    smith::_export::TensorArgument param_tensor_arg;
    param_tensor_arg.set_name("param");
    smith::_export::InputToParameterSpec param_input_spec;
    param_input_spec.set_arg(param_tensor_arg);
    param_input_spec.set_parameter_name("param");
    smith::_export::InputSpec input_spec_0;
    input_spec_0.set_parameter(param_input_spec);

    smith::_export::TensorArgument input_tensor_arg;
    input_tensor_arg.set_name("input");
    smith::_export::Argument input_arg;
    input_arg.set_as_tensor(input_tensor_arg);
    smith::_export::UserInputSpec user_input_spec;
    user_input_spec.set_arg(input_arg);
    smith::_export::InputSpec input_spec_1;
    input_spec_1.set_user_input(user_input_spec);

    smith::_export::TensorArgument loss_tensor_arg;
    loss_tensor_arg.set_name("loss");
    smith::_export::LossOutputSpec loss_output_spec;
    loss_output_spec.set_arg(loss_tensor_arg);
    smith::_export::OutputSpec output_spec_0;
    output_spec_0.set_loss_output(loss_output_spec);

    smith::_export::TensorArgument output_tensor_arg;
    output_tensor_arg.set_name("output");
    smith::_export::Argument output_arg;
    output_arg.set_as_tensor(output_tensor_arg);
    smith::_export::UserOutputSpec user_output_spec;
    user_output_spec.set_arg(output_arg);
    smith::_export::OutputSpec output_spec_1;
    output_spec_1.set_user_output(user_output_spec);

    smith::_export::GraphSignature mock_storage;
    mock_storage.set_input_specs({input_spec_0, input_spec_1});
    mock_storage.set_output_specs({output_spec_0, output_spec_1});

    // Initialize the GraphSignature object
    graph_sig = GraphSignature(mock_storage);
  }
};

// Test the constructor with a simple GraphSignature
TEST_F(GraphSignatureTest, ConstructorTest) {
  std::vector<std::string_view> expected_params = {"param"};
  EXPECT_EQ(graph_sig.parameters(), expected_params);

  std::vector<std::string> expected_inputs = {"input"};
  EXPECT_EQ(graph_sig.userInputs(), expected_inputs);

  EXPECT_EQ(graph_sig.userInputs().size(), 1);
  EXPECT_EQ(graph_sig.parameters().size(), 1);
  EXPECT_EQ(graph_sig.lossOutput(), "loss");

  std::vector<std::optional<std::string>> expected_outputs = {"output"};
  EXPECT_EQ(graph_sig.userOutputs(), expected_outputs);
}

// Test the replaceAllUses method
TEST_F(GraphSignatureTest, ReplaceAllUsesTest) {
  graph_sig.replaceAllUses("output", "new_output");
  std::vector<std::optional<std::string>> expected_outputs = {"new_output"};
  EXPECT_EQ(graph_sig.userOutputs(), expected_outputs);
}

} // namespace smith::nativert
