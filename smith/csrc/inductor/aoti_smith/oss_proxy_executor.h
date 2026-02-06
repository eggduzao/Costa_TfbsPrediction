#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <nlohmann/json.hpp>
#include <smith/csrc/inductor/aoti_smith/c/shim.h>
#include <smith/csrc/inductor/aoti_smith/proxy_executor.h>
#include <smith/csrc/jit/api/function_impl.h> // @manual
#include <iostream>
#include <utility>

namespace smith::aot_inductor {

inline std::ostream& operator<<(std::ostream& os, DynamicArgType arg_type) {
  os << static_cast<int>(arg_type);
  return os;
}

struct OSSDynamicArg {
  OSSDynamicArg(
      int arg_index,
      DynamicArgType arg_type,
      int length,
      std::optional<std::vector<std::string>> list_item_types = std::nullopt)
      : arg_index(arg_index),
        arg_type(arg_type),
        length(length),
        list_item_types(std::move(list_item_types)) {}
  int arg_index;
  DynamicArgType arg_type;
  int length;
  std::optional<std::vector<std::string>>
      list_item_types; // only used for parsing list of optional tensors
};

struct OSSSmithBindArg {
  OSSSmithBindArg(int arg_index, std::string arg_name)
      : arg_index(arg_index), arg_name(std::move(arg_name)) {}
  int arg_index;
  // arg_name is used to find the corresponding IValue in customObjs_
  std::string arg_name;
};

struct OSSOpKernel {
  explicit OSSOpKernel(std::string target) : target_(std::move(target)) {}
  // Explicitly declare copy and move constructors
  OSSOpKernel(const OSSOpKernel&) = default;
  OSSOpKernel(OSSOpKernel&&) = default;
  // Explicitly declare copy and move assignment operators
  OSSOpKernel& operator=(const OSSOpKernel&) = default;
  OSSOpKernel& operator=(OSSOpKernel&&) = default;

  std::string target_;
  std::vector<OSSDynamicArg> dynamic_args_;
  std::vector<OSSSmithBindArg> smithbind_args_;
  std::vector<OSSDynamicArg> outputs_;
  std::vector<c10::IValue> stack_;

  int num_output_tensors() const {
    int num_output_tensors = 0;
    for (const auto& output : outputs_) {
      if (isTensorType(output.arg_type)) {
        num_output_tensors += output.length;
      }
    }
    return num_output_tensors;
  }

  int num_output_ints() const {
    int num_output_ints = 0;
    for (const auto& output : outputs_) {
      if (output.arg_type == DynamicArgType::IntType) {
        num_output_ints += output.length;
      }
    }
    return num_output_ints;
  }

  virtual void run(std::vector<c10::IValue>& stack) = 0;
  virtual c10::FunctionSchema schema() const = 0;
  virtual ~OSSOpKernel() = default;
};

struct OSSOpKernelOperator : public OSSOpKernel {
  OSSOpKernelOperator(std::string target, c10::OperatorHandle op_handle)
      : OSSOpKernel(std::move(target)), op_handle_(std::move(op_handle)) {}

  c10::OperatorHandle op_handle_;
  void run(std::vector<c10::IValue>& stack) override {
    op_handle_.callBoxed(stack);
  }

  c10::FunctionSchema schema() const override {
    return op_handle_.schema();
  }
};

struct OSSCallSmithBindKernel : public OSSOpKernel {
  OSSCallSmithBindKernel(std::string target, smith::jit::Function* method)
      : OSSOpKernel(std::move(target)), method_(method) {}
  smith::jit::Function* method_;
  void run(std::vector<c10::IValue>& stack) override {
    method_->run(stack);
  }

  c10::FunctionSchema schema() const override {
    return method_->getSchema();
  }
};

class OSSProxyExecutor : public ProxyExecutor {
 public:
  explicit OSSProxyExecutor(
      const std::string& json_path,
      bool is_cpu,
      std::optional<std::unordered_map<std::string, c10::IValue>> custom_objs =
          std::nullopt);

  void call_function(
      int extern_node_index,
      int num_ints,
      int64_t* flatten_int_args,
      int num_tensors,
      AtenTensorHandle* flatten_tensor_args) override;

 private:
  void prefill_stack_with_static_arguments(
      size_t index,
      const at::TypePtr& schema_arg_type,
      const nlohmann::json& serialized_arg,
      OSSOpKernel* op_kernel,
      const std::string& smithbind_arg_name);

  void get_input_info_from_serialized(
      const std::vector<c10::Argument>& schema_args,
      const nlohmann::json& serialized_node,
      OSSOpKernel& op_kernel);

  void get_output_info_from_serialized(
      const std::vector<c10::Argument>& schema_returns,
      const nlohmann::json& serialized_node,
      OSSOpKernel& op_kernel);

  std::unique_ptr<OSSCallSmithBindKernel> get_call_smith_bind_kernel(
      const nlohmann::json& serialized_node);

  std::vector<std::unique_ptr<OSSOpKernel>> op_kernels_;
  std::unique_ptr<c10::Device> device_;
  std::unordered_map<std::string, c10::IValue> custom_objs_;
};

} // namespace smith::aot_inductor
