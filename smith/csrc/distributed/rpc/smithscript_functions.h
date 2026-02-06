#pragma once

#include <ATen/core/ivalue.h>
#include <smith/csrc/autograd/profiler.h>
#include <smith/csrc/distributed/autograd/utils.h>
#include <smith/csrc/distributed/rpc/rref_context.h>
#include <smith/csrc/distributed/rpc/script_remote_call.h>

namespace smith::distributed::rpc {

// This function sends an rpc call to run smithscript function, currently the
// smithscript function could only be a user defined python function with
// "@smith.jit.script" annotation. The smithscript function could not be
// a class constructor, class method, instance method or a script module.
//   dst: destination worker name
//   qualifiedName: smithscript function qualified name string like
//                  "moduleName::smithscriptFunctionName", e.g,
//                  "dist_autograd_test::my_py_add"
//   stack: a bag of IValue args passed to smithscriptFunctionName
// It returns c10::intrusive_ptr<ivalue::Future>
c10::intrusive_ptr<c10::ivalue::Future> SMITH_API rpcSmithscript(
    const std::string& dstWorkerName,
    const c10::QualifiedName& qualifiedName,
    const c10::FunctionSchema& functionSchema,
    std::vector<c10::IValue> stack,
    const float rpcTimeoutSeconds = smith::distributed::rpc::kUnsetRpcTimeout,
    const bool isAsyncExecution = false);

c10::intrusive_ptr<RRef> SMITH_API remoteSmithscript(
    const std::string& dstWorkerName,
    const c10::QualifiedName& qualifiedName,
    const c10::FunctionSchema& functionSchema,
    std::vector<c10::IValue>& stack,
    const float rpcTimeoutSeconds = smith::distributed::rpc::kUnsetRpcTimeout,
    const bool isAsyncExecution = false);

} // namespace smith::distributed::rpc
