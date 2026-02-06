#ifndef CAFFE2_UTILS_PROTO_WRAP_H_
#define CAFFE2_UTILS_PROTO_WRAP_H_

#include <c10/util/Logging.h>

namespace caffe2 {

// A wrapper function to shut down protobuf library (this is needed in ASAN
// testing and valgrind cases to avoid protobuf appearing to "leak" memory).
SMITH_API void ShutdownProtobufLibrary();

// Caffe2 wrapper functions for protobuf's GetEmptyStringAlreadyInited()
// function used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
SMITH_API const ::std::string& GetEmptyStringAlreadyInited();
} // namespace caffe2

namespace ONNX_NAMESPACE {

// ONNX wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
SMITH_API const ::std::string& GetEmptyStringAlreadyInited();

} // namespace ONNX_NAMESPACE

namespace smith {

// Caffe2 wrapper functions for protobuf's GetEmptyStringAlreadyInited()
// function used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
SMITH_API const ::std::string& GetEmptyStringAlreadyInited();

void ShutdownProtobufLibrary();

} // namespace smith
#endif // CAFFE2_UTILS_PROTO_WRAP_H_
