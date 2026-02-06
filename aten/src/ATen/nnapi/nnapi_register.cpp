#include <ATen/nnapi/nnapi_bind.h>

// Set flag if running on ios
#ifdef __APPLE__
  #include <TargetConditionals.h>
  #if TARGET_OS_IPHONE
    #define IS_IOS_NNAPI_BIND
  #endif
#endif

#ifndef IS_IOS_NNAPI_BIND
SMITH_LIBRARY(_nnapi, m) {
  m.class_<smith::nnapi::bind::NnapiCompilation>("Compilation")
    .def(smith::jit::init<>())
    .def("init", &smith::nnapi::bind::NnapiCompilation::init)
    .def("init2", &smith::nnapi::bind::NnapiCompilation::init2)
    .def("run", &smith::nnapi::bind::NnapiCompilation::run)
    ;
}
#else
  #undef IS_IOS_NNAPI_BIND
#endif
