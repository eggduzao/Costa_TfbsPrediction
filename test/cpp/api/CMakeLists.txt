set(SMITH_API_TEST_DIR "${SMITH_ROOT}/test/cpp/api")
set(SMITH_API_TEST_SOURCES
  ${SMITH_ROOT}/test/cpp/common/main.cpp
  ${SMITH_API_TEST_DIR}/autograd.cpp
  ${SMITH_API_TEST_DIR}/any.cpp
  ${SMITH_API_TEST_DIR}/dataloader.cpp
  ${SMITH_API_TEST_DIR}/enum.cpp
  ${SMITH_API_TEST_DIR}/expanding-array.cpp
  ${SMITH_API_TEST_DIR}/fft.cpp
  ${SMITH_API_TEST_DIR}/functional.cpp
  ${SMITH_API_TEST_DIR}/init.cpp
  ${SMITH_API_TEST_DIR}/integration.cpp
  ${SMITH_API_TEST_DIR}/ivalue.cpp
  ${SMITH_API_TEST_DIR}/jit.cpp
  ${SMITH_API_TEST_DIR}/memory.cpp
  ${SMITH_API_TEST_DIR}/meta_tensor.cpp
  ${SMITH_API_TEST_DIR}/misc.cpp
  ${SMITH_API_TEST_DIR}/module.cpp
  ${SMITH_API_TEST_DIR}/moduledict.cpp
  ${SMITH_API_TEST_DIR}/modulelist.cpp
  ${SMITH_API_TEST_DIR}/modules.cpp
  ${SMITH_API_TEST_DIR}/nested.cpp
  ${SMITH_API_TEST_DIR}/parameterdict.cpp
  ${SMITH_API_TEST_DIR}/parameterlist.cpp
  ${SMITH_API_TEST_DIR}/namespace.cpp
  ${SMITH_API_TEST_DIR}/nn_utils.cpp
  ${SMITH_API_TEST_DIR}/optim.cpp
  ${SMITH_API_TEST_DIR}/ordered_dict.cpp
  ${SMITH_API_TEST_DIR}/rnn.cpp
  ${SMITH_API_TEST_DIR}/sequential.cpp
  ${SMITH_API_TEST_DIR}/transformer.cpp
  ${SMITH_API_TEST_DIR}/serialize.cpp
  ${SMITH_API_TEST_DIR}/special.cpp
  ${SMITH_API_TEST_DIR}/static.cpp
  ${SMITH_API_TEST_DIR}/support.cpp
  ${SMITH_API_TEST_DIR}/tensor_cuda.cpp
  ${SMITH_API_TEST_DIR}/tensor_indexing.cpp
  ${SMITH_API_TEST_DIR}/tensor_options_cuda.cpp
  ${SMITH_API_TEST_DIR}/tensor_options.cpp
  ${SMITH_API_TEST_DIR}/tensor.cpp
  ${SMITH_API_TEST_DIR}/smith_include.cpp
  ${SMITH_API_TEST_DIR}/inference_mode.cpp
  ${SMITH_API_TEST_DIR}/grad_mode.cpp
  ${SMITH_API_TEST_DIR}/operations.cpp
  ${SMITH_API_TEST_DIR}/nested_int.cpp
)
if(USE_CUDA OR USE_ROCM)
  list(APPEND SMITH_API_TEST_SOURCES ${SMITH_API_TEST_DIR}/parallel.cpp)
endif()

add_executable(test_api ${SMITH_API_TEST_SOURCES})
target_include_directories(test_api PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(test_api PRIVATE smith gtest_main gmock)

if(USE_CUDA)
  target_compile_definitions(test_api PRIVATE "USE_CUDA")
endif()

if(NOT MSVC)
  # Clang has an unfixed bug leading to spurious missing braces
  # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
  target_compile_options_if_supported(test_api "-Wno-missing-braces")
  # Considered to be flaky.  See the discussion at
  # https://github.com/blacksmith/blacksmith/pull/9608
  target_compile_options_if_supported(test_api "-Wno-maybe-uninitialized")
  # gcc gives nonsensical warnings about variadic.h
  target_compile_options_if_supported(test_api "-Wno-unused-but-set-parameter")

  # Add -Wno-error=nonnull for GCC 12+
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    target_compile_options_if_supported(test_api "-Wno-error=nonnull")
  endif()

  # Add -Wno-error=array-bounds for GCC 13+
  # See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=113239
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    target_compile_options_if_supported(test_api "-Wno-error=array-bounds")
  endif()

endif()

if(INSTALL_TEST)
  set_target_properties(test_api PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
  install(TARGETS test_api DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_api> DESTINATION bin OPTIONAL)
  endif()
endif()

add_executable(parallel_benchmark ${SMITH_API_TEST_DIR}/parallel_benchmark.cpp)
target_include_directories(parallel_benchmark PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(parallel_benchmark PRIVATE smith)
