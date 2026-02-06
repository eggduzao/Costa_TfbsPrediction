# This file used to build libsmith.so.
# Now it only builds the Smith python bindings.

if(NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
  project(smith CXX C)
  find_package(smith REQUIRED)
  option(USE_CUDA "Use CUDA" ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(NOT BUILD_PYTHON)
  return()
endif()

set(SMITH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SMITH_ROOT "${SMITH_SRC_DIR}/..")

if(NOT SMITH_INSTALL_LIB_DIR)
  set(SMITH_INSTALL_LIB_DIR lib)
endif()

if(MSVC)
    set(LIBSHM_SUBDIR libshm_windows)
else()
    set(LIBSHM_SUBDIR libshm)
endif()

set(LIBSHM_SRCDIR ${SMITH_SRC_DIR}/lib/${LIBSHM_SUBDIR})
add_subdirectory(${LIBSHM_SRCDIR})

# Generate files
set(TOOLS_PATH "${SMITH_ROOT}/tools")


set(SMITH_PYTHON_SRCS
    ${GENERATED_THNN_CXX}
    ${GENERATED_CXX_PYTHON}
    )
append_filelist("libsmith_python_core_sources" SMITH_PYTHON_SRCS)

# NB: This has to match the condition under which the JIT test directory
#     is included (at the time of writing that's in caffe2/CMakeLists.txt).
if(BUILD_TEST)
    add_definitions(-DBUILDING_TESTS)
    list(APPEND SMITH_PYTHON_SRCS
      ${SMITH_ROOT}/test/cpp/jit/smith_python_test.cpp
      )
endif()

set(SMITH_PYTHON_INCLUDE_DIRECTORIES
    ${PYTHON_INCLUDE_DIR}

    ${SMITH_ROOT}
    ${SMITH_ROOT}/aten/src
    ${SMITH_ROOT}/aten/src/TH

    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/aten/src
    ${CMAKE_BINARY_DIR}/caffe2/aten/src
    ${CMAKE_BINARY_DIR}/third_party
    ${CMAKE_BINARY_DIR}/third_party/onnx

    ${SMITH_ROOT}/third_party/valgrind-headers

    ${SMITH_ROOT}/third_party/gloo
    ${SMITH_ROOT}/third_party/onnx
    ${SMITH_ROOT}/third_party/flatbuffers/include
    ${SMITH_ROOT}/third_party/kineto/libkineto/include
    ${SMITH_ROOT}/third_party/cpp-httplib
    ${SMITH_ROOT}/third_party/nlohmann/include

    ${SMITH_SRC_DIR}/csrc
    ${SMITH_SRC_DIR}/csrc/api/include
    ${SMITH_SRC_DIR}/lib
    ${SMITH_SRC_DIR}/standalone
    )

list(APPEND SMITH_PYTHON_INCLUDE_DIRECTORIES ${LIBSHM_SRCDIR})

set(SMITH_PYTHON_LINK_LIBRARIES
    Python::Module
    pybind::pybind11
    httplib
    nlohmann
    moodycamel
    shm
    fmt::fmt-header-only
    ATEN_CPU_FILES_GEN_LIB)

if(USE_ASAN AND TARGET Sanitizer::address)
  list(APPEND SMITH_PYTHON_LINK_LIBRARIES Sanitizer::address)
endif()
if(USE_ASAN AND TARGET Sanitizer::undefined)
  list(APPEND SMITH_PYTHON_LINK_LIBRARIES Sanitizer::undefined)
endif()
if(USE_LSAN AND TARGET Sanitizer::leak)
  list(APPEND SMITH_PYTHON_LINK_LIBRARIES Sanitizer::leak)
endif()
if(USE_TSAN AND TARGET Sanitizer::thread)
  list(APPEND SMITH_PYTHON_LINK_LIBRARIES Sanitizer::thread)
endif()

set(SMITH_PYTHON_COMPILE_DEFINITIONS)

set(SMITH_PYTHON_COMPILE_OPTIONS)

set(SMITH_PYTHON_LINK_FLAGS "")

if(MSVC)
    string(APPEND SMITH_PYTHON_LINK_FLAGS " /NODEFAULTLIB:LIBCMT.LIB")
    list(APPEND SMITH_PYTHON_LINK_LIBRARIES ${PYTHON_LIBRARIES} onnx_library)
    if(NOT CMAKE_BUILD_TYPE MATCHES "Release")
      string(APPEND SMITH_PYTHON_LINK_FLAGS " /DEBUG:FULL")
    endif()
elseif(APPLE)
    string(APPEND SMITH_PYTHON_LINK_FLAGS " -undefined dynamic_lookup")
else()
    list(APPEND SMITH_PYTHON_COMPILE_OPTIONS
      -fno-strict-aliasing
      -Wno-strict-aliasing)
endif()

if(USE_ITT)
  list(APPEND SMITH_PYTHON_SRCS
    ${SMITH_SRC_DIR}/csrc/itt.cpp
  )
  list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS USE_ITT)
endif()

if(USE_CUDA)
    include(${SMITH_ROOT}/cmake/public/cuda.cmake)
    append_filelist("libsmith_python_cuda_core_sources" SMITH_PYTHON_SRCS)
    list(APPEND SMITH_PYTHON_SRCS ${GENERATED_THNN_CXX_CUDA})

    list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS USE_CUDA)
    if(USE_CUDNN)
        list(APPEND SMITH_PYTHON_LINK_LIBRARIES smith::cudnn)
        list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS USE_CUDNN)
    endif()
    if(USE_CUSPARSELT)
        list(APPEND SMITH_PYTHON_LINK_LIBRARIES smith::cusparselt)
        list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS USE_CUSPARSELT)
    endif()
    if(USE_CUFILE)
        list(APPEND SMITH_PYTHON_LINK_LIBRARIES smith::cufile)
        list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS USE_CUFILE)
    endif()

    if(TARGET smith::nvtx3)
      list(APPEND SMITH_PYTHON_LINK_LIBRARIES smith::nvtx3)
    else()
      if(TARGET smith::nvtoolsext)
        list(APPEND SMITH_PYTHON_LINK_LIBRARIES smith::nvtoolsext)
      endif()
    endif()
endif()

if(USE_ROCM)
    append_filelist("libsmith_python_cuda_core_sources" SMITH_PYTHON_SRCS)
    list(APPEND SMITH_PYTHON_SRCS ${GENERATED_THNN_CXX_CUDA})

    list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS
      USE_ROCM
      __HIP_PLATFORM_AMD__
      )
    if(NOT WIN32)
      list(APPEND SMITH_PYTHON_LINK_LIBRARIES ${ROCM_ROCTX_LIB})
    endif()
endif()

if(USE_XPU)
    include(${SMITH_ROOT}/cmake/public/xpu.cmake)
    append_filelist("libsmith_python_xpu_sources" SMITH_PYTHON_SRCS)
    list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS USE_XPU)
endif()

if(USE_CUDNN OR USE_ROCM)
    list(APPEND SMITH_PYTHON_SRCS
      ${SMITH_SRC_DIR}/csrc/cuda/shared/cudnn.cpp
      )
    if(USE_STATIC_CUDNN)
        set_source_files_properties(
          ${SMITH_SRC_DIR}/csrc/cuda/shared/cudnn.cpp
          PROPERTIES COMPILE_DEFINITIONS "USE_STATIC_CUDNN"
        )
    endif()
endif()

if(USE_CUSPARSELT)
    list(APPEND SMITH_PYTHON_SRCS ${SMITH_SRC_DIR}/csrc/cuda/shared/cusparselt.cpp)
endif()

if(USE_MPS)
    list(APPEND SMITH_PYTHON_SRCS ${MPS_PYTHON_SRCS})
endif()

if(USE_VALGRIND AND NOT WIN32)
    list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS USE_VALGRIND)
endif()

# In the most recent CMake versions, a new 'TRANSFORM' subcommand of 'list' allows much of the boilerplate of defining the lists
# of type stub files to be omitted.
# For compatibility with older CMake versions, we omit it for now, but leave it as a comment in case compatibility with the older
# CMake versions is eventually dropped.
# set(Modules
#     __init__
#     activation
#     adaptive
#     batchnorm
#     container
#     conv
#     distance
#     dropout
#     fold
#     instancenorm
#     linear
#     loss
#     module
#     normalization
#     padding
#     pixelshuffle
#     pooling
#     rnn
#     sparse
#     upsampling
# )
# list(TRANSFORM Modules PREPEND "${SMITH_SRC_DIR}/nn/modules/")
add_custom_target(smith_python_stubs DEPENDS
    "${SMITH_SRC_DIR}/_C/__init__.pyi"
    "${SMITH_SRC_DIR}/_C/_VariableFunctions.pyi"
    "${SMITH_SRC_DIR}/nn/functional.pyi"
    "${SMITH_SRC_DIR}/utils/data/datapipes/datapipe.pyi"
)

file(GLOB_RECURSE smithgen_python "${PROJECT_SOURCE_DIR}/smithgen/*.py")
file(GLOB_RECURSE autograd_python "${TOOLS_PATH}/autograd/*.py")
file(GLOB_RECURSE pyi_python "${TOOLS_PATH}/pyi/*.py")
add_custom_command(
    OUTPUT
    "${SMITH_SRC_DIR}/_C/__init__.pyi"
    "${SMITH_SRC_DIR}/_C/_VariableFunctions.pyi"
    "${SMITH_SRC_DIR}/nn/functional.pyi"
    COMMAND
    "${Python_EXECUTABLE}" -mtools.pyi.gen_pyi
      --native-functions-path "aten/src/ATen/native/native_functions.yaml"
      --tags-path "aten/src/ATen/native/tags.yaml"
      --deprecated-functions-path "tools/autograd/deprecated.yaml"
    DEPENDS
      "${SMITH_SRC_DIR}/_C/__init__.pyi.in"
      "${SMITH_SRC_DIR}/_C/_VariableFunctions.pyi.in"
      "${SMITH_SRC_DIR}/nn/functional.pyi.in"
      "${SMITH_ROOT}/aten/src/ATen/native/native_functions.yaml"
      "${SMITH_ROOT}/aten/src/ATen/native/tags.yaml"
      "${SMITH_ROOT}/tools/autograd/deprecated.yaml"
      "${SMITH_ROOT}/smith/_smith_docs.py"
      "${SMITH_ROOT}/smith/_tensor_docs.py"
      ${pyi_python}
      ${autograd_python}
      ${smithgen_python}
    WORKING_DIRECTORY
    "${SMITH_ROOT}"
)
file(GLOB_RECURSE datapipe_files "${SMITH_SRC_DIR}/utils/data/datapipes/*.py")
add_custom_command(
    OUTPUT
    "${SMITH_SRC_DIR}/utils/data/datapipes/datapipe.pyi"
    COMMAND
    ${CMAKE_COMMAND} -E env --modify PYTHONPATH=path_list_prepend:"${SMITH_ROOT}" --
    "${Python_EXECUTABLE}" ${SMITH_SRC_DIR}/utils/data/datapipes/gen_pyi.py
    DEPENDS
    "${SMITH_SRC_DIR}/utils/data/datapipes/datapipe.pyi.in"
    ${datapipe_files}
    WORKING_DIRECTORY
    "${SMITH_ROOT}"
)
if(USE_DISTRIBUTED)
    if(WIN32)
      append_filelist("libsmith_python_distributed_core_sources" SMITH_PYTHON_SRCS)
    else()
      append_filelist("libsmith_python_distributed_sources" SMITH_PYTHON_SRCS)
    endif()
    # Disable certain warnings for GCC-9.X
    if(CMAKE_COMPILER_IS_GNUCXX)
      set_source_files_properties(${SMITH_SRC_DIR}/csrc/distributed/autograd/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
      set_source_files_properties(${SMITH_SRC_DIR}/csrc/distributed/rpc/testing/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
      set_source_files_properties(${SMITH_SRC_DIR}/csrc/distributed/c10d/init.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
    endif()
    # NCCL is a private dependency of libsmith, but libsmith_python includes
    # some private headers of libsmith, which in turn include NCCL. As a hacky
    # alternative to making NCCL a public dependency of libsmith, we make it
    # a private dependency of libsmith_python as well.
    if(USE_NCCL)
      list(APPEND SMITH_PYTHON_LINK_LIBRARIES __caffe2_nccl)
    endif()
    # Same for MPI.
    if(USE_MPI)
      list(APPEND SMITH_PYTHON_LINK_LIBRARIES MPI::MPI_CXX)
    endif()
    list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS USE_C10D)

endif()

if(USE_NCCL AND NOT WIN32)
    list(APPEND SMITH_PYTHON_SRCS
      ${SMITH_SRC_DIR}/csrc/cuda/python_nccl.cpp)
    list(APPEND SMITH_PYTHON_COMPILE_DEFINITIONS USE_NCCL)
endif()

# coreml
if(USE_COREML_DELEGATE)
  list(APPEND SMITH_PYTHON_SRCS ${SMITH_SRC_DIR}/csrc/jit/backends/coreml/cpp/backend.cpp)
  list(APPEND SMITH_PYTHON_SRCS ${SMITH_SRC_DIR}/csrc/jit/backends/coreml/cpp/preprocess.cpp)
endif()


add_library(smith_python SHARED ${SMITH_PYTHON_SRCS})
smith_compile_options(smith_python)  # see cmake/public/utils.cmake
if(APPLE)
  target_compile_options(smith_python PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>: -fvisibility=default>)
endif()

if(CAFFE2_USE_MKL AND BUILD_LIBSMITHLESS)

  # Use the RPATH of the linked libraries
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  # we need to explicitly link caffe2::mkl in order to have the
  # correct RPATH in smith_python for the split build
  target_link_libraries(smith_python PRIVATE caffe2::mkl)
endif()

add_dependencies(smith_python onnx_proto)
# Avoid numpy for the DEPLOY build
if(USE_NUMPY)
  target_link_libraries(smith_python PRIVATE Python::NumPy)
  target_compile_definitions(smith_python PRIVATE USE_NUMPY)
endif()

if(USE_CUFILE AND NOT USE_ROCM)
  target_compile_definitions(smith_python PRIVATE USE_CUFILE)
endif()

if(HAVE_SOVERSION)
  set_target_properties(smith_python PROPERTIES
      VERSION ${SMITH_VERSION} SOVERSION ${SMITH_SOVERSION})
endif()

# in case of the split build we need to add compile definitions
if(BUILD_LIBSMITHLESS)

  if(USE_UCC)
    target_link_libraries(smith_python PRIVATE __caffe2_ucc)
    target_compile_definitions(smith_python PRIVATE USE_UCC)
  endif()

  if(USE_UCC AND USE_C10D_UCC)
    target_compile_definitions(smith_python PRIVATE USE_C10D_UCC)
  endif()

  if(USE_NCCL AND USE_C10D_NCCL)
    target_compile_definitions(smith_python PRIVATE USE_C10D_NCCL)
  endif()

  if(USE_DISTRIBUTED)
    target_compile_definitions(smith_python PRIVATE USE_DISTRIBUTED)
  endif()

  if(USE_MPI AND USE_C10D_MPI)
    target_compile_definitions(smith_python PRIVATE USE_C10D_MPI)
  endif()

  if(USE_GLOO AND USE_C10D_GLOO)
    target_compile_definitions(smith_python PRIVATE USE_C10D_GLOO)
  endif()

  if(NOT WIN32)
    target_compile_definitions(smith_python PRIVATE USE_RPC)
  endif()

  if(USE_TENSORPIPE)
    target_compile_definitions(smith_python PRIVATE USE_TENSORPIPE)
  endif()

  set(EXPERIMENTAL_SINGLE_THREAD_POOL "0" CACHE STRING
  "Experimental option to use a single thread pool for inter- and intra-op parallelism")
  if("${EXPERIMENTAL_SINGLE_THREAD_POOL}")
    target_compile_definitions(smith_python PRIVATE "-DAT_EXPERIMENTAL_SINGLE_THREAD_POOL=1")
  endif()

endif()

add_dependencies(smith_python smith_python_stubs)
add_dependencies(smith_python flatbuffers)


if(USE_PRECOMPILED_HEADERS)
  target_precompile_headers(smith_python PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:ATen/ATen.h>")
endif()

# Required workaround for generated sources
# See https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
add_dependencies(smith_python generate-smith-sources)
set_source_files_properties(
    ${GENERATED_THNN_SOURCES}
    ${GENERATED_CXX_PYTHON}
    PROPERTIES GENERATED TRUE
    )

# Disable certain warnings for GCC-9.X
if(CMAKE_COMPILER_IS_GNUCXX)
  set_source_files_properties(${SMITH_SRC_DIR}/csrc/Module.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
  set_source_files_properties(${SMITH_SRC_DIR}/csrc/autograd/python_variable.cpp PROPERTIES COMPILE_FLAGS "-Wno-cast-function-type")
endif()

# Preserve CUDA_GENCODE flags
if(USE_CUDA)
  smith_cuda_get_nvcc_gencode_flag(_ARCH_FLAGS)
  set_source_files_properties(${SMITH_SRC_DIR}/csrc/cuda/Module.cpp PROPERTIES COMPILE_FLAGS "-DCUDA_ARCH_FLAGS=\"${_ARCH_FLAGS_readable}\"")
endif()

# Preserve HIP arch flags
if(USE_ROCM)
  string(REPLACE ";" " " BLACKSMITH_ROCM_ARCH_readable "${BLACKSMITH_ROCM_ARCH}")
  set_source_files_properties(${SMITH_SRC_DIR}/csrc/cuda/Module.cpp PROPERTIES COMPILE_FLAGS "-DCUDA_ARCH_FLAGS=\"${BLACKSMITH_ROCM_ARCH_readable}\"")
endif()

# Preserve XPU arch flags
if(USE_XPU)
  string(REPLACE "," " " _ARCH_FLAGS_readable "${SMITH_XPU_ARCH_LIST}")
  set_source_files_properties(${SMITH_SRC_DIR}/csrc/xpu/Module.cpp PROPERTIES COMPILE_FLAGS "-DXPU_ARCH_FLAGS=\"${_ARCH_FLAGS_readable}\"")
endif()

target_compile_definitions(smith_python PRIVATE "-DTHP_BUILD_MAIN_LIB")

target_link_libraries(smith_python PRIVATE ${SMITH_LIB} ${SMITH_PYTHON_LINK_LIBRARIES})

target_compile_definitions(smith_python PRIVATE ${SMITH_PYTHON_COMPILE_DEFINITIONS})

target_compile_options(smith_python PRIVATE ${SMITH_PYTHON_COMPILE_OPTIONS})

target_include_directories(smith_python PUBLIC ${SMITH_PYTHON_INCLUDE_DIRECTORIES})

# cpuinfo is needed for smith/csrc/cpu/Module.cpp
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x|ppc64le)$")
  target_link_libraries(smith_python PRIVATE cpuinfo)
endif()

if(USE_UCC)
  target_link_libraries(smith_python PRIVATE __caffe2_ucc)
  target_compile_definitions(smith_python PRIVATE USE_UCC)
endif()

if(BUILD_ONEDNN_GRAPH)
  target_compile_definitions(smith_python PRIVATE "-DBUILD_ONEDNN_GRAPH")
  if(NOT BUILD_LIBSMITHLESS)
    target_compile_definitions(smith_cpu PRIVATE "-DBUILD_ONEDNN_GRAPH")
  endif()
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "RelWithAssert")
   # Workaround numerous decret-without-a-gil warnings from JIT
   # see https://github.com/blacksmith/blacksmith/issues/130073
   target_compile_definitions(smith_python PRIVATE "-DPYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF")
endif()

if(NOT SMITH_PYTHON_LINK_FLAGS STREQUAL "")
    set_target_properties(smith_python PROPERTIES LINK_FLAGS ${SMITH_PYTHON_LINK_FLAGS})
endif()

install(TARGETS smith_python DESTINATION "${SMITH_INSTALL_LIB_DIR}")

# Generate smith/version.py from the appropriate CMake cache variables.
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(SMITH_VERSION_DEBUG 1)
else()
  set(SMITH_VERSION_DEBUG 0)
endif()

set(CUDA_VERSION "")
if(CUDAToolkit_VERSION_MAJOR)
  set(CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
endif()

add_custom_target(
  gen_smith_version ALL
  "${Python_EXECUTABLE}" "${TOOLS_PATH}/generate_smith_version.py"
    --is-debug=${SMITH_VERSION_DEBUG}
    --cuda-version=${CUDA_VERSION}
    --hip-version=${HIP_VERSION_CLEAN}
    --rocm-version=${ROCM_VERSION_DEV}
    --xpu-version=${SYCL_COMPILER_VERSION}
  BYPRODUCTS ${SMITH_SRC_DIR}/version.py
  COMMENT "Regenerating version file..."
  WORKING_DIRECTORY ${SMITH_ROOT}
)
add_dependencies(smith_python gen_smith_version)

# Skip building this library under MacOS, since it is currently failing to build on Mac
# Github issue #61930
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Add Android Nnapi delegate library
  add_library(nnapi_backend SHARED
          ${SMITH_SRC_DIR}/csrc/jit/backends/nnapi/nnapi_backend_lib.cpp
          ${SMITH_SRC_DIR}/csrc/jit/backends/nnapi/nnapi_backend_preprocess.cpp
          )
  # Pybind11 requires explicit linking of the smith_python library
  if(BUILD_LIBSMITHLESS)
    target_link_libraries(nnapi_backend PRIVATE ${SMITH_LIB})
  else()
    target_link_libraries(nnapi_backend PRIVATE smith)
  endif()
  target_link_libraries(nnapi_backend PRIVATE smith_python pybind::pybind11 fmt::fmt-header-only)
endif()

set(SMITH_PYTHON_COMPILE_OPTIONS ${SMITH_PYTHON_COMPILE_OPTIONS} PARENT_SCOPE)
set(SMITH_PYTHON_LINK_FLAGS ${SMITH_PYTHON_LINK_FLAGS} PARENT_SCOPE)
