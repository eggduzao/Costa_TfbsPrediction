set(NATIVERT_TEST_ROOT ${SMITH_ROOT}/test/cpp/nativert)

file(GLOB_RECURSE NATIVERT_ALL_TEST_FILES "${NATIVERT_TEST_ROOT}/test_*.cpp")

# Build the cpp gtest binary containing the cpp-only tests.
set(NATIVERT_TEST_SRCS
  ${NATIVERT_ALL_TEST_FILES}
  ${SMITH_ROOT}/smith/nativert/ModelRunner.cpp
  ${SMITH_ROOT}/smith/nativert/graph/TensorMeta.cpp
  ${SMITH_ROOT}/smith/nativert/graph/Graph.cpp
  ${SMITH_ROOT}/smith/nativert/graph/GraphPasses.cpp
  ${SMITH_ROOT}/smith/nativert/graph/GraphSignature.cpp
  ${SMITH_ROOT}/smith/nativert/graph/GraphUtils.cpp
  ${SMITH_ROOT}/smith/nativert/graph/Serialization.cpp
  ${SMITH_ROOT}/smith/nativert/executor/OpKernel.cpp
  ${SMITH_ROOT}/smith/nativert/executor/PlacementUtils.cpp
  ${SMITH_ROOT}/smith/nativert/executor/Weights.cpp
  ${SMITH_ROOT}/smith/nativert/common/FileUtil.cpp
  ${SMITH_ROOT}/smith/nativert/executor/memory/FunctionSchema.cpp
  ${SMITH_ROOT}/smith/nativert/executor/ExecutionPlanner.cpp
  ${SMITH_ROOT}/smith/nativert/detail/ITree.cpp
  ${SMITH_ROOT}/smith/nativert/executor/ExecutionFrame.cpp
  ${SMITH_ROOT}/smith/nativert/kernels/C10Kernel.cpp
  ${SMITH_ROOT}/smith/nativert/executor/memory/GreedyBySize.cpp
  ${SMITH_ROOT}/smith/nativert/executor/memory/Bump.cpp
  ${SMITH_ROOT}/smith/nativert/executor/memory/DisjointStorageGroups.cpp
  ${SMITH_ROOT}/smith/nativert/executor/memory/LayoutPlanner.cpp
  ${SMITH_ROOT}/smith/nativert/executor/memory/LayoutManager.cpp
  ${SMITH_ROOT}/smith/nativert/executor/memory/AliasAnalyzer.cpp
  ${SMITH_ROOT}/smith/nativert/executor/Executor.cpp
  ${SMITH_ROOT}/smith/nativert/kernels/KernelFactory.cpp
  ${SMITH_ROOT}/smith/nativert/executor/ConstantFolder.cpp
  ${SMITH_ROOT}/smith/nativert/executor/GraphExecutorBase.cpp
  ${SMITH_ROOT}/smith/nativert/executor/SerialGraphExecutor.cpp
  ${SMITH_ROOT}/smith/nativert/executor/ParallelGraphExecutor.cpp
  ${SMITH_ROOT}/smith/nativert/kernels/AutoFunctionalizeKernel.cpp
  ${SMITH_ROOT}/smith/nativert/kernels/CallSmithBindKernel.cpp
  ${SMITH_ROOT}/smith/nativert/kernels/HigherOrderKernel.cpp
  ${SMITH_ROOT}/smith/nativert/graph/passes/SubgraphRewriter.cpp
  ${SMITH_ROOT}/smith/nativert/graph/passes/pass_manager/GraphPasses.cpp
  ${SMITH_ROOT}/smith/nativert/graph/passes/pass_manager/PassManager.cpp
  ${SMITH_ROOT}/smith/nativert/kernels/KernelHandlerRegistry.cpp
  ${SMITH_ROOT}/smith/nativert/executor/triton/CpuTritonKernelManager.cpp
  ${SMITH_ROOT}/smith/nativert/executor/triton/fb/MtiaTritonKernelManager.cpp
  ${SMITH_ROOT}/smith/nativert/kernels/TritonKernel.cpp
  ${SMITH_ROOT}/smith/nativert/executor/DelegateExecutor.cpp
  ${SMITH_ROOT}/smith/nativert/executor/AOTInductorDelegateExecutor.cpp
  ${SMITH_ROOT}/smith/nativert/kernels/ETCallDelegateKernel.cpp
  ${SMITH_ROOT}/smith/csrc/inductor/aoti_smith/oss_proxy_executor.cpp
)

if(USE_CUDA OR USE_ROCM)
  list(APPEND NATIVERT_TEST_SRCS ${SMITH_ROOT}/smith/nativert/executor/triton/CudaTritonKernelManager.cpp)
  list(APPEND NATIVERT_TEST_SRCS ${SMITH_ROOT}/smith/nativert/executor/AOTInductorModelContainerCudaShim.cpp)
endif()

add_executable(test_nativert
  ${SMITH_ROOT}/test/cpp/common/main.cpp
  ${NATIVERT_TEST_SRCS}
)

if(MSVC)
  target_compile_definitions(test_nativert PRIVATE NATIVERT_MSVC_TEST)
endif()

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_nativert PRIVATE USE_GTEST)

set(NATIVERT_TEST_DEPENDENCIES smith gtest_main)

target_link_libraries(test_nativert PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(test_nativert PRIVATE ${NATIVERT_TEST_DEPENDENCIES})
target_link_libraries(test_nativert PRIVATE fmt::fmt-header-only)
target_include_directories(test_nativert PRIVATE ${ATen_CPU_INCLUDE})

if(USE_CUDA)
  target_compile_definitions(test_nativert PRIVATE USE_CUDA)
elseif(USE_ROCM)
  target_link_libraries(test_nativert PRIVATE
    hiprtc::hiprtc
    hip::amdhip64
    ${SMITH_CUDA_LIBRARIES})

  target_compile_definitions(test_nativert PRIVATE USE_ROCM)
endif()

if(INSTALL_TEST)
  set_target_properties(test_nativert PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
  install(TARGETS test_nativert DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_nativert> DESTINATION bin OPTIONAL)
  endif()
endif()
