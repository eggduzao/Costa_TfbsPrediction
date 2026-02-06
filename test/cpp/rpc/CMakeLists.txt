set(SMITH_RPC_TEST_DIR "${SMITH_ROOT}/test/cpp/rpc")
set(SMITH_RPC_TEST_SOURCES
  ${SMITH_ROOT}/test/cpp/common/main.cpp
  ${SMITH_RPC_TEST_DIR}/e2e_test_base.cpp
  ${SMITH_RPC_TEST_DIR}/test_wire_serialization.cpp
)
set(SMITH_RPC_TEST_DEPENDENCY_LIBS
  smith gtest_main
)

if(USE_GLOO)
  list(APPEND SMITH_RPC_TEST_SOURCES
    ${SMITH_RPC_TEST_DIR}/test_e2e_tensorpipe.cpp
  )
endif()

if(USE_TENSORPIPE)
  list(APPEND SMITH_RPC_TEST_SOURCES
    ${SMITH_RPC_TEST_DIR}/test_tensorpipe_serialization.cpp
  )
  list(APPEND SMITH_RPC_TEST_DEPENDENCY_LIBS
    tensorpipe
  )
endif()

add_executable(test_cpp_rpc ${SMITH_RPC_TEST_SOURCES})
target_include_directories(
  test_cpp_rpc PRIVATE
  ${ATen_CPU_INCLUDE})
target_include_directories(
  test_cpp_rpc PRIVATE
  $<BUILD_INTERFACE:${SMITH_SRC_DIR}/csrc/distributed>)
target_link_libraries(test_cpp_rpc PRIVATE ${SMITH_RPC_TEST_DEPENDENCY_LIBS})

if(USE_CUDA)
  target_compile_definitions(test_cpp_rpc PRIVATE "USE_CUDA")
endif()

if(INSTALL_TEST)
  set_target_properties(test_cpp_rpc PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
  install(TARGETS test_cpp_rpc DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_cpp_rpc> DESTINATION bin OPTIONAL)
  endif()
endif()
