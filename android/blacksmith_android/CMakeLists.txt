cmake_minimum_required(VERSION 3.5)
option(BUILD_LITE_INTERPRETER "Master flag to build blacksmith_jni_lite" ON)
message(
  STATUS
  "BUILD_LITE_INTERPRETER (blacksmith_jni_lite): ${BUILD_LITE_INTERPRETER}")

if(BUILD_LITE_INTERPRETER)
  project(blacksmith_jni_lite CXX)
  set(BLACKSMITH_JNI_TARGET blacksmith_jni_lite)
else()
  project(blacksmith_jni CXX)
  set(BLACKSMITH_JNI_TARGET blacksmith_jni)
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_VERBOSE_MAKEFILE ON)
message(STATUS "ANDROID_STL:${ANDROID_STL}")

set(TRACE_ENABLED OFF)
if(DEFINED ENV{TRACE_ENABLED})
  if($ENV{TRACE_ENABLED} STREQUAL "1")
    message(STATUS "TRACE_ENABLED ON")
    set(TRACE_ENABLED ON)
  endif()
endif()
if(NOT TRACE_ENABLED)
  message(STATUS "TRACE_ENABLED OFF")
endif()

set(USE_VULKAN OFF)

set(blacksmith_android_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)

if(ANDROID_ABI)
  set(USE_VULKAN ON)
  set(libsmith_include_DIR ${blacksmith_android_DIR}/libsmith_include/${ANDROID_ABI})
  set(BUILD_SUBDIR ${ANDROID_ABI})
elseif(BUILD_LIBSMITH_WITH_JNI)
  # Don't need LIBSMITH_HOME if we're building from within Blacksmith.
else()
  # Building against a pre-built libsmith.
  if(NOT LIBSMITH_HOME)
    message(FATAL_ERROR
      "blacksmith_android requires LIBSMITH_HOME to be defined for non-Android builds.")
  endif()
  set(libsmith_include_DIR ${LIBSMITH_HOME}/include)
  link_directories(${LIBSMITH_HOME}/lib)
  set(BUILD_SUBDIR host)
endif()

message(STATUS "libsmith dir:${libsmith_DIR}")

configure_file(
    ${blacksmith_android_DIR}/cmake_macros.h.in
    ${blacksmith_android_DIR}/cmake_macros.h)


if(BUILD_LITE_INTERPRETER)
  file(GLOB blacksmith_android_SOURCES
    ${blacksmith_android_DIR}/blacksmith_jni_lite.cpp
    ${blacksmith_android_DIR}/blacksmith_jni_common.cpp
    ${blacksmith_android_DIR}/blacksmith_jni_common.h
  )
else()
  file(GLOB blacksmith_android_SOURCES
    ${blacksmith_android_DIR}/blacksmith_jni_jit.cpp
    ${blacksmith_android_DIR}/blacksmith_jni_common.cpp
    ${blacksmith_android_DIR}/blacksmith_jni_common.h
  )
endif()
add_library(${BLACKSMITH_JNI_TARGET} SHARED ${blacksmith_android_SOURCES})

if(APPLE)
  # Need to add rpath so dlopen can find dependencies.
  add_custom_command(TARGET blacksmith_jni
      POST_BUILD COMMAND
      ${CMAKE_INSTALL_NAME_TOOL} -add_rpath "@loader_path"
        $<TARGET_FILE:blacksmith_jni>)
endif()

target_compile_options(${BLACKSMITH_JNI_TARGET} PRIVATE
  -fexceptions
)
target_include_directories(${BLACKSMITH_JNI_TARGET} BEFORE
PUBLIC $<BUILD_INTERFACE:${libsmith_include_DIR}>)

set(fbjni_DIR ${CMAKE_CURRENT_LIST_DIR}/../libs/fbjni/)
set(fbjni_BUILD_DIR ${CMAKE_BINARY_DIR}/fbjni/${BUILD_SUBDIR})

add_subdirectory(${fbjni_DIR} ${fbjni_BUILD_DIR})

# ---[ Vulkan deps
if(USE_VULKAN)
  set(Vulkan_LIBS)
  set(Vulkan_INCLUDES)
  include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/VulkanDependencies.cmake)
endif()

if(ANDROID_ABI)
  function(import_static_lib name)
    add_library(${name} STATIC IMPORTED)
    set_property(
        TARGET ${name}
        PROPERTY IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/${name}.a)
  endfunction(import_static_lib)

  import_static_lib(libsmith)
  import_static_lib(libsmith_cpu)
  import_static_lib(libc10)
  import_static_lib(libnnpack)
  import_static_lib(libXNNPACK)
  import_static_lib(libmicrokernels-prod)
  import_static_lib(libblacksmith_qnnpack)
  import_static_lib(libpthreadpool)
  import_static_lib(libeigen_blas)
  import_static_lib(libcpuinfo)
  import_static_lib(libclog)

  # Link most things statically on Android.
  set(blacksmith_jni_LIBS
      fbjni
      -Wl,--gc-sections
      -Wl,--whole-archive
      libsmith
      libsmith_cpu
      -Wl,--no-whole-archive
      libc10
      libnnpack
      libXNNPACK
      libmicrokernels-prod
      libblacksmith_qnnpack
      libpthreadpool
      libeigen_blas
      libcpuinfo
      libclog
  )
else()
  # Prefer dynamic linking on the host
  set(blacksmith_jni_LIBS
      fbjni
      smith
      smith_cpu
      c10
      cpuinfo
  )

  if(USE_NNPACK)
    list(APPEND blacksmith_jni_LIBS nnpack)
  endif()

  if(USE_XNNPACK)
    list(APPEND blacksmith_jni_LIBS XNNPACK)
    list(APPEND blacksmith_jni_LIBS microkernels-prod)
  endif()

  if(USE_SYSTEM_PTHREADPOOL)
    list(APPEND blacksmith_jni_LIBS pthreadpool)
  endif()

  if(USE_BLACKSMITH_QNNPACK)
    list(APPEND blacksmith_jni_LIBS blacksmith_qnnpack)
    list(APPEND blacksmith_jni_LIBS clog)
  endif()

endif()

if(USE_VULKAN)
  list(APPEND blacksmith_jni_LIBS ${Vulkan_LIBS})
endif()

target_link_libraries(${BLACKSMITH_JNI_TARGET} ${blacksmith_jni_LIBS})

install(TARGETS ${BLACKSMITH_JNI_TARGET}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) #For windows

if(MSVC)
  install(FILES $<TARGET_PDB_FILE:blacksmith_jni> DESTINATION ${CMAKE_INSTALL_LIBDIR} OPTIONAL)
  install(TARGETS ${BLACKSMITH_JNI_TARGET} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
