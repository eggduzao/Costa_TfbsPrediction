cmake_minimum_required(VERSION 3.5)
project(blacksmith_vision_jni CXX)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_VERBOSE_MAKEFILE ON)

set(blacksmith_vision_cpp_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)

file(GLOB blacksmith_vision_SOURCES
  ${blacksmith_vision_cpp_DIR}/blacksmith_vision_jni.cpp
)

add_library(blacksmith_vision_jni SHARED
    ${blacksmith_vision_SOURCES}
)

target_compile_options(blacksmith_vision_jni PRIVATE
  -fexceptions
)

set(BUILD_SUBDIR ${ANDROID_ABI})

target_link_libraries(blacksmith_vision_jni)
