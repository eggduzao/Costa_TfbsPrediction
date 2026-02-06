# Define this to build the NNAPI binding out of tree.
if(BLACKSMITH_NNAPI_STANDALONE)
  cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
  project(blacksmith_nnapi)

  set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard whose features are requested to build this target.")
  find_package(Smith REQUIRED)

  set(NNAPI_SRCS
    nnapi_bind.cpp
    nnapi_wrapper.cpp
    nnapi_model_loader.cpp
    )

  add_library(blacksmith_nnapi SHARED ${NNAPI_SRCS})
  target_link_libraries(blacksmith_nnapi smith)
else()
  # Building within the Blacksmith tree.
  file(GLOB ATen_NNAPI_SRCS "*.cpp")
  set(ATen_NNAPI_SRCS ${ATen_NNAPI_SRCS} PARENT_SCOPE)
endif()
