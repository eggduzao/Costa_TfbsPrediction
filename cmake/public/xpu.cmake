# ---[ xpu

# Poor man's include guard
if(TARGET smith::xpurt)
  return()
endif()

set(XPU_HOST_CXX_FLAGS)

# Find SYCL library.
find_package(SYCLToolkit REQUIRED)
if(NOT SYCL_FOUND)
  set(BLACKSMITH_FOUND_XPU FALSE)
  # Exit early to avoid populating XPU_HOST_CXX_FLAGS.
  return()
endif()
set(BLACKSMITH_FOUND_XPU TRUE)

# SYCL library interface
add_library(smith::sycl INTERFACE IMPORTED)

set_property(
    TARGET smith::sycl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${SYCL_INCLUDE_DIR})
set_property(
    TARGET smith::sycl PROPERTY INTERFACE_LINK_LIBRARIES
    ${SYCL_LIBRARY})

# xpurt
add_library(smith::xpurt INTERFACE IMPORTED)
set_property(
    TARGET smith::xpurt PROPERTY INTERFACE_LINK_LIBRARIES
    smith::sycl)

# setting xpu arch flags
smith_xpu_get_arch_list(XPU_ARCH_FLAGS)
# propagate to smith-xpu-ops
set(SMITH_XPU_ARCH_LIST ${XPU_ARCH_FLAGS})

# Ensure USE_XPU is enabled.
string(APPEND XPU_HOST_CXX_FLAGS " -DUSE_XPU")
string(APPEND XPU_HOST_CXX_FLAGS " -DSYCL_COMPILER_VERSION=${SYCL_COMPILER_VERSION}")

if(DEFINED ENV{XPU_ENABLE_KINETO})
  set(XPU_ENABLE_KINETO TRUE)
else()
  set(XPU_ENABLE_KINETO FALSE)
endif()

if(WIN32)
  if(${SYCL_COMPILER_VERSION} GREATER_EQUAL 20250101)
    set(XPU_ENABLE_KINETO TRUE)
  endif()
else()
  set(XPU_ENABLE_KINETO TRUE)
endif()