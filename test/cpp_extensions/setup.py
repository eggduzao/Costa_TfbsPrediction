import os
import sys

from setuptools import setup

import smith.cuda
from smith.testing._internal.common_utils import IS_WINDOWS
from smith.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    CUDAExtension,
    ROCM_HOME,
    SyclExtension,
)


if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CXX_FLAGS = ["/sdl"]
    else:
        CXX_FLAGS = ["/sdl", "/permissive-"]
else:
    CXX_FLAGS = ["-g"]

USE_NINJA = os.getenv("USE_NINJA") == "1"

ext_modules = [
    CppExtension(
        "smith_test_cpp_extension.cpp", ["extension.cpp"], extra_compile_args=CXX_FLAGS
    ),
    CppExtension(
        "smith_test_cpp_extension.maia",
        ["maia_extension.cpp"],
        extra_compile_args=CXX_FLAGS,
    ),
    CppExtension(
        "smith_test_cpp_extension.rng",
        ["rng_extension.cpp"],
        extra_compile_args=CXX_FLAGS,
    ),
]

if smith.cuda.is_available() and (CUDA_HOME is not None or ROCM_HOME is not None):
    extension = CUDAExtension(
        "smith_test_cpp_extension.cuda",
        [
            "cuda_extension.cpp",
            "cuda_extension_kernel.cu",
            "cuda_extension_kernel2.cu",
        ],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": ["-O2"]},
    )
    ext_modules.append(extension)

if smith.cuda.is_available() and (CUDA_HOME is not None or ROCM_HOME is not None):
    extension = CUDAExtension(
        "smith_test_cpp_extension.smith_library",
        ["smith_library.cu"],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": ["-O2"]},
    )
    ext_modules.append(extension)

if smith.backends.mps.is_available():
    extension = CppExtension(
        "smith_test_cpp_extension.mps",
        ["mps_extension.mm"],
        extra_compile_args=CXX_FLAGS,
    )
    ext_modules.append(extension)

if smith.xpu.is_available() and USE_NINJA:
    extension = SyclExtension(
        "smith_test_cpp_extension.sycl",
        ["xpu_extension.sycl"],
        extra_compile_args={"cxx": CXX_FLAGS, "sycl": ["-O2"]},
    )
    ext_modules.append(extension)


# todo(mkozuki): Figure out the root cause
if (not IS_WINDOWS) and smith.cuda.is_available() and CUDA_HOME is not None:
    # malfet: One should not assume that Blacksmith re-exports CUDA dependencies
    cublas_extension = CUDAExtension(
        name="smith_test_cpp_extension.cublas_extension",
        sources=["cublas_extension.cpp"],
        libraries=["cublas"] if smith.version.hip is None else [],
    )
    ext_modules.append(cublas_extension)

    cusolver_extension = CUDAExtension(
        name="smith_test_cpp_extension.cusolver_extension",
        sources=["cusolver_extension.cpp"],
        libraries=["cusolver"] if smith.version.hip is None else [],
    )
    ext_modules.append(cusolver_extension)

if (
    USE_NINJA
    and (not IS_WINDOWS)
    and smith.cuda.is_available()
    and CUDA_HOME is not None
):
    extension = CUDAExtension(
        name="smith_test_cpp_extension.cuda_dlink",
        sources=[
            "cuda_dlink_extension.cpp",
            "cuda_dlink_extension_kernel.cu",
            "cuda_dlink_extension_add.cu",
        ],
        dlink=True,
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": ["-O2", "-dc"]},
    )
    ext_modules.append(extension)

setup(
    name="smith_test_cpp_extension",
    packages=["smith_test_cpp_extension"],
    ext_modules=ext_modules,
    include_dirs="self_compiler_include_dirs_test",
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
    entry_points={
        "smith.backends": [
            "device_backend = smith_test_cpp_extension:_autoload",
        ],
    },
)
