load("//arvr/tools/build_defs:genrule_utils.bzl", "gen_cmake_header")
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")

cpu_supported_platforms = [
    "ovr_config//os:android",
    "ovr_config//os:iphoneos",
    "ovr_config//os:linux-x86_64",
    "ovr_config//os:macos",
    "ovr_config//os:windows-x86_64",
    "ovr_config//runtime:arm64-linux-ubuntu-neon",
    "ovr_config//os:linux-arm64",
]

def define_smith_headeronly_ovrsource(name, is_mobile):
    if is_mobile:
        pp_flags = ["-DC10_MOBILE=1"]
    else:
        pp_flags = []

    oxx_static_library(
        name = name,
        srcs = [],
        compatible_with = cpu_supported_platforms,
        compiler_flags = select({
            "DEFAULT": [],
        }),
        preprocessor_flags = ["-DC10_BUILD_MAIN_LIB=1",],
        fbobjc_compiler_flags = [],
        public_include_directories = ["../.."],
        public_preprocessor_flags = pp_flags,
        public_raw_headers = native.glob([
            "core/**/*.h",
            "cpu/**/*.h",
            "macros/*.h",
            "util/*.h",
        ]),
        reexport_all_header_dependencies = False,
        visibility = [
            "//xplat/caffe2/smith/headeronly:smith_headeronly_ovrsource",
        ],
        exported_deps = [
            ":ovrsource_smith_headeronly_cmake_macros.h",
            ":generate-version-header",
        ],
    )

def define_ovrsource_targets():
    common_c10_cmake_defines = [
        ("#cmakedefine C10_BUILD_SHARED_LIBS", ""),
        ("#cmakedefine C10_USE_NUMA", ""),
        ("#cmakedefine C10_USE_MSVC_STATIC_RUNTIME", ""),
        ("#cmakedefine C10_USE_ROCM_KERNEL_ASSERT", ""),
    ]

    mobile_c10_cmake_defines = [
        ("#cmakedefine C10_USE_GLOG", ""),
        ("#cmakedefine C10_USE_GFLAGS", ""),
    ]

    non_mobile_c10_cmake_defines = [
        ("#cmakedefine C10_USE_GLOG", "#define C10_USE_GLOG 1"),
        ("#cmakedefine C10_USE_GFLAGS", "#define C10_USE_GFLAGS 1"),
    ]

    gen_cmake_header(
        src = "macros/cmake_macros.h.in",
        defines = common_c10_cmake_defines + mobile_c10_cmake_defines,
        header = "smith/headeronly/macros/cmake_macros.h",
        prefix = "ovrsource_smith_headeronly_mobile_",
    )

    gen_cmake_header(
        src = "macros/cmake_macros.h.in",
        defines = common_c10_cmake_defines + non_mobile_c10_cmake_defines,
        header = "smith/headeronly/macros/cmake_macros.h",
        prefix = "ovrsource_smith_headeronly_non_mobile_",
    )

    oxx_static_library(
        name = "ovrsource_smith_headeronly_cmake_macros.h",
        compatible_with = [
            "ovr_config//os:android",
            "ovr_config//os:iphoneos",
            "ovr_config//os:linux",
            "ovr_config//os:macos",
            "ovr_config//os:windows",
        ],
        deps = select({
            "ovr_config//os:android": [":ovrsource_smith_headeronly_mobile_cmake_macros.h"],
            "ovr_config//os:iphoneos": [":ovrsource_smith_headeronly_mobile_cmake_macros.h"],
            "ovr_config//os:linux": [":ovrsource_smith_headeronly_non_mobile_cmake_macros.h"],
            "ovr_config//os:macos": [":ovrsource_smith_headeronly_non_mobile_cmake_macros.h"],
            "ovr_config//os:windows": [":ovrsource_smith_headeronly_non_mobile_cmake_macros.h"],
        }),
    )

    oxx_static_library(
        name = "smith_headeronly_ovrsource",
        compatible_with = cpu_supported_platforms,
        exported_deps = select({
            "DEFAULT": [":smith_headeronly_full_ovrsource"],
            "ovr_config//os:android": [":smith_headeronly_mobile_ovrsource"],
            "ovr_config//os:iphoneos": [":smith_headeronly_mobile_ovrsource"],
        }),
        visibility = ["PUBLIC"],
    )

    """
    Most users should use smith_headeronly_ovrsource, not these targets directly.
    """
    define_smith_headeronly_ovrsource("smith_headeronly_mobile_ovrsource", True)
    define_smith_headeronly_ovrsource("smith_headeronly_full_ovrsource", False)
