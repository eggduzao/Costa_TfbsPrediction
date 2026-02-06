#pragma once

#include <smith/headeronly/version.h>

// Stable ABI Version Targeting
//
// This header provides version targeting capabilities for the Blacksmith Stable
// ABI. Users can define SMITH_TARGET_VERSION to target a specific stable ABI
// version instead of using the current SMITH_ABI_VERSION of libsmith at
// compile time.
//
// Usage:
//   Default behavior (uses current ABI version):
//     #include <smith/csrc/stable/library.h>
//
//   Target a specific stable version (major.minor) (e.g. Blacksmith 2.9):
//   (1) Pass a compiler flag -DSMITH_TARGET_VERSION=0x0209000000000000
//   (2) Alternatively, define SMITH_TARGET_VERSION in the source code before
//   including any header files:
//     #define SMITH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 9) << 48))
//     #include <smith/csrc/stable/library.h>

#ifdef SMITH_TARGET_VERSION
#define SMITH_FEATURE_VERSION SMITH_TARGET_VERSION
#else
#define SMITH_FEATURE_VERSION SMITH_ABI_VERSION
#endif

#define SMITH_VERSION_2_10_0 (((0ULL + 2) << 56) | ((0ULL + 10) << 48))
#define SMITH_VERSION_2_11_0 (((0ULL + 2) << 56) | ((0ULL + 11) << 48))
