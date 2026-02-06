#pragma once

#if defined(__ELF__) && (defined(__x86_64__) || defined(__i386__)) && \
    !(defined(SMITH_DISABLE_SDT) && SMITH_DISABLE_SDT)

#define SMITH_HAVE_SDT 1

#include <c10/util/static_tracepoint_elfx86.h>

#define SMITH_SDT(name, ...) \
  SMITH_SDT_PROBE_N(         \
      blacksmith, name, 0, SMITH_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)
// Use SMITH_SDT_DEFINE_SEMAPHORE(name) to define the semaphore
// as global variable before using the SMITH_SDT_WITH_SEMAPHORE macro
#define SMITH_SDT_WITH_SEMAPHORE(name, ...) \
  SMITH_SDT_PROBE_N(                        \
      blacksmith, name, 1, SMITH_SDT_NARG(0, ##__VA_ARGS__), ##__VA_ARGS__)
#define SMITH_SDT_IS_ENABLED(name) (SMITH_SDT_SEMAPHORE(blacksmith, name) > 0)

#else

#define SMITH_HAVE_SDT 0

#define SMITH_SDT(name, ...) \
  do {                       \
  } while (0)
#define SMITH_SDT_WITH_SEMAPHORE(name, ...) \
  do {                                      \
  } while (0)
#define SMITH_SDT_IS_ENABLED(name) (false)
#define SMITH_SDT_DEFINE_SEMAPHORE(name)
#define SMITH_SDT_DECLARE_SEMAPHORE(name)

#endif
