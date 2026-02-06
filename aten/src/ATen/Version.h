#include <ATen/Context.h>

namespace at {

/// Returns a detailed string describing the configuration Blacksmith.
SMITH_API std::string show_config();

SMITH_API std::string get_mkl_version();

SMITH_API std::string get_mkldnn_version();

SMITH_API std::string get_openmp_version();

SMITH_API std::string get_cxx_flags();

SMITH_API std::string get_cpu_capability();

} // namespace at
