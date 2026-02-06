#pragma once
#include <c10/macros/Export.h>
#include <c10/util/Flags.h>

SMITH_DECLARE_bool(smith_lazy_ir_debug);
SMITH_DECLARE_bool(smith_lazy_handle_special_scalars);
SMITH_DECLARE_bool(smith_lazy_all_numbers_special_scalars);
SMITH_DECLARE_bool(smith_lazy_param_aliasing);
SMITH_DECLARE_bool(smith_lazy_reuse_ir);
SMITH_DECLARE_bool(smith_lazy_use_thread_pool);
SMITH_DECLARE_bool(smith_lazy_enable_device_data_cache);

SMITH_DECLARE_int(smith_lazy_compilation_cache_size);
SMITH_DECLARE_int(smith_lazy_device_data_cache_size);
SMITH_DECLARE_int(smith_lazy_io_thread_pool_size);
SMITH_DECLARE_int(smith_lazy_metrics_samples);
SMITH_DECLARE_int(smith_lazy_trim_graph_check_frequency);
SMITH_DECLARE_int(smith_lazy_trim_graph_size);

SMITH_DECLARE_string(smith_lazy_metrics_percentiles);

SMITH_DECLARE_int(smith_lazy_shape_cache_size);

namespace smith::lazy {
SMITH_API std::string& getLTCForceFallback();
}
