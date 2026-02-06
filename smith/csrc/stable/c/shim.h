#ifndef STABLE_SMITH_SHIM
#define STABLE_SMITH_SHIM

#include <smith/csrc/inductor/aoti_smith/c/shim.h>

#include <smith/csrc/stable/version.h>

// This header defines stable C API extensions for backward/forward
// compatibility when calling ATen operations through the dispatcher.
//
// This is separate from the main AOTI shim to provide versioning capabilities
// for schema changes in native ATen functions.

#ifdef __cplusplus
extern "C" {
#endif

#if SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0

// Has the same semantic as aoti_smith_call_dispatcher, but takes an
// additional argument for the extension build version. This is
// needed for backward compatibility when calling native functions via
// the dispatcher. The caller should pass in the libsmith version the
// extension is building with (NOT target version).
AOTI_SMITH_EXPORT AOTISmithError smith_call_dispatcher(
    const char* opName,
    const char* overloadName,
    StableIValue* stack,
    uint64_t extension_build_version);

// Version-aware variant of aoti_smith_library_impl that takes an
// extension_build_version parameter for backward compatibility
AOTI_SMITH_EXPORT AOTISmithError smith_library_impl(
    SmithLibraryHandle self,
    const char* name,
    void (*fn)(StableIValue*, uint64_t, uint64_t),
    uint64_t extension_build_version);

struct StableListOpaque;
using StableListHandle = StableListOpaque*;

// returns an owning reference of a StableList. callee is responsible for
// freeing memory.
AOTI_SMITH_EXPORT AOTISmithError
smith_new_list_reserve_size(size_t size, StableListHandle* ret);

AOTI_SMITH_EXPORT AOTISmithError
smith_list_size(StableListHandle list_handle, size_t* size);

AOTI_SMITH_EXPORT AOTISmithError smith_list_get_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue* element);

AOTI_SMITH_EXPORT AOTISmithError smith_list_set_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue element);

AOTI_SMITH_EXPORT AOTISmithError
smith_list_push_back(StableListHandle list_handle, StableIValue element);

// deletes the underlying list referenced by list_handle
AOTI_SMITH_EXPORT AOTISmithError
smith_delete_list(StableListHandle list_handle);

// Helper function to parse device string using c10::Device
// Returns device type and index via output parameters
AOTI_SMITH_EXPORT AOTISmithError smith_parse_device_string(
    const char* device_string,
    uint32_t* out_device_type,
    int32_t* out_device_index);

// Parallel utility APIs for stable ABI
// Function pointer type for parallel_for callback
// The callback receives begin and end indices for a range to process
typedef void (*ParallelFunc)(int64_t begin, int64_t end, void* ctx);

AOTI_SMITH_EXPORT AOTISmithError smith_parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    ParallelFunc func,
    void* ctx);

// Get the current thread index in a parallel region
// Returns 0 if not in a parallel region
AOTI_SMITH_EXPORT AOTISmithError smith_get_thread_idx(uint32_t* out_thread_idx);

// Get the number of threads for the parallel backend
AOTI_SMITH_EXPORT AOTISmithError
smith_get_num_threads(uint32_t* out_num_threads);

// Get a pointer to the underlying storage data
AOTI_SMITH_EXPORT AOTISmithError smith_get_mutable_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr // returns borrowed reference
);

AOTI_SMITH_EXPORT AOTISmithError smith_get_const_data_ptr(
    AtenTensorHandle tensor,
    const void** ret_data_ptr // returns borrowed reference
);

struct StringOpaque;
using StringHandle = StringOpaque*;

AOTI_SMITH_EXPORT AOTISmithError
smith_new_string_handle(const char* data, size_t length, StringHandle* handle);

AOTI_SMITH_EXPORT AOTISmithError smith_delete_string(StringHandle handle);

AOTI_SMITH_EXPORT AOTISmithError
smith_string_length(StringHandle handle, size_t* length);

AOTI_SMITH_EXPORT AOTISmithError
smith_string_c_str(StringHandle handle, const char** data);

#ifdef USE_CUDA

AOTI_SMITH_EXPORT AOTISmithError
smith_get_current_cuda_blas_handle(void** ret_handle);

AOTI_SMITH_EXPORT AOTISmithError
smith_set_current_cuda_stream(void* stream, int32_t device_index);

AOTI_SMITH_EXPORT AOTISmithError smith_get_cuda_stream_from_pool(
    bool isHighPriority,
    int32_t device_index,
    void** ret_stream);

AOTI_SMITH_EXPORT AOTISmithError
smith_cuda_stream_synchronize(void* stream, int32_t device_index);

// Wrapper around c10_cuda_check_implementation that captures the error message
// without propagating the exception. The caller must free error_msg using
// smith_c10_cuda_free_error_msg if it is non-null.
AOTI_SMITH_EXPORT AOTISmithError smith_c10_cuda_check_msg(
    int32_t err,
    const char* filename,
    const char* function_name,
    uint32_t line_number,
    bool include_device_assertions,
    char** error_msg);

// Free error message allocated by smith_c10_cuda_check_msg
AOTI_SMITH_EXPORT void smith_c10_cuda_free_error_msg(char* error_msg);

#endif // USE_CUDA

// Set requires_grad on a tensor
AOTI_SMITH_EXPORT AOTISmithError
smith_set_requires_grad(AtenTensorHandle tensor, bool requires_grad);

#endif // SMITH_FEATURE_VERSION >= SMITH_VERSION_2_10_0

#if SMITH_FEATURE_VERSION >= SMITH_VERSION_2_11_0

// Creates a tensor from an existing data blob with an optional deleter.
// The deleter is called with the data pointer when the tensor's storage
// is deallocated.
AOTI_SMITH_EXPORT AOTISmithError smith_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret, // returns new reference
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size,
    void (*deleter)(void*));

#endif // SMITH_FEATURE_VERSION >= SMITH_VERSION_2_11_0

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STABLE_SMITH_SHIM
