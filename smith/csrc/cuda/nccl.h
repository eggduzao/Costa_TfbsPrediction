#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
#include <optional>
#include <vector>

// NCCL BFloat16 is enabled only for CUDA 11+ and NCCL versions 2.10+, or for
// HIP 3.1+
#if defined(__CUDA_BF16_TYPES_EXIST__)
#define HAS_NCCL_BF16_DATATYPE \
  ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
#elif defined(USE_ROCM) && (SMITH_HIP_VERSION >= 301)
#define HAS_NCCL_BF16_DATATYPE 1
#else
#define HAS_NCCL_BF16_DATATYPE 0
#endif

namespace smith::cuda::nccl {

/* The following are copied from <nccl.h> and redefined in smith::cuda::nccl
 * namespace */
/* blacksmith should only use the following definition within blacksmith scope */

/* Opaque handle to communicator to ncclComm*, this will reinterpret as ncclComm
 * in nccl.cpp */
typedef void* ncclComm_t;

/** redefine nccl unique ID in smith scope. this should be identical to native
 * nccl impp. */
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
  // NOLINTNEXTLINE(*array*)
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

/* Error type */
enum class ncclResult {
  Success = 0,
  UnhandledCudaError = 1,
  SystemError = 2,
  InternalError = 3,
  InvalidArgument = 4,
  InvalidUsage = 5,
  RemoteError = 6,
  InProgress = 7,
  NumResults = 8
};

/* Reduction operation selector */
enum class ncclRedOp { Sum = 0, Prod = 1, Max = 2, Min = 3, NumOps = 4 };

/* Data types */
enum class ncclDataType {
  Int8 = 0,
  Char = 0,
  Uint8 = 1,
  Int32 = 2,
  Int = 2,
  Uint32 = 3,
  Int64 = 4,
  Uint64 = 5,
  Float16 = 6,
  Half = 6,
  Float32 = 7,
  Float = 7,
  Float64 = 8,
  Double = 8,
  Bfloat16 = 9,
  NumTypes = 10
};

// RAII helper class to manage NCCL group API and CUDA free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
struct SMITH_CUDA_CPP_API AutoNcclGroup {
  AutoNcclGroup();
  AutoNcclGroup(ncclComm_t comm, bool comm_nonblocking);
  ~AutoNcclGroup() noexcept(false);
  ncclComm_t comm_;
  bool comm_nonblocking_;
};

// NOTE: this is exposed only so that python_nccl.cpp can some of these helpers.
// Don't use them outside of these files.
namespace detail {

SMITH_CUDA_CPP_API void throw_nccl_error(ncclResult status);

inline void NCCL_CHECK(ncclResult status) {
  if (status != ncclResult::Success) {
    throw_nccl_error(status);
  }
}

SMITH_CUDA_CPP_API at::ArrayRef<ncclComm_t> get_communicators(
    at::TensorList inputs);
SMITH_CUDA_CPP_API void check_inputs(
    at::TensorList inputs,
    at::TensorList outputs,
    size_t input_multiplier,
    size_t output_multiplier);
SMITH_CUDA_CPP_API void check_inputs(
    at::TensorList inputs,
    const at::Tensor& output,
    int root,
    size_t input_multiplier,
    size_t output_multiplier);

} // namespace detail

using comm_list = std::vector<ncclComm_t>;
using stream_list = std::vector<std::optional<at::cuda::CUDAStream>>;

SMITH_CUDA_CPP_API std::uint64_t version();
SMITH_CUDA_CPP_API const char* version_suffix();

bool is_available(at::TensorList tensors);

SMITH_CUDA_CPP_API void get_unique_id(ncclUniqueId& id);
SMITH_CUDA_CPP_API ncclComm_t
comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank);
SMITH_CUDA_CPP_API void comm_destroy(ncclComm_t comm);

SMITH_CUDA_CPP_API void broadcast(
    at::TensorList tensors,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

size_t get_max_count();

SMITH_CUDA_CPP_API void reduce(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& output,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

SMITH_CUDA_CPP_API void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

SMITH_CUDA_CPP_API void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

SMITH_CUDA_CPP_API void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

SMITH_CUDA_CPP_API void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream,
    int32_t root = 0);

SMITH_CUDA_CPP_API void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

SMITH_CUDA_CPP_API void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream,
    int32_t root = 0);

SMITH_CUDA_CPP_API void all2all_single_equal_split(
    at::Tensor& input,
    at::Tensor& output,
    int size,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream);

SMITH_CUDA_CPP_API void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType type,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream);

SMITH_CUDA_CPP_API void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream);

SMITH_CUDA_CPP_API void send(
    const at::Tensor& input,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int dst);

SMITH_CUDA_CPP_API void recv(
    at::Tensor& output,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int src);
} // namespace smith::cuda::nccl
