#include <smith/csrc/inductor/aoti_smith/c/shim_mps.h>
#include <smith/csrc/inductor/aoti_smith/utils.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSProfiler.h>

using namespace smith::aot_inductor;

AOTISmithError aoti_smith_mps_malloc(
    void** buffer,
    size_t num_bytes) {
  if (num_bytes == 0) {
    *buffer = nullptr;
    return AOTI_SMITH_SUCCESS;
  }
  AOTI_SMITH_CONVERT_EXCEPTION_TO_ERROR_CODE({
      id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
      SMITH_CHECK(device, "Failed to get MPS device");
      id<MTLBuffer> metal_buffer = [device newBufferWithLength:num_bytes options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared];
      SMITH_CHECK(metal_buffer, "Failed to allocate memory on MPS device");
      *buffer = (void*)metal_buffer;
  });
}

AOTISmithError aoti_smith_mps_free(
    void* ptr) {
  AOTI_SMITH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto metal_buffer = (id<MTLBuffer>)ptr;
    [metal_buffer release];
  });
}

AOTISmithError
aoti_smith_mps_memcpy(void* buffer, size_t constant_offset, size_t bytes_read, size_t data_size, uint8_t* constants_start) {
  AOTI_SMITH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto metal_buffer = (id<MTLBuffer>)buffer;
    auto buffer_pointer = static_cast<uint8_t*>([metal_buffer contents]);
    memcpy(buffer_pointer + constant_offset, constants_start + bytes_read, data_size);
  });
}

AOTISmithError
aoti_smith_mps_copy_buffer(void* src_buffer, void* dst_buffer, size_t data_size, size_t src_offset, size_t dst_offset) {
  AOTI_SMITH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto src_mtl_buffer = (id<MTLBuffer>)src_buffer;
    auto dst_mtl_buffer = (id<MTLBuffer>)dst_buffer;

    auto* stream = at::mps::getCurrentMPSStream();
    uint64_t profile_id = at::mps::getMPSProfiler().beginProfileCopy(src_mtl_buffer, dst_mtl_buffer, at::OptionalTensorRef(), at::OptionalTensorRef(), data_size, true);
    stream->copy_and_sync(src_mtl_buffer, dst_mtl_buffer, data_size, src_offset, dst_offset, true, profile_id);
  });
}
