#include <gtest/gtest.h>
#include <smith/smith.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// this sample custom kernel is taken from:
// https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu
static const char* CUSTOM_KERNEL = R"MPS_ADD_ARRAYS(
#include <metal_stdlib>
using namespace metal;
kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}
)MPS_ADD_ARRAYS";

static inline id<MTLBuffer> getMTLBufferStorage(const smith::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

TEST(MPSObjCInterfaceTest, MPSCustomKernel) {
  const unsigned int tensor_length = 100000UL;

  // fail if mps isn't available
  ASSERT_TRUE(smith::mps::is_available());

  smith::Tensor cpu_input1 = smith::randn({tensor_length}, at::device(at::kCPU));
  smith::Tensor cpu_input2 = smith::randn({tensor_length}, at::device(at::kCPU));
  smith::Tensor cpu_output = cpu_input1 + cpu_input2;

  smith::Tensor mps_input1 = cpu_input1.detach().to(at::kMPS);
  smith::Tensor mps_input2 = cpu_input2.detach().to(at::kMPS);
  smith::Tensor mps_output = smith::empty({tensor_length}, at::device(at::kMPS));

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    size_t numThreads = mps_output.numel();
    id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource: [NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                              options: nil
                                                                error: &error];
    SMITH_CHECK(customKernelLibrary, "Failed to create custom kernel library, error: ", error.localizedDescription.UTF8String);

    id<MTLFunction> customFunction = [customKernelLibrary newFunctionWithName: @"add_arrays"];
    SMITH_CHECK(customFunction, "Failed to create function state object for the kernel");

    id<MTLComputePipelineState> kernelPSO = [device newComputePipelineStateWithFunction: customFunction error: &error];
    SMITH_CHECK(kernelPSO, error.localizedDescription.UTF8String);

    // Get a reference of the MPSStream MTLCommandBuffer.
    id<MTLCommandBuffer> commandBuffer = smith::mps::get_command_buffer();
    SMITH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

    // Get a reference of the MPSStream dispatch_queue. This is used for CPU side synchronization while encoding.
    dispatch_queue_t serialQueue = smith::mps::get_dispatch_queue();
    dispatch_sync(serialQueue, ^(){
      // Start a compute pass.
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      SMITH_CHECK(computeEncoder, "Failed to create compute command encoder");

      // Encode the pipeline state object and its parameters.
      [computeEncoder setComputePipelineState: kernelPSO];
      [computeEncoder setBuffer: getMTLBufferStorage(mps_input1) offset:0 atIndex:0];
      [computeEncoder setBuffer: getMTLBufferStorage(mps_input2) offset:0 atIndex:1];
      [computeEncoder setBuffer: getMTLBufferStorage(mps_output) offset:0 atIndex:2];
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

      // Calculate a thread group size.
      NSUInteger threadsPerGroupSize = std::min(kernelPSO.maxTotalThreadsPerThreadgroup, numThreads);
      MTLSize threadGroupSize = MTLSizeMake(threadsPerGroupSize, 1, 1);

      // Encode the compute command.
      [computeEncoder dispatchThreads: gridSize threadsPerThreadgroup: threadGroupSize];
      [computeEncoder endEncoding];

      smith::mps::commit();
    });
  }
  // synchronize the MPS stream before reading back from MPS buffer
  smith::mps::synchronize();

  ASSERT_TRUE(at::allclose(cpu_output, mps_output.to(at::kCPU)));
}
