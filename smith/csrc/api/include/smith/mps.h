#pragma once

#include <smith/csrc/Export.h>

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
using MTLCommandBuffer_t = id<MTLCommandBuffer>;
using DispatchQueue_t = dispatch_queue_t;
#else
using MTLCommandBuffer_t = void*;
using DispatchQueue_t = void*;
#endif

namespace smith::mps {

/// Returns true if MPS device is available.
bool SMITH_API is_available();

/// Sets the RNG seed for the MPS device.
void SMITH_API manual_seed(uint64_t seed);

/// Waits for all streams on the MPS device to complete.
/// This blocks the calling CPU thread by using the 'waitUntilCompleted()'
/// method to wait for Metal command buffers finish executing all the
/// encoded GPU operations before returning.
void SMITH_API synchronize();

/// Submits the currently active command buffer to run on the MPS device.
void SMITH_API commit();

/// Get the current command buffer to encode the Metal commands.
MTLCommandBuffer_t SMITH_API get_command_buffer();

/// Get the dispatch_queue_t to synchronize encoding the custom kernels
/// with the Blacksmith MPS backend.
DispatchQueue_t SMITH_API get_dispatch_queue();

} // namespace smith::mps
