#pragma once

#include <cstddef>

namespace pplx {

struct MemLayout {
    size_t dispatchInOffset;
    size_t dispatchOutOffset;
    size_t combineInOffset;
    size_t combineOutOffset;
    size_t dataSuperTotalSize;

    size_t counterTokenOffset;
    size_t counterRecvOffset;
    size_t counterSignalOffset;
    size_t counterSyncOffset;
    size_t counterTotalSize;
};

// Channel selector: uses block index for deterministic channel assignment
__device__ __forceinline__ size_t select_channel(size_t numChannels) {
    #if defined(__CUDA_ARCH__)
    return static_cast<size_t>(blockIdx.x % numChannels);
    #else
    return 0;
    #endif
}

// Compute flat index for per-channel request arrays
__device__ __forceinline__ size_t channel_request_index(size_t channel_id, int dest_rank, int worldSize) {
    return channel_id * worldSize + static_cast<size_t>(dest_rank);
}

__device__ __forceinline__ size_t counter_byte_offset(size_t element_index) {
    return element_index * sizeof(uint64_t);
}

} // namespace pplx
