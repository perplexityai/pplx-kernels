#include "core/device_utils.cuh"
#include "core/utils.h"
#include "core/nixl_utils.cuh"
#include "internode.h"

#include <cuda.h>
#include <nvtx3/nvToolsExt.h>

using namespace pplx;

template <typename T, typename U, size_t NUM_WARPS, bool DO_SEND, bool DO_RECV>
__global__ __launch_bounds__(NUM_WARPS * 32, 1) void combineKernel(
    U *outTokens,
    size_t outTokensStrideElem,
    const uint32_t *indices,
    size_t indicesStrideElem,
    size_t indicesStrideRow,
    const float *weights,
    size_t weightsStrideElem,
    size_t weightsStrideRow,
    const T *expertX,
    size_t expertXStrideElem,
    size_t expertXStrideRow,
    size_t expertsPerToken,
    size_t maxNumTokens,
    size_t numExperts,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    unsigned *boundM,
    unsigned m,
    uint32_t *sourceExpert,
    const uint32_t *sourceIndex,
    const uint32_t *sourceOffset,
    const uint32_t *sourceGroup,
    uint64_t *combineSignalBuffer,
    uint64_t *combineSignalLocalBuffer,
    uint64_t *combineSyncBuffer,
    uint64_t *combineSyncLocalBuffer,
    uint32_t &globalTokenIndex,
    std::byte *xBufferIn,
    std::byte *xBufferOut,
    const MemLayout memLayout,
    std::byte **peerDataBases,
    std::byte **peerCounterBases,
    size_t numChannels,
    const nixlGpuXferReqH* dataReqs,
    const nixlGpuXferReqH* signalReqs
) {

  const unsigned numLocalExperts = numExperts / worldSize;
  const size_t stride = hiddenDim * sizeof(T);
  constexpr unsigned WARP_SIZE = 32;
  uint32_t warpId = threadIdx.x / WARP_SIZE;

  const size_t channelId = select_channel(numChannels);

  if (DO_SEND) {
    const size_t numSendTokens = __ldg(&globalTokenIndex);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += gridDim.x * blockDim.x) {
        const size_t syncCounterOffset = memLayout.counterSyncOffset + counter_byte_offset(rank);

        if (peerCounterBases[i] != nullptr) {
          uint64_t one = 1ull;
          __nv_atomic_store(
              (uint64_t*)(peerCounterBases[i] + syncCounterOffset),
              &one,
              __NV_ATOMIC_RELEASE, PPLX_ATOMIC_SCOPE);
        } else {
          size_t reqIdx = channel_request_index(channelId, i, worldSize);
          nixl_status_t status = nixlGpuPostSignalXferReq<nixl_gpu_level_t::THREAD>(
              signalReqs[reqIdx],
              /*signal_desc_index=*/0,
              /*signal_inc=*/1,
              /*signal_offset=*/syncCounterOffset,
              /*channel_id=*/0,
              /*is_no_delay=*/true,
              /*status=*/nullptr);
          PPLX_DEVICE_ASSERT_POST_STATUS(status);
        }
    }

    // Dispatch the tokens from the expert to the DP groups.
    for (uint32_t token = blockIdx.x; token < numSendTokens; token += gridDim.x) {
      const uint32_t expert = __ldg(&sourceExpert[token]);
      const uint32_t offset = __ldg(&sourceOffset[token]);

      // Copy the token to symmetric memory for send.
      std::byte *xTokenPtr = xBufferIn + token * stride;
      {
        const T *expertXTokenPtr = expertX + expert * expertXStrideRow + offset * expertXStrideElem;

        int4 *dstPtr = (int4 *)xTokenPtr;
        const int4 *srcPtr = (const int4 *)expertXTokenPtr;
        dstPtr += threadIdx.x;
        srcPtr += threadIdx.x;

        const unsigned n = stride / sizeof(int4);
#pragma unroll(4)
        for (unsigned j = threadIdx.x; j < n; j += blockDim.x) {
          *dstPtr = __ldg(srcPtr);
          dstPtr += blockDim.x;
          srcPtr += blockDim.x;
        }
        __syncthreads();
      }

      const uint32_t dstExpert = rank * numLocalExperts + expert;

      const uint32_t source = __ldg(&sourceIndex[token]);
      const uint32_t dp = __ldg(&sourceGroup[token]);
      for (unsigned i = warpId; i < dpSize; i += NUM_WARPS) {
        const int dstRank = dp * dpSize + i;
        const unsigned index = dstExpert * maxNumTokens + source;

        // Unified offset calculation for both P2P and NIXL paths
        const size_t dstTokenOffset = index * stride;
        const size_t srcTokenOffset = token * stride;
        const size_t dstDataOffset = memLayout.combineOutOffset + dstTokenOffset;
        const size_t srcDataOffset = memLayout.combineInOffset + srcTokenOffset;

        const size_t signalCounterOffset = memLayout.counterSignalOffset + counter_byte_offset(source);

        if (peerCounterBases[dstRank] != nullptr) {
          device::warp_streaming_copy_and_sync(
              peerDataBases[dstRank] + dstDataOffset,
              xTokenPtr, stride, threadIdx.x % WARP_SIZE);
          if (threadIdx.x % WARP_SIZE == 0) {
            __nv_atomic_add(
                (uint64_t*)(peerCounterBases[dstRank] + signalCounterOffset),
                1ull,
                __NV_ATOMIC_RELEASE, PPLX_ATOMIC_SCOPE);
          }
        } else {
          if (threadIdx.x % WARP_SIZE == 0) {
            size_t sizeBytes = stride;
            size_t reqIdx = channel_request_index(channelId, dstRank, worldSize);
            nixl_status_t stData = nixlGpuPostSingleWriteXferReq<nixl_gpu_level_t::THREAD>(
                dataReqs[reqIdx],
                /*desc_index=*/0,
                /*local_offset=*/srcDataOffset,
                /*remote_offset=*/dstDataOffset,
                /*size=*/sizeBytes,
                /*channel_id=*/0,
                /*is_no_delay=*/false,
                /*status=*/nullptr);
            PPLX_DEVICE_ASSERT_POST_STATUS(stData);

            nixl_status_t stSignal = nixlGpuPostSignalXferReq<nixl_gpu_level_t::THREAD>(
                signalReqs[reqIdx],
                /*signal_desc_index=*/0,
                /*signal_inc=*/1,
                /*signal_offset=*/signalCounterOffset,
                /*channel_id=*/0,
                /*is_no_delay=*/true,
                /*status=*/nullptr);
            PPLX_DEVICE_ASSERT_POST_STATUS(stSignal);
          }
        }
        __syncwarp();
      }
    }
  }

  // Synchronize the grid to ensure that tokens routed within the rank are
  // correctly transported from one block to another.
  if (DO_RECV) {
    if (DO_SEND) {
      cooperative_groups::this_grid().sync();
    }

    // Compute the weighed sum of the input tokens.
    const size_t localNumTokens = boundM ? __ldg(boundM) : m;
    for (unsigned i = blockIdx.x; i < localNumTokens; i += gridDim.x) {
      // Split counters: remote (peers) and local-only (self)
      uint64_t *remoteSignalCounter = combineSignalBuffer + i;
      uint64_t *localOnlyCounter   = combineSignalLocalBuffer + i;

      // Wait on sum(remote + local) == expertsPerToken
      // Gate busy-wait to the first thread in the block
      if (threadIdx.x == 0) {
        device::nixl_wait_until_sum_eq(
            remoteSignalCounter,
            localOnlyCounter,
            expertsPerToken
        );
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        uint64_t zero = 0ull;
        pplx::st_flag_release(remoteSignalCounter, 0ull);
        __nv_atomic_store(localOnlyCounter, &zero, __NV_ATOMIC_RELEASE, PPLX_ATOMIC_SCOPE);
      }
      __syncthreads();

      U *dstPtr = outTokens + i * outTokensStrideElem;
      constexpr unsigned VEC_SIZE = 8;
      for (unsigned j = threadIdx.x * VEC_SIZE; j < hiddenDim; j += blockDim.x * VEC_SIZE) {
        float sum[VEC_SIZE];

#pragma unroll
        for (unsigned l = 0; l < VEC_SIZE; ++l) {
          sum[l] = 0.0f;
        }

        for (unsigned k = 0; k < expertsPerToken; ++k) {
          const uint32_t expert = __ldg(&indices[i * expertsPerToken + k]);
          const float weight = __ldg(&weights[i * weightsStrideRow + k]);

#pragma unroll
          for (unsigned l = 0; l < VEC_SIZE; ++l) {
            std::byte *xDstPtr = xBufferOut + (expert * maxNumTokens + i) * stride;
            sum[l] += weight * (float)((T *)xDstPtr)[j + l];
          }
        }

#pragma unroll
        for (unsigned l = 0; l < VEC_SIZE; ++l) {
          dstPtr[j + l] = sum[l];
        }
      }
    }

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += gridDim.x * blockDim.x) {
      uint64_t *remoteSyncCounter = combineSyncBuffer + i;
      uint64_t *localSyncCounter = combineSyncLocalBuffer + i;

      device::nixl_wait_until_sum_eq(
          remoteSyncCounter,
          localSyncCounter,
          1);

      uint64_t zero = 0ull;
      pplx::st_flag_release(remoteSyncCounter, 0ull);
      __nv_atomic_store(localSyncCounter, &zero, __NV_ATOMIC_RELEASE, PPLX_ATOMIC_SCOPE);
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      globalTokenIndex = 0;
    }
  }
}

template <typename T, typename U>
void AllToAllInterNode::combine(
    const Strided1D<U> &outTokens,
    const Strided2D<uint32_t> &indices,
    const Strided2D<float> &weights,
    const Strided2D<T> &expertX,
    unsigned m,
    const unsigned *boundM,
    SplitMode splitMode,
    cudaStream_t stream
) {
  constexpr size_t NUM_WARPS = 16;

  const size_t numLocalExperts = numExperts / worldSize;
  const size_t numDPGroups = worldSize / dpSize;
  const size_t batchNumTokens = numLocalExperts * numDPGroups * maxNumTokens;
  const size_t numBlocks = std::min(static_cast<size_t>(numSMs), batchNumTokens);

  assert(hiddenDimBytes % 16 == 0);

  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * 32, 1, 1);

  uint64_t *combineSignalBufferArg = combineSignalBuffer;
  uint64_t *combineSyncBufferArg = combineSyncBuffer;
  PPLX_ASSERT(combineSignalBuffer != nullptr, "NIXL combine remote signal buffer not initialized");
  PPLX_ASSERT(combineSyncBuffer != nullptr, "NIXL combine remote sync buffer not initialized");
  PPLX_ASSERT(combineLocalSignalBuffer != nullptr, "NIXL combine local signal buffer not initialized");
  PPLX_ASSERT(combineLocalSyncBuffer != nullptr, "NIXL combine local sync buffer not initialized");
  uint64_t *combineSignalLocalBufferArg = combineLocalSignalBuffer;
  uint64_t *combineSyncLocalBufferArg = combineLocalSyncBuffer;

  void *args[] = {
      const_cast<U **>(&outTokens.data),
      const_cast<size_t *>(&outTokens.strideElem),
      const_cast<uint32_t **>(&indices.data),
      const_cast<size_t *>(&indices.strideElem),
      const_cast<size_t *>(&indices.strideRow),
      const_cast<float **>(&weights.data),
      const_cast<size_t *>(&weights.strideElem),
      const_cast<size_t *>(&weights.strideRow),
      const_cast<T **>(&expertX.data),
      const_cast<size_t *>(&expertX.strideElem),
      const_cast<size_t *>(&expertX.strideRow),
      const_cast<size_t *>(&expertsPerToken),
      const_cast<size_t *>(&maxNumTokens),
      const_cast<size_t *>(&numExperts),
      const_cast<unsigned *>(&rank),
      const_cast<unsigned *>(&worldSize),
      const_cast<unsigned *>(&dpSize),
      const_cast<size_t *>(&hiddenDim),
      const_cast<unsigned **>(&boundM),
      &m,
      &sourceExpert,
      &sourceIndex,
      &sourceOffset,
      &sourceGroup,

      &combineSignalBufferArg,
      &combineSignalLocalBufferArg,
      &combineSyncBufferArg,
      &combineSyncLocalBufferArg,
      &tokenIndex,
      &xCombineIn,
      &xCombineOut,
      &memLayout,
      &peerDataBases,
      &peerCounterBases,
      const_cast<size_t*>(&numChannels),
      &dChannelDataReqsFlat,
      &dChannelSignalReqsFlat,
  };

  nvtxRangePush("combine");
  switch (splitMode) {
  case SplitMode::SEND:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&combineKernel<T, U, NUM_WARPS, true, false>, dimGrid, dimBlock, args, 0, stream
    ));
    break;
  case SplitMode::RECV:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&combineKernel<T, U, NUM_WARPS, false, true>, dimGrid, dimBlock, args, 0, stream
    ));
    break;
  case SplitMode::NONE:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&combineKernel<T, U, NUM_WARPS, true, true>, dimGrid, dimBlock, args, 0, stream
    ));
    break;
  default:
    PPLX_UNREACHABLE("invalid split mode");
  }
  nvtxRangePop();
}

#define INSTANTIATE_COMBINE(T, U)                                                                  \
  template void AllToAllInterNode::combine<T, U>(                                                  \
      const Strided1D<U> &outTokens,                                                               \
      const Strided2D<uint32_t> &indices,                                                          \
      const Strided2D<float> &weights,                                                             \
      const Strided2D<T> &expertX,                                                                 \
      unsigned m,                                                                                  \
      const unsigned *boundM,                                                                      \
      SplitMode splitMode,                                                                         \
      cudaStream_t stream                                                                          \
  );

INSTANTIATE_COMBINE(float, nv_bfloat16)
INSTANTIATE_COMBINE(float, half)
INSTANTIATE_COMBINE(nv_bfloat16, nv_bfloat16)
INSTANTIATE_COMBINE(nv_bfloat16, half)
INSTANTIATE_COMBINE(half, nv_bfloat16)
INSTANTIATE_COMBINE(half, half)
