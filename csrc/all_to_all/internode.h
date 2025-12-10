#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "all_to_all/all_to_all.h"
#include "core/buffer.h"

#include "core/nixl_utils.cuh"
#include <nixl.h>

namespace pplx {

/// @brief All-to-all broadcast kernel.
///
/// This kernel keeps track of the intermediary buffers used by
/// the all-to-all broadcast and wraps around the kernel launcher.
class AllToAllInterNode final : public AllToAll {
public:
  AllToAllInterNode(
      size_t maxNumTokens,
      size_t numExperts,
      size_t expertsPerToken,
      unsigned rank,
      unsigned worldSize,
      unsigned dpSize,
      size_t hiddenDim,
      size_t hiddenDimBytes,
      size_t hiddenDimScaleBytes
  );

  ~AllToAllInterNode();

  /// Launches the all-to-all dispatch kernel.
  ///
  /// @param outTokensPerExpert An output array which contains the number of
  /// tokens routed to the experts. Shape: [numLocalExperts].
  ///
  /// @param expertX The output buffer for per-expert tokens.
  /// Shape: [numLocalExperts, maxNumTokens * numDPGroups, hiddenDim].
  ///
  /// @param expertXScale The output buffer for per-expert scale of X.
  /// Shape: [numLocalExperts, maxNumTokens * numDPGroups, hiddenDimScale].

  /// @param dpX The input tokens X.
  /// Shape: [m, hiddenDim].
  ///
  /// @param dpXScale The input scale of X.
  /// Shape: [m, hiddenDimScale].
  ///
  /// @param indices The input indices of size [numExperts, maxNumTokens].
  /// Shape: [numExperts, maxNumTokens].
  ///
  /// @param m The size of the buffers allocated for dpX/dpXScale.
  ///
  /// @param boundM The dynamic number of tokens populated in dpX/dpXScale.
  /// Shape: [1].
  ///
  /// @param doSend Whether to send the data to other ranks (for overlapping).
  ///
  /// @param doRecv Whether to receive the data from other ranks (for
  /// overlapping).
  ///
  /// @param stream The CUDA stream to launch the kernel on.
  void dispatch(
      const Strided1D<int32_t> &outTokensPerExpert,
      const Strided2D<std::byte> &expertX,
      const Strided3D<float> &expertXScale,
      const Strided1D<std::byte> &dpX,
      const Strided2D<float> &dpXScale,
      const Strided2D<uint32_t> &indices,
      unsigned m,
      const unsigned *boundM,
      SplitMode splitMode,
      cudaStream_t stream
  );

  /// Launches the all-to-all combine kernel.
  ///
  /// @param outTokens The output tokens.
  /// Shape: [numExperts, maxNumTokens].
  ///
  /// @param indices The input indices of size [numExperts, maxNumTokens].
  /// Shape: [numExperts, maxNumTokens].
  ///
  /// @param weights The input weights of size [numExperts, maxNumTokens].
  /// Shape: [numExperts, maxNumTokens].
  ///
  /// @param expertX The input tokens X.
  /// Shape: [numLocalExperts, maxNumTokens * numDPGroups, hiddenDim].
  ///
  /// @param numTokensPerExpert The number of tokens per expert.
  /// Shape: [numLocalExperts].
  ///
  /// @param m The size of the buffers allocated for dpX/dpXScale.
  ///
  /// @param boundM The dynamic number of tokens populated in dpX/dpXScale.
  /// Shape: [1].
  ///
  /// @param stream The CUDA stream to launch the kernel on.
  template <typename T, typename U>
  void combine(
      const Strided1D<U> &outTokens,
      const Strided2D<uint32_t> &indices,
      const Strided2D<float> &weights,
      const Strided2D<T> &expertX,
      unsigned m,
      const unsigned *boundM,
      SplitMode splitMode,
      cudaStream_t stream
  );

private:
  /// The maximum number of tokens in a batch.
  const size_t maxBatchTokens;

  /// @section Internal buffers communicating between dispatch and combine.
  uint32_t *sourceIndex = nullptr;
  uint32_t *sourceExpert = nullptr;
  uint32_t *sourceOffset = nullptr;
  uint32_t *sourceGroup = nullptr;
  uint32_t *sourceToken = nullptr;
  uint32_t *tokenIndex = nullptr;

  /// @section Pre-allocated symmetric shared memory workspace.
  uint32_t *numTokensPerDP = nullptr;
  uint64_t *combineSignalBuffer = nullptr;
  uint64_t *combineSyncBuffer = nullptr;
  uint64_t *combineLocalSignalBuffer = nullptr;
  uint64_t *combineLocalSyncBuffer = nullptr;
  std::byte *xDispatchIn = nullptr;
  std::byte *xDispatchOut = nullptr;
  std::byte *xCombineIn = nullptr;
  std::byte *xCombineOut = nullptr;

  struct alignas(16) PeerInfo {
    int deviceId;
    int rank;

    void* dataBase;              ///< Base of DATA super-region (dispatch/combine buffers)
    size_t dataSize;             ///< Size of DATA super-region
    void* countersRemoteBase;    ///< Base of REMOTE counter block (RDMA targets)
    size_t countersRemoteSize;   ///< Size of REMOTE counter block
    void* countersLocalBase;     ///< Base of LOCAL counter block (self/P2P fast-path data)
    size_t countersLocalSize;    ///< Size of LOCAL counter block

    #ifdef PPLX_ENABLE_P2P
    cudaIpcMemHandle_t dataSuperHandle;
    cudaIpcMemHandle_t countersLocalHandle;
    char bootId[40];
    ino_t ipcNamespaceInode;
    #endif
  };

  /// @section NIXL-specific members
  const size_t numChannels;
  std::vector<PeerInfo> peerInfos;

  std::unique_ptr<nixlAgent> agent;
  nixlBackendH* backend = nullptr;

  std::vector<nixlGpuXferReqH> gpuDataReqs;
  std::vector<nixlGpuXferReqH> gpuSignalReqs;
  std::vector<nixlXferReqH*> cpuXferReqs;

  nixlGpuXferReqH* dChannelDataReqsFlat = nullptr;
  nixlGpuXferReqH* dChannelSignalReqsFlat = nullptr;

  nixl_reg_dlist_t originalDataDlist;
  nixl_reg_dlist_t originalCountersDlist;

  uint64_t* nixlTokenRemoteCounters = nullptr;
  uint64_t* nixlTokenLocalCounters = nullptr;
  uint64_t* nixlRecvRemoteCounters = nullptr;
  uint64_t* nixlRecvLocalCounters = nullptr;

  std::byte** peerDataBases = nullptr;
  std::byte** peerCounterBases = nullptr;

  MemLayout memLayout;
  std::byte* dataSuperBase = nullptr;
  std::byte* countersRemoteBase = nullptr;
  std::byte* countersLocalBase = nullptr;

  std::vector<void*> hPeerDataSuperBases;
  std::vector<void*> hPeerCountersLocalBases;

  void initializeNixlAgent();
  void destroyNixlAgent();

  void wireupNixl();
  void unwireNixl();

  void createAllNixlRequests();
  void destroyAllNixlRequests();

  void setupPeerArrays();
  void destroyPeerArrays();

  void initializeMemoryLayout(
      size_t dispatchInSize,
      size_t dispatchOutSize,
      size_t combineInSize,
      size_t combineOutSize,
      size_t counterSpan,
      size_t maxNumTokens,
      size_t worldSize);

};

} // namespace pplx
