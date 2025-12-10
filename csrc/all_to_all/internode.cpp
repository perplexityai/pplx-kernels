
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

#include "all_to_all/internode.h"
#include "core/utils.h"
#include "core/cuda_utils.h"

using namespace pplx;

constexpr size_t SUPER_REGION_ALIGNMENT = 256;
constexpr int MAX_METADATA_CHECK_ATTEMPTS = 10000;
constexpr int MAX_NOTIFICATION_POLL_ATTEMPTS = 10000;
constexpr int RETRY_SLEEP_MS = 10;
constexpr int METADATA_EXCHANGE_INITIAL_SLEEP_MS = 200;
constexpr int CLEANUP_SLEEP_MS = 100;

static std::unique_ptr<nixlAgent> createNixlAgent(const std::string& agentName) {
  nixlAgentConfig config(
      true,
      false,
      0,
      nixl_thread_sync_t::NIXL_THREAD_SYNC_RW,
      1,
      0,
      100000,
      false,
      std::chrono::microseconds(100000000)
  );

  return std::make_unique<nixlAgent>(agentName, config);
}

static nixl_xfer_dlist_t createSingleDescDlist(void* ptr, size_t size, int deviceId) {
  nixl_xfer_dlist_t dlist(VRAM_SEG);
  dlist.addDesc(nixlBlobDesc(
    reinterpret_cast<uintptr_t>(ptr),
    size,
    deviceId,
    ""
  ));
  return dlist;
}

AllToAllInterNode::AllToAllInterNode(
    size_t maxNumTokens,
    size_t numExperts,
    size_t expertsPerToken,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    size_t hiddenDimBytes,
    size_t hiddenDimScaleBytes
)
    : AllToAll(
          maxNumTokens,
          numExperts,
          expertsPerToken,
          rank,
          worldSize,
          dpSize,
          hiddenDim,
          hiddenDimBytes,
          hiddenDimScaleBytes
      ),
      maxBatchTokens(numLocalExperts * numDPGroups * maxNumTokens)
      , numChannels(PPLX_NIXL_NUM_CHANNELS)
      , agent(createNixlAgent(std::to_string(rank)))
      , originalDataDlist(VRAM_SEG)
      , originalCountersDlist(VRAM_SEG)
      , hPeerDataSuperBases(worldSize, nullptr)
      , hPeerCountersLocalBases(worldSize, nullptr)
{
  // Buffers for token counts.
  numTokensPerDP = mallocZeroBuffer<uint32_t>(numLocalExperts * numDPGroups);

  const size_t perTokenBytes = round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), 16);

  size_t dispatchInSize = maxNumTokens * perTokenBytes;
  size_t dispatchOutSize = maxBatchTokens * perTokenBytes;
  size_t combineInSize = maxBatchTokens * hiddenDim * sizeof(float);
  size_t combineOutSize = maxNumTokens * numExperts * hiddenDim * sizeof(float);
  size_t counter_span = numLocalExperts * numDPGroups * sizeof(uint64_t);

  // Allocate super-regions and wire up NIXL
  // Initialize unified memory layout (DATA super-region + counter blocks)
  initializeMemoryLayout(
      dispatchInSize,
      dispatchOutSize,
      combineInSize,
      combineOutSize,
      counter_span,
      maxNumTokens,
      worldSize);

  // Allocate DATA super-region based on calculated layout
  dataSuperBase = mallocZeroBuffer<std::byte>(memLayout.dataSuperTotalSize);
  PPLX_ASSERT(dataSuperBase != nullptr, "Failed to allocate DATA super-region");

  // Point buffer pointers to offsets within super-region
  xDispatchIn = dataSuperBase + memLayout.dispatchInOffset;
  xDispatchOut = dataSuperBase + memLayout.dispatchOutOffset;
  xCombineIn = dataSuperBase + memLayout.combineInOffset;
  xCombineOut = dataSuperBase + memLayout.combineOutOffset;

  // Allocate COUNTER blocks based on calculated layout
  countersRemoteBase = mallocZeroBuffer<std::byte>(memLayout.counterTotalSize);
  PPLX_ASSERT(countersRemoteBase != nullptr, "Failed to allocate REMOTE counter block");

  countersLocalBase = mallocZeroBuffer<std::byte>(memLayout.counterTotalSize);
  PPLX_ASSERT(countersLocalBase != nullptr, "Failed to allocate LOCAL counter block");

  // Point counter/signal pointers to their offsets for host-side convenience
  nixlTokenRemoteCounters = reinterpret_cast<uint64_t*>(
      countersRemoteBase + memLayout.counterTokenOffset);
  nixlRecvRemoteCounters = reinterpret_cast<uint64_t*>(
      countersRemoteBase + memLayout.counterRecvOffset);
  combineSignalBuffer = reinterpret_cast<uint64_t*>(
      countersRemoteBase + memLayout.counterSignalOffset);
  combineSyncBuffer = reinterpret_cast<uint64_t*>(
      countersRemoteBase + memLayout.counterSyncOffset);

  nixlTokenLocalCounters = reinterpret_cast<uint64_t*>(
      countersLocalBase + memLayout.counterTokenOffset);
  nixlRecvLocalCounters = reinterpret_cast<uint64_t*>(
      countersLocalBase + memLayout.counterRecvOffset);
  combineLocalSignalBuffer = reinterpret_cast<uint64_t*>(
      countersLocalBase + memLayout.counterSignalOffset);
  combineLocalSyncBuffer = reinterpret_cast<uint64_t*>(
      countersLocalBase + memLayout.counterSyncOffset);

  initializeNixlAgent();
  wireupNixl();
  createAllNixlRequests();
  setupPeerArrays();

  // Buffers for token tracking.
  sourceIndex = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceExpert = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceOffset = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceGroup = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceToken = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  tokenIndex = mallocZeroBuffer<uint32_t>(1);
}

AllToAllInterNode::~AllToAllInterNode() {
  CUDACHECK(cudaDeviceSynchronize());
  destroyPeerArrays();
  destroyAllNixlRequests();
  unwireNixl();
  destroyNixlAgent();

  CUDACHECK(cudaFree(numTokensPerDP));

  CUDACHECK(cudaFree(dataSuperBase));
  CUDACHECK(cudaFree(countersRemoteBase));
  CUDACHECK(cudaFree(countersLocalBase));

  CUDACHECK(cudaFree(sourceIndex));
  CUDACHECK(cudaFree(sourceExpert));
  CUDACHECK(cudaFree(sourceOffset));
  CUDACHECK(cudaFree(sourceGroup));
  CUDACHECK(cudaFree(sourceToken));
  CUDACHECK(cudaFree(tokenIndex));
}

void AllToAllInterNode::initializeNixlAgent() {
  const char* masterPortStr = std::getenv("MASTER_PORT");
  PPLX_ASSERT(masterPortStr != nullptr, "MASTER_PORT environment variable not set");

  const int baseEnv = std::stoi(masterPortStr);
  PPLX_ASSERT(baseEnv >= 0 && baseEnv <= 65535, "MASTER_PORT out of range");

  PPLX_ASSERT(agent != nullptr, "Failed to create NIXL agent");

  nixl_mem_list_t mems;
  nixl_b_params_t initParams;
  nixl_status_t status = agent->getPluginParams("UCX", mems, initParams);
  PPLX_ASSERT(status == NIXL_SUCCESS, "Failed to get UCX plugin parameters");

  initParams["ucx_error_handling_mode"] = "none";
  initParams["num_workers"] = std::to_string(numChannels);

  status = agent->createBackend("UCX", initParams, backend);
  PPLX_ASSERT(status == NIXL_SUCCESS && backend != nullptr, "Failed to create UCX backend");

  int deviceId;
  CUDACHECK(cudaGetDevice(&deviceId));

  // Register super-regions with NIXL agent
  nixl_reg_dlist_t dataSuperDlist(VRAM_SEG);
  dataSuperDlist.addDesc(nixlBlobDesc(
      reinterpret_cast<uintptr_t>(dataSuperBase),
      memLayout.dataSuperTotalSize,
      deviceId,
      "data_super"));
  originalDataDlist = dataSuperDlist;

  status = agent->registerMem(dataSuperDlist);
  PPLX_ASSERT(status == NIXL_SUCCESS, "Failed to register DATA super-region");

  nixl_reg_dlist_t countersRemoteDlist(VRAM_SEG);
  countersRemoteDlist.addDesc(nixlBlobDesc(
      reinterpret_cast<uintptr_t>(countersRemoteBase),
      memLayout.counterTotalSize,
      deviceId,
      "counters_remote"));
  originalCountersDlist = countersRemoteDlist;

  status = agent->registerMem(countersRemoteDlist);
  PPLX_ASSERT(status == NIXL_SUCCESS, "Failed to register REMOTE counter block");

  // Create extraParams with backend for GPU signal operations
  nixl_opt_args_t extraParams;
  extraParams.backends.push_back(backend);

  size_t signalSize = 0;
  status = agent->getGpuSignalSize(signalSize, &extraParams);
  PPLX_ASSERT(status == NIXL_SUCCESS, "Failed to query GPU signal size");
  PPLX_ASSERT(signalSize == sizeof(uint64_t), "GPU signal size is not 8 bytes");

  status = agent->prepGpuSignal(originalCountersDlist, &extraParams);
  PPLX_ASSERT(status == NIXL_SUCCESS, "Failed to prepare GPU signal");
}

void AllToAllInterNode::destroyNixlAgent() {
  if (!agent) return;

  agent->deregisterMem(originalDataDlist);
  agent->deregisterMem(originalCountersDlist);
}

void AllToAllInterNode::wireupNixl() {
  int deviceId;
  CUDACHECK(cudaGetDevice(&deviceId));

  PeerInfo localInfo = {
    .deviceId = deviceId,
    .rank = static_cast<int>(rank),
    .dataBase = dataSuperBase,
    .dataSize = memLayout.dataSuperTotalSize,
    .countersRemoteBase = countersRemoteBase,
    .countersRemoteSize = memLayout.counterTotalSize,
    .countersLocalBase = countersLocalBase,
    .countersLocalSize = memLayout.counterTotalSize
  };

  #ifdef PPLX_ENABLE_P2P
  {
    CUDACHECK(cudaIpcGetMemHandle(&localInfo.dataSuperHandle, dataSuperBase));
    CUDACHECK(cudaIpcGetMemHandle(&localInfo.countersLocalHandle, countersLocalBase));

    std::ifstream bootIdFile("/proc/sys/kernel/random/boot_id");
    if (bootIdFile.is_open()) {
      std::string bootId;
      std::getline(bootIdFile, bootId);
      strncpy(localInfo.bootId, bootId.c_str(), sizeof(localInfo.bootId) - 1);
      localInfo.bootId[sizeof(localInfo.bootId) - 1] = '\0';
    }

    struct stat st;
    if (stat("/proc/self/ns/ipc", &st) == 0) {
      localInfo.ipcNamespaceInode = st.st_ino;
    }
  }
  #endif

  nixl_status_t status = agent->sendLocalMD();
  PPLX_ASSERT(status == NIXL_SUCCESS, "Failed to send local metadata");

  std::this_thread::sleep_for(std::chrono::milliseconds(METADATA_EXCHANGE_INITIAL_SLEEP_MS));

  for (unsigned peerRank = 0; peerRank < worldSize; ++peerRank) {
    if (peerRank == rank) continue;

    const std::string remoteAgentName = std::to_string(peerRank);

    status = agent->fetchRemoteMD(remoteAgentName);
    PPLX_ASSERT(status == NIXL_SUCCESS, "Failed to fetch metadata");

    nixl_xfer_dlist_t emptyDescs(VRAM_SEG);
    int checkAttempts = 0;
    while (checkAttempts < MAX_METADATA_CHECK_ATTEMPTS) {
      status = agent->checkRemoteMD(remoteAgentName, emptyDescs);
      if (status == NIXL_SUCCESS) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_SLEEP_MS));
      ++checkAttempts;
    }
    PPLX_ASSERT(status == NIXL_SUCCESS, "Failed to check metadata");
  }

  peerInfos.resize(worldSize);
  peerInfos[rank] = localInfo;

  const std::string wireupPrefix = "pplx_super_region:";

  for (unsigned targetRank = 0; targetRank < worldSize; ++targetRank) {
    if (targetRank == rank) continue;
    std::string payload(reinterpret_cast<const char*>(&localInfo), sizeof(PeerInfo));
    const std::string remoteAgentName = std::to_string(targetRank);
    status = agent->genNotif(remoteAgentName, wireupPrefix + payload);
    PPLX_ASSERT(status == NIXL_SUCCESS, "Failed to send notification");
  }

  const int expectedNotifs = worldSize > 0 ? static_cast<int>(worldSize) - 1 : 0;
  std::vector<bool> received(worldSize, false);
  int receivedCount = 0;

  for (int attempt = 0; attempt < MAX_NOTIFICATION_POLL_ATTEMPTS && receivedCount < expectedNotifs; ++attempt) {
    nixl_notifs_t notifMap;
    agent->getNotifs(notifMap);

    for (const auto& [senderName, messages] : notifMap) {
      for (const auto& msg : messages) {
        PPLX_ASSERT(msg.rfind(wireupPrefix, 0) == 0, "Unexpected NIXL notification");

        std::string data = msg.substr(wireupPrefix.size());
        PPLX_ASSERT(data.size() == sizeof(PeerInfo), "Received PeerInfo payload of unexpected size");

        PeerInfo receivedPeer{};
        memcpy(&receivedPeer, data.data(), sizeof(PeerInfo));

        PPLX_ASSERT(receivedPeer.rank < worldSize, "Received PeerInfo with invalid rank");
        PPLX_ASSERT(receivedPeer.rank != rank, "Received unexpected self PeerInfo notification");
        PPLX_ASSERT(!received[receivedPeer.rank], "Received duplicate PeerInfo");

        peerInfos[receivedPeer.rank] = receivedPeer;
        received[receivedPeer.rank] = true;
        ++receivedCount;
      }
    }

    if (receivedCount < expectedNotifs)
      std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_SLEEP_MS));
  }
  PPLX_ASSERT(receivedCount >= expectedNotifs, "Missing PeerInfo from some ranks");
}

void AllToAllInterNode::unwireNixl() {
  if (!agent) return;

  agent->invalidateLocalMD();
  std::this_thread::sleep_for(std::chrono::milliseconds(CLEANUP_SLEEP_MS));
}

void AllToAllInterNode::createAllNixlRequests() {
  // Unified request architecture:
  // - channel_data_reqs_flat[channel * worldSize + dest]: Data write requests
  // - channel_signal_reqs_flat[channel * worldSize + dest]: Signal write requests
  // Each request covers entire super-region with offset selection at nixlPostXferReq time

  gpuDataReqs.resize(numChannels * worldSize, nullptr);
  gpuSignalReqs.resize(numChannels * worldSize, nullptr);

  int localDev = -1;
  CUDACHECK(cudaGetDevice(&localDev));

  nixl_xfer_dlist_t srcDataSuper = createSingleDescDlist(
    dataSuperBase, memLayout.dataSuperTotalSize, localDev);

  nixl_xfer_dlist_t dummySrcDlist = createSingleDescDlist(
    countersRemoteBase, sizeof(uint64_t), localDev);

  for (int workerIdx = 0; workerIdx < (int)numChannels; ++workerIdx) {
    for (unsigned destRank = 0; destRank < worldSize; ++destRank) {
      if (destRank == rank) continue;

      const size_t flatIdx = (size_t)workerIdx * worldSize + (size_t)destRank;
      const std::string dstName = std::to_string(destRank);

      nixl_opt_args_t workerParams;
      workerParams.customParam = "worker_id=" + std::to_string(workerIdx);

      nixl_xfer_dlist_t dstDataSuper = createSingleDescDlist(
        peerInfos[destRank].dataBase,
        peerInfos[destRank].dataSize,
        peerInfos[destRank].deviceId);

      nixlXferReqH* dataCpuReq;
      nixl_status_t status = agent->createXferReq(
        NIXL_WRITE, srcDataSuper, dstDataSuper, dstName,
        dataCpuReq, &workerParams);

      PPLX_ASSERT(status == NIXL_SUCCESS && dataCpuReq != nullptr, "Failed to create unified DATA request");

      nixlGpuXferReqH dataGpuReq;
      status = agent->createGpuXferReq(*dataCpuReq, dataGpuReq);
      PPLX_ASSERT(status == NIXL_SUCCESS && dataGpuReq != nullptr, "Failed to create GPU DATA request");
      gpuDataReqs[flatIdx] = dataGpuReq;
      cpuXferReqs.push_back(dataCpuReq);

      nixl_xfer_dlist_t dstSignalDlist = createSingleDescDlist(
        peerInfos[destRank].countersRemoteBase,
        sizeof(uint64_t),
        peerInfos[destRank].deviceId);

      nixlXferReqH* signalCpuReq;
      status = agent->createXferReq(
        NIXL_WRITE, dummySrcDlist, dstSignalDlist, dstName,
        signalCpuReq, &workerParams);

      PPLX_ASSERT(status == NIXL_SUCCESS && signalCpuReq != nullptr, "Failed to create unified SIGNAL request");

      nixlGpuXferReqH signalGpuReq;
      status = agent->createGpuXferReq(*signalCpuReq, signalGpuReq);
      PPLX_ASSERT(status == NIXL_SUCCESS && signalGpuReq != nullptr, "Failed to create GPU SIGNAL request");
      gpuSignalReqs[flatIdx] = signalGpuReq;
      cpuXferReqs.push_back(signalCpuReq);
    }
  }

  nixlGpuXferReqH* dUnifiedDataReqs = nullptr;
  nixlGpuXferReqH* dUnifiedSignalReqs = nullptr;

  size_t unifiedSize = numChannels * worldSize * sizeof(nixlGpuXferReqH);
  CUDACHECK(cudaMalloc(&dUnifiedDataReqs, unifiedSize));
  CUDACHECK(cudaMemcpy(dUnifiedDataReqs, gpuDataReqs.data(),
                       unifiedSize, cudaMemcpyHostToDevice));

  CUDACHECK(cudaMalloc(&dUnifiedSignalReqs, unifiedSize));
  CUDACHECK(cudaMemcpy(dUnifiedSignalReqs, gpuSignalReqs.data(),
                       unifiedSize, cudaMemcpyHostToDevice));

  dChannelDataReqsFlat = dUnifiedDataReqs;
  dChannelSignalReqsFlat = dUnifiedSignalReqs;
}

void AllToAllInterNode::destroyAllNixlRequests() {
  if (agent) {
    for (auto gpuReq : gpuDataReqs) {
      if (!gpuReq) continue;
      agent->releaseGpuXferReq(gpuReq);
    }
    for (auto gpuReq : gpuSignalReqs) {
      if (!gpuReq) continue;
      agent->releaseGpuXferReq(gpuReq);
    }
    for (auto cpuReq : cpuXferReqs) {
      if (!cpuReq) continue;
      agent->releaseXferReq(cpuReq);
    }
  }

  CUDACHECK(cudaFree(dChannelDataReqsFlat));
  CUDACHECK(cudaFree(dChannelSignalReqsFlat));
}

void AllToAllInterNode::setupPeerArrays() {
  // Populate self-rank entries in host vectors
  hPeerDataSuperBases[rank] = dataSuperBase;
  hPeerCountersLocalBases[rank] = countersLocalBase;

  #ifdef PPLX_ENABLE_P2P
  if (!peerInfos.empty() && worldSize > 1) {
    PPLX_ASSERT(rank < peerInfos.size() && peerInfos[rank].bootId[0] != '\0',
                "P2P enabled but PeerInfo lacks super-region IPC handles");

    for (int peer = 0; peer < worldSize; ++peer) {
      if (peer == rank) continue;

      // Check if peer is on same node
      bool sameNode = (strcmp(peerInfos[peer].bootId, peerInfos[rank].bootId) == 0 &&
                       peerInfos[peer].ipcNamespaceInode == peerInfos[rank].ipcNamespaceInode);

      if (!sameNode) {
        hPeerDataSuperBases[peer] = nullptr;
        hPeerCountersLocalBases[peer] = nullptr;
        continue;
      }

      cudaError_t err = cudaIpcOpenMemHandle(&hPeerDataSuperBases[peer],
                                             peerInfos[peer].dataSuperHandle,
                                             cudaIpcMemLazyEnablePeerAccess);
      if (err != cudaSuccess) {
        std::cerr << "[P2P] cudaIpcOpenMemHandle(data_super) failed for peer " << peer
                  << ": " << cudaGetErrorString(err) << std::endl;
        hPeerDataSuperBases[peer] = nullptr;
        hPeerCountersLocalBases[peer] = nullptr;
        continue;
      }

      err = cudaIpcOpenMemHandle(&hPeerCountersLocalBases[peer],
                                 peerInfos[peer].countersLocalHandle,
                                 cudaIpcMemLazyEnablePeerAccess);
      if (err != cudaSuccess) {
        std::cerr << "[P2P] cudaIpcOpenMemHandle(counters_local) failed for peer " << peer
                  << ": " << cudaGetErrorString(err) << std::endl;
        cudaIpcCloseMemHandle(hPeerDataSuperBases[peer]);
        hPeerDataSuperBases[peer] = nullptr;
        hPeerCountersLocalBases[peer] = nullptr;
        continue;
      }
    }
  }
  #endif

  CUDACHECK(cudaMalloc(&peerDataBases, sizeof(*peerDataBases) * worldSize));
  CUDACHECK(cudaMalloc(&peerCounterBases, sizeof(*peerCounterBases) * worldSize));
  CUDACHECK(cudaMemcpy(peerDataBases, hPeerDataSuperBases.data(),
                       sizeof(*peerDataBases) * worldSize, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(peerCounterBases, hPeerCountersLocalBases.data(),
                       sizeof(*peerCounterBases) * worldSize, cudaMemcpyHostToDevice));
}

void AllToAllInterNode::destroyPeerArrays() {
  #ifdef PPLX_ENABLE_P2P
  size_t closed_data = 0, closed_counters = 0;
  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer == rank) continue;

    if (hPeerDataSuperBases[peer]) {
      cudaError_t err = cudaIpcCloseMemHandle(hPeerDataSuperBases[peer]);
      if (err != cudaSuccess) {
        std::cerr << "[P2P] cleanup close(data_super) peer " << peer << ": "
                  << cudaGetErrorString(err) << std::endl;
      } else {
        ++closed_data;
      }
    }
    if (hPeerCountersLocalBases[peer]) {
      cudaError_t err = cudaIpcCloseMemHandle(hPeerCountersLocalBases[peer]);
      if (err != cudaSuccess) {
        std::cerr << "[P2P] cleanup close(counters_local) peer " << peer << ": "
                  << cudaGetErrorString(err) << std::endl;
      } else {
        ++closed_counters;
      }
    }
  }
  #endif

  CUDACHECK(cudaFree(peerDataBases));
  CUDACHECK(cudaFree(peerCounterBases));
}

void AllToAllInterNode::initializeMemoryLayout(
    size_t dispatchInSize,
    size_t dispatchOutSize,
    size_t combineInSize,
    size_t combineOutSize,
    size_t counterSpan,
    size_t maxNumTokens,
    size_t worldSize) {
  // Initialize DATA super-region offsets
  memLayout.dispatchInOffset = 0;
  memLayout.dispatchOutOffset = round_up(dispatchInSize, SUPER_REGION_ALIGNMENT);
  memLayout.combineInOffset = round_up(
      memLayout.dispatchOutOffset + dispatchOutSize,
      SUPER_REGION_ALIGNMENT);
  memLayout.combineOutOffset = round_up(
      memLayout.combineInOffset + combineInSize,
      SUPER_REGION_ALIGNMENT);
  memLayout.dataSuperTotalSize = memLayout.combineOutOffset + combineOutSize;

  // Initialize COUNTER block offsets (shared layout for both remote and local blocks)
  memLayout.counterTokenOffset = 0;
  memLayout.counterRecvOffset = round_up(
      memLayout.counterTokenOffset + counterSpan,
      SUPER_REGION_ALIGNMENT);
  memLayout.counterSignalOffset = round_up(
      memLayout.counterRecvOffset + counterSpan,
      SUPER_REGION_ALIGNMENT);
  memLayout.counterSyncOffset = round_up(
      memLayout.counterSignalOffset + sizeof(uint64_t) * maxNumTokens,
      SUPER_REGION_ALIGNMENT);
  memLayout.counterTotalSize = memLayout.counterSyncOffset + sizeof(uint64_t) * worldSize;
}
