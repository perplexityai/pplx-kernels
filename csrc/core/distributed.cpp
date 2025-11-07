#include "core/distributed.h"
#include "core/cuda_utils.h"
#include "core/utils.h"

using namespace pplx;

#ifdef USE_ROCM

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

__global__ void alltoallmem(int *source, int *dest, size_t nelem, rocshmem_team_t team) {
    __shared__ rocshmem_ctx_t ctx;
    int64_t ctx_type = 0;

    rocshmem_wg_init();
    rocshmem_wg_ctx_create(ctx_type, &ctx);
    int num_pes = rocshmem_ctx_n_pes(ctx);

    rocshmem_ctx_int_alltoall_wg(ctx, team, dest, source, nelem);

    rocshmem_ctx_quiet(ctx);
    __syncthreads();

    rocshmem_wg_ctx_destroy(&ctx);
    rocshmem_wg_finalize();
}

#endif

Distributed::Distributed(unsigned rank, unsigned worldSize)
    : rank(rank),
      worldSize(worldSize) {}

Distributed::~Distributed() {}

DistributedNVSHMEM::DistributedNVSHMEM(unsigned rank, unsigned worldSize)
    : Distributed(rank, worldSize) {}

void DistributedNVSHMEM::allToAllImpl(const void *input, void *output, size_t size, size_t count) {
#ifdef USE_ROCM
  PPLX_ASSERT(count == worldSize, "count must be equal to world size");

  void *srcBuffer = nvshmem_malloc(size * count);
  PPLX_ASSERT(srcBuffer != nullptr, "Failed to allocate src buffer");
  void *dstBuffer = nvshmem_malloc(size * count);
  PPLX_ASSERT(dstBuffer != nullptr, "Failed to allocate dst buffer");

  CUDACHECK(cudaMemcpy(srcBuffer, input, size * count, cudaMemcpyHostToDevice));

  rocshmem_team_t team_reduce_world_dup;
  team_reduce_world_dup = ROCSHMEM_TEAM_INVALID;
  rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, count, nullptr, 0,
                              &team_reduce_world_dup);

  HIP_MPI_CHECK(hipDeviceSynchronize());

  int threadsPerBlock=256;
  dim3 grid(1);
  dim3 block(threadsPerBlock);

  alltoallmem<<<grid, block, 0, 0 >>>((int *)srcBuffer, (int *)dstBuffer, size, team_reduce_world_dup);

  HIP_MPI_CHECK(hipDeviceSynchronize());

  nvshmem_free(srcBuffer);
  nvshmem_free(dstBuffer);

  rocshmem_finalize();
#else
  PPLX_ASSERT(count == worldSize, "count must be equal to world size");

  void *srcBuffer = nvshmem_malloc(size * count);
  PPLX_ASSERT(srcBuffer != nullptr, "Failed to allocate src buffer");
  void *dstBuffer = nvshmem_malloc(size * count);
  PPLX_ASSERT(dstBuffer != nullptr, "Failed to allocate dst buffer");

  CUDACHECK(cudaMemcpy(srcBuffer, input, size * count, cudaMemcpyHostToDevice));

  nvshmem_alltoallmem(NVSHMEM_TEAM_WORLD, dstBuffer, srcBuffer, size);
  nvshmem_quiet();

  CUDACHECK(cudaMemcpy(output, dstBuffer, size * count, cudaMemcpyDeviceToHost));

  nvshmem_free(dstBuffer);
  nvshmem_free(srcBuffer);
#endif
}
