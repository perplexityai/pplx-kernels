#pragma once

#define USE_ROCM 1

#ifdef USE_ROCM

#include <rocshmem/rocshmem.hpp>

#include "hip_dist_defs.h"

__global__ void alltoallmem(int *source, int *dest, size_t nelem, rocshmem::rocshmem_team_t team);

#else
#include <nvshmem.h>
#endif

#include <vector>

namespace pplx {

class Distributed {
public:
  Distributed(unsigned rank, unsigned worldSize);

  virtual ~Distributed();

  template <typename T> std::vector<T> allToAll(const std::vector<T> &input) {
    std::vector<T> output(input.size());
    allToAllImpl(input.data(), output.data(), sizeof(T), input.size());
    return output;
  }

  template <typename T> std::vector<T> allGather(const T &input) {
    std::vector<T> tmp(worldSize, input);
    return allToAll(tmp);
  }

protected:
  virtual void allToAllImpl(const void *input, void *output, size_t size, size_t count) = 0;

protected:
  unsigned rank;
  unsigned worldSize;
};

class DistributedNVSHMEM final : public Distributed {
public:
  DistributedNVSHMEM(unsigned rank, unsigned worldSize);

private:
  void allToAllImpl(const void *input, void *output, size_t size, size_t count) override;
};

} // namespace pplx
