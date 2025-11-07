#pragma once

#ifdef USE_ROCM

#include <rocshmem/rocshmem.hpp>

#else
#include <nvshmem.h>
#endif

#include "core/cuda_utils.h"

#ifdef USE_ROCM

#define NVSHMEMCHECK(stmt)                                                                     \
  do {                                                                                         \
    int result = (stmt);                                                                        \
    if (ROCSHMEM_SUCCESS != result) {                                                         \
      fprintf(stderr, "[%s:%d] rocshmem failed with error %d \n", __FILE__, __LINE__, result); \
      MPI_Abort(MPI_COMM_WORLD, result);                                                       \
      exit(-1);                                                                                \
    }                                                                                          \
  } while (0)

#else

#define NVSHMEMCHECK(stmt)                                                                         \
  do {                                                                                             \
    int result = (stmt);                                                                           \
    if (NVSHMEMX_SUCCESS != result) {                                                              \
      fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, result);      \
      exit(-1);                                                                                    \
    }                                                                                              \
  } while (0)

#endif // USE_ROCM
