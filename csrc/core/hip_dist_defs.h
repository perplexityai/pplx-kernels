#pragma once

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#ifdef __HIP_PLATFORM_NVIDIA__
#undef __HIP_PLATFORM_NVIDIA__
#endif

#if defined(__HIP_PLATFORM_AMD__)

#include <iostream>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include <rocshmem/rocshmem.hpp>

// function alias

#define nvshmem_init(...) rocshmem::rocshmem_init(__VA_ARGS__)

#define nvshmem_malloc(...) rocshmem::rocshmem_malloc(__VA_ARGS__)

#define nvshmem_free(...) rocshmem::rocshmem_free(__VA_ARGS__)

#define HIP_MPI_CHECK(condition) {                                        \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, error);                             \
        }                                                                 \
    }

#endif // defined(__HIP_PLATFORM_AMD__)
