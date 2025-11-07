#pragma once

#include <climits>
#include <cstdint>

#if defined(__CUDA_ARCH__) || defined(__HIPCC__) || defined(__CUDACC__)
#include <hip/hip_runtime.h>

#define PPLX_HOST __host__
#define PPLX_DEVICE __device__

#define WARP_SIZE 64 // warpSize is restriced in device codes since SDK 7.0

#else

#include <cuda_runtime.h>

#define PPLX_HOST
#define PPLX_DEVICE

#define WARP_SIZE 32

#endif

#define PPLX_HOST_DEVICE PPLX_HOST PPLX_DEVICE

namespace pplx {

/// Return the next power of 2 following the given number.
PPLX_HOST_DEVICE inline uint32_t next_pow_2(const uint32_t num) {
  if (num <= 1) {
    return num;
  }
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename T> PPLX_HOST_DEVICE T ceil_div(T x, T y) { return (x + y - 1) / y; }

template <typename T> PPLX_HOST_DEVICE T round_up(T x, T y) { return ceil_div<T>(x, y) * y; }

} // namespace pplx

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_bf16.h>
#include <hip/hip_common.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

namespace amdgpu {

#define DEVICE_INLINE __device__ __inline__

template <typename T>
DEVICE_INLINE T shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);

template <>
DEVICE_INLINE float shfl_xor_sync(unsigned mask, float var, int laneMask, int width) {
  return __shfl_xor(var, laneMask, width);
}

template <>
DEVICE_INLINE int shfl_xor_sync(unsigned mask, int var, int laneMask, int width) {
  return __shfl_xor(var, laneMask, width);
}

} // namespace amdgpu

template <typename T>
DEVICE_INLINE T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = 32/*use CUDA warp size as default*/) {
  return amdgpu::shfl_xor_sync(mask, var, laneMask, width);
}

#endif // defined(__HIP_PLATFORM_AMD__)
