#pragma once

#include <cstdio>
#include "core/common_utils.h"
#include "core/atomic.cuh"
#include <nixl.h>
#include <gpu/ucx/nixl_device.cuh>

#define PPLX_ENABLE_DEVICE_ASSERT 0

#if PPLX_ENABLE_DEVICE_ASSERT == 1
#define PPLX_DEVICE_ASSERT(cond)                                                                   \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                         \
      asm("trap;");                                                                                \
    }                                                                                              \
  } while (0)

// Assert macro for NIXL post operations (accepts SUCCESS or IN_PROG)
#define PPLX_DEVICE_ASSERT_POST_STATUS(status)                                                     \
  do {                                                                                             \
    if ((status) != NIXL_SUCCESS && (status) != NIXL_IN_PROG) {                                   \
      printf("PPLX NIXL Post Assert Failed: expected SUCCESS(0) or IN_PROG(1), got=%d at %s:%d\n",\
             (int)(status), __FILE__, __LINE__);                                                   \
      asm("trap;");                                                                                \
    }                                                                                              \
  } while (0)
#else
#define PPLX_DEVICE_ASSERT(cond)
#define PPLX_DEVICE_ASSERT_POST_STATUS(status) \
  do {                                         \
    (void)(status);                            \
  } while (0)
#endif

namespace pplx {
namespace device {

// A wrapper for the kernels that is used to guard against compilation on
// architectures that will never use the kernel.
template <typename Kernel> struct enable_sm90_or_later : Kernel {
  template <typename... Args> __device__ void operator()(Args &&...args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

__forceinline__ __device__ unsigned warp_sum(unsigned value) {
  value += __shfl_xor_sync(0xffffffff, value, 16);
  value += __shfl_xor_sync(0xffffffff, value, 8);
  value += __shfl_xor_sync(0xffffffff, value, 4);
  value += __shfl_xor_sync(0xffffffff, value, 2);
  value += __shfl_xor_sync(0xffffffff, value, 1);
  return value;
}

__forceinline__ __device__ bool warp_and(bool value) {
  value &= __shfl_xor_sync(0xffffffff, value, 16);
  value &= __shfl_xor_sync(0xffffffff, value, 8);
  value &= __shfl_xor_sync(0xffffffff, value, 4);
  value &= __shfl_xor_sync(0xffffffff, value, 2);
  value &= __shfl_xor_sync(0xffffffff, value, 1);
  return value;
}

__forceinline__ __device__ float half_warp_reduce_max(float value) {
  auto mask = __activemask();
  value = max(value, __shfl_xor_sync(mask, value, 8));
  value = max(value, __shfl_xor_sync(mask, value, 4));
  value = max(value, __shfl_xor_sync(mask, value, 2));
  value = max(value, __shfl_xor_sync(mask, value, 1));
  return value;
}

__device__ __forceinline__ int4 ld_nc_global(const int4* ptr) {
  int4 ret;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void st_na_global(int4* ptr, const int4& value) {
  asm volatile("st.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
              :: "l"(ptr), "r"(value.x), "r"(value.y),
                 "r"(value.z), "r"(value.w) : "memory");
}

__device__ __forceinline__ void warp_streaming_copy_and_sync(
    void* dst_ptr, const void* src_ptr, size_t size, int lane_id
) {
  const int num_int4 = size / sizeof(int4);
  int4* dst = reinterpret_cast<int4*>(dst_ptr);
  const int4* src = reinterpret_cast<const int4*>(src_ptr);

  const int warp_stride = 32;
  int i = lane_id;
  for (; i < num_int4; i += warp_stride * 8) {
#pragma unroll 4
    for (int u = 0; u < 8; ++u) {
      int j = i + u * warp_stride;
      if (j < num_int4) {
        int4 val = pplx::device::ld_nc_global(src + j);
        pplx::device::st_na_global(dst + j, val);
      }
    }
  }
  __syncwarp();
}

// Wait until the sum of a remote-modified counter and a local-only counter equals expected_value.
__device__ inline void nixl_wait_until_sum_eq(
    uint64_t* remote_ptr,
    uint64_t* local_ptr,
    uint64_t expected_value
) {
  uint64_t r;
  uint64_t l;

  do {
    r = pplx::ld_flag_acquire(remote_ptr);
    __nv_atomic_load(local_ptr, &l, __NV_ATOMIC_ACQUIRE, PPLX_ATOMIC_SCOPE);
  } while (r + l != expected_value);
}

// Wait until the sum of remote+local counters becomes non-zero, returns the sum.
__device__ inline uint64_t nixl_wait_until_sum_ne_zero(
    uint64_t* remote_ptr,
    uint64_t* local_ptr
) {
  uint64_t r;
  uint64_t l;

  do {
    r = pplx::ld_flag_acquire(remote_ptr);
    __nv_atomic_load(local_ptr, &l, __NV_ATOMIC_ACQUIRE, PPLX_ATOMIC_SCOPE);
  } while (r + l == 0);

  return r + l;
}


} // namespace device
} // namespace pplx
