#pragma once

#include <cstdint>

#ifdef USE_ROCM
// ROCM device side atomic
#include <atomic>
#endif

namespace pplx {

__forceinline__ __device__ void st_flag_volatile(uint32_t *flag_addr, uint32_t flag) {
  #ifdef __HIPCC__
  __threadfence_system();
  // *reinterpret_cast<volatile uint32_t *>(flag_addr) = flag;
  __builtin_nontemporal_store(flag, flag_addr);
  #else
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
  #endif
}

__forceinline__ __device__ uint32_t ld_flag_volatile(uint32_t *flag_addr) {
  uint32_t flag;
  #ifdef __HIPCC__
  // NOTE(yiakwy) : volatile access ensures direct access L2 cache/system memory bypassing L1 cache
  __threadfence_system();
  // flag = *reinterpret_cast<volatile uint32_t *>(flag_addr);
  flag = __builtin_nontemporal_load(flag_addr);
  #else
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  #endif
  return flag;
}

__forceinline__ __device__ uint32_t ld_flag_acquire(uint32_t *flag_addr) {
  uint32_t flag;
  #ifdef __HIPCC__
  // NOTE (yiakwy) : ensuare all memory accesses before this inst visible to all threads, closest to ld.aquire ptx meaning
  __threadfence_system();
  flag = *flag_addr;
  #else
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  #endif
  return flag;
}

__forceinline__ __device__ void st_flag_release(uint32_t *flag_addr, uint32_t flag) {
  #ifdef __HIPCC__
  *flag_addr = flag;
  // NOTE (yiakwy) : ensure all writes to memory visible by all threads
  __threadfence_system();
  #else
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
  #endif
}

__forceinline__ __device__ uint32_t add_flag_release(uint32_t *addr, uint32_t val) {
  uint32_t flag;
  #ifdef __HIPCC__
  flag = atomicAdd((unsigned int*)addr, val);
  // NOTE (yiakwy) : ensure all writes to memory visible by all threads
  __threadfence_system();
  #else
  asm volatile("atom.release.sys.global.add.u32 %0, [%1], %2;" : "=r"(flag) : "l"(addr), "r"(val));
  #endif
  return flag;
}

} // namespace pplx
