#include "core/cuda_utils.h"
#include "kernels.h"

#include <time.h>

#include <cstdint>

// NOTE (yiakwy) : remove the file
// #include <cuda_runtime.h>

namespace amdgpu {

  __device__ void usleep(unsigned long long delay_us)
  {
      // NOTE(yiakwy) : 1600 MHz in MI300, 2000 MHz in MI350
      const double clock_frequency_mhz = 2000ULL;
  
      unsigned long long wait_cycles = static_cast<unsigned long long>(delay_us * clock_frequency_mhz);
      
      unsigned long long start_clock = clock64();
      while (clock64() - start_clock < wait_cycles) {
          ; // Do nothing, just spin
      }
  }

} // namespace amdgpu

__global__ void sleep_kernel(uint64_t ms) {
  for (int i = 0; i < ms; i++) {
#ifdef USE_ROCM
    amdgpu::usleep(1000);
#else
    __nanosleep(1000000); // NOTE (yiakwy) : asm("nanosleep.u32 64;" ::: "memory"); supported by NV PTX
#endif
  }
}

void pplx::sleepOnStream(double seconds, cudaStream_t stream) {
  uint64_t ms = seconds * 1000;
  void *args[] = {&ms};
  dim3 grid(1);
  dim3 block(1);
  CUDACHECK(cudaLaunchKernel((void *)sleep_kernel, grid, block, args, 0, stream));
}
