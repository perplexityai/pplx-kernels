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

#include <rocprofiler-sdk-roctx/roctx.h>

#define nvtxRangePush(...) roctxRangePushA(__VA_ARGS__)

#define nvtxRangePop(...) roctxRangePop(__VA_ARGS__)

#endif // defined(__HIP_PLATFORM_AMD__)
