#pragma once

#include "core/buffer.h"
#include "core/utils.h"

#include <cstdint>

#include <algorithm>
#include <ostream>
#include <random>
#include <vector>

#if USE_ROCM

#include <hip/hip_fp8.h>

// NOTE (yiakwy) : __hip_fp8_e4m3_fnuz only represents -240..0..240
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;

__hip_fp8_storage_t
convert_float_to_fp8(float in, /* Input val */
                    __hip_fp8_interpretation_t
                        interpret, /* interpretation of number E4M3/E5M2 */
                    __hip_saturation_t sat /* Saturation behavior */
) {
    return __hip_cvt_float_to_fp8(in, sat, interpret);
}

float convert_fp8_to_float(
    __hip_fp8_storage_t in, /* Input val */
    __hip_fp8_interpretation_t
        interpret /* interpretation of number E4M3/E5M2 */
) {
    __half hf = __hip_cvt_fp8_to_halfraw(in, interpret);
    return static_cast<float>(hf);
}

#else
#include <cuda_fp8.h>
#endif

namespace pplx {

/// Test data for all-to-all dispatch/combine.
template <typename T> struct RankTestData {
  const size_t m;
  const size_t hiddenDim;
  const size_t hiddenDimScale;
  const size_t numExperts;
  const size_t expertsPerToken;
  HostBuffer<T> x;
  HostBuffer<float> xScale;
  HostBuffer<uint32_t> indices;
  HostBuffer<float> weights;
  HostBuffer<uint32_t> numRouted;

  RankTestData(
      std::mt19937 &gen,
      size_t maxNumTokens,
      size_t numExperts,
      size_t expertsPerToken,
      size_t hiddenDim,
      size_t blockSize
  );

  std::ostream &print(std::ostream &os) const;
};

template <typename T>
RankTestData<T>::RankTestData(
    std::mt19937 &gen,
    size_t maxNumTokens,
    size_t numExperts,
    size_t expertsPerToken,
    size_t hiddenDim,
    size_t blockSize
)
    : m(std::uniform_int_distribution<>(1, maxNumTokens)(gen)),
      hiddenDim(hiddenDim),
      hiddenDimScale(ceil_div(hiddenDim, blockSize)),
      numExperts(numExperts),
      expertsPerToken(expertsPerToken),
      x(m * hiddenDim),
      xScale(m * hiddenDimScale),
      indices(m * expertsPerToken),
      weights(m * expertsPerToken),
      numRouted(numExperts) {
  std::uniform_int_distribution<> expert(0, numExperts - 1);

  // Populate dummy routing information.
  for (size_t i = 0; i < numExperts; ++i) {
    numRouted[i] = 0;
  }

  for (size_t i = 0; i < m; ++i) {
    std::vector<uint32_t> experts(numExperts);
    for (size_t j = 0; j < numExperts; ++j) {
      experts[j] = j;
    }
    std::shuffle(experts.begin(), experts.end(), gen);
    std::uniform_real_distribution<> weight(0.0f, 1.0f);
    for (size_t j = 0; j < expertsPerToken; ++j) {
      uint32_t expert = experts[j];
      uint32_t loc = numRouted[expert]++;
      indices[i * expertsPerToken + j] = expert;
      weights[i * expertsPerToken + j] = weight(gen);
    }
  }

  // Populate the tokens.
  if constexpr (std::is_integral<T>::value) {
    // std::uniform_int_distribution<> value(-256, 256);
    std::uniform_int_distribution<> value(-224, 224);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < hiddenDim; ++j) {
        x[i * hiddenDim + j] = value(gen);
      }
    }
  } else {
    std::uniform_real_distribution<> value(-10.0f, 10.0f);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < hiddenDim; ++j) {
        if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
#if USE_ROCM
          x[i * hiddenDim + j] = static_cast<__hip_fp8_e4m3_fnuz>(convert_float_to_fp8(value(gen), __HIP_E4M3_FNUZ, __HIP_SATFINITE));
#else
          x[i * hiddenDim + j] = __nv_cvt_float_to_fp8(value(gen), __NV_SATFINITE, __NV_E4M3);
#endif
        } else {
          x[i * hiddenDim + j] = value(gen);
        }
      }
    }
  }

  // Populate the scales.
  std::uniform_real_distribution<> value(0, 100.0f);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < hiddenDimScale; ++j) {
      xScale[i * hiddenDimScale + j] = value(gen);
    }
  }
}

template <typename T> std::ostream &RankTestData<T>::print(std::ostream &os) const {
  for (unsigned j = 0; j < m; ++j) {
    os << "#" << j << " ->";
    for (unsigned k = 0; k < expertsPerToken; ++k) {
      auto e = indices[j * expertsPerToken + k];
      auto w = weights[j * expertsPerToken + k];
      os << " " << e << ":" << w;
    }
    os << std::endl;
    os << "    ";
    for (unsigned k = 0; k < hiddenDim; ++k) {
      os << (float)x[j * hiddenDim + k] << " ";
    }
    os << std::endl;
    os << "    ";
    for (unsigned k = 0; k < hiddenDimScale; ++k) {
      os << xScale[j * hiddenDimScale + k] << " ";
    }
    os << std::endl;
  }
  for (unsigned j = 0; j < numExperts; ++j) {
    const size_t numTokens = numRouted[j];
    os << "Expert " << j << ": " << numTokens << " tokens" << std::endl;
  }
  os << std::flush << std::endl;
  return os;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const RankTestData<T> &data) {
  return data.print(os);
}

} // namespace pplx
