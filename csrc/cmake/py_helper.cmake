# Adapt from: https://github.com/neuralmagic/vllm-flash-attention/blob/main/cmake/utils.cmake

function (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

macro (append_torch_cmake_prefix_path)
  run_python(TORCH_CMAKE_PREFIX_PATH
    "import torch; print(torch.utils.cmake_prefix_path);" "Failed to locate torch path")
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
  message(STATUS "TORCH_CMAKE_PREFIX_PATH : ${TORCH_CMAKE_PREFIX_PATH}")
endmacro()

#
# Get additional GPU compiler flags from torch.
#
function (get_torch_gpu_compiler_flags OUT_GPU_FLAGS GPU_LANG)
  if (${GPU_LANG} STREQUAL "CUDA")
    #
    # Get common NVCC flags from torch.
    #
    run_python(GPU_FLAGS
      "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND GPU_FLAGS "-DENABLE_FP8")
    endif()
    if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
      list(REMOVE_ITEM GPU_FLAGS
        "-D__CUDA_NO_HALF_OPERATORS__"
        "-D__CUDA_NO_HALF_CONVERSIONS__"
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
        "-D__CUDA_NO_HALF2_OPERATORS__")
    endif()

  elseif(${GPU_LANG} STREQUAL "HIP")
    #
    # Get common HIP/HIPCC flags from torch.
    #
    run_python(GPU_FLAGS
      "import torch.utils.cpp_extension as t; print(';'.join(t.COMMON_HIP_FLAGS + t.COMMON_HIPCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    list(APPEND GPU_FLAGS
      "-DUSE_ROCM"
      "-DENABLE_FP8"
      "-U__HIP_NO_HALF_CONVERSIONS__"
      "-U__HIP_NO_HALF_OPERATORS__"
      "-Werror=unused-variable"
      "-fno-gpu-rdc")

  endif()
  set(${OUT_GPU_FLAGS} ${GPU_FLAGS} PARENT_SCOPE)
endfunction()