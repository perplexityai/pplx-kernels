cmake_minimum_required(VERSION 3.22)
project(PPLXKernels
        VERSION 0.0.1
        DESCRIPTION "PPLX Kernels"
        LANGUAGES CXX)


set(ROCM_HOME "/opt/rocm" CACHE PATH "ROCM SDK INSTALLATION HOME")
if (NOT IS_DIRECTORY ${ROCM_HOME})
  message(WARNING "ROCM_HOME ${ROCM_HOME} is not a directory")
endif()

if (LINUX)
  # SDK Root in CMAKE config file; LINUX system defaults to ENV{ROCM_PATH}; WIN32 system defaults to ENV{HIP_PATH}
  set(ENV{ROCM_PATH} ${ROCM_HOME})
endif()

if(NOT DEFINED HIP_CMAKE_PATH)
    if(NOT DEFINED ENV{HIP_CMAKE_PATH})
        # NOTE(yiakwy) : find_package(HIP) will first search for 
        #    cmake/Modules/FindAMDDeviceLibs.cmake
        # , then
        #   /opt/rocm/lib/cmake/AMDDeviceLibs/AMDDeviceLibsConfig.cmake
        # this will add hip::host, hip::device dependencies to be linked by any hip targets (ROCM >= 6.x).
        # Add hip-config.cmake to CMake module search path.
        # set(HIP_CMAKE_PATH "${ROCM_HOME}/share/rocm/cmake" "${ROCM_HOME}/share/rocmcmakebuildtools/cmake/" CACHE PATH "Path to which HIP has been installed")
        # NOTE(yiakwy) : adding ${ROCM_HOME}/lib/cmake/hip has conflicts with 3rdparty/mscclpp
        set(ROCSHMEM_LIB_DIR "/root/rocshmem")
        set(ROCSHMEM_CMAKE_PATH "${ROCSHMEM_LIB_DIR}/lib/cmake/rocshmem")

        set(HIP_CMAKE_PATH 
            # NOTE (yiakwy) : by default rocm install in local directory ~/rocshmem
            "${ROCSHMEM_CMAKE_PATH}"
            "${ROCM_HOME}/lib/cmake/AMDDeviceLibs" 
            "${ROCM_HOME}/lib/cmake/amd_comgr" 
            "${ROCM_HOME}/lib/cmake/hsa-runtime64" 
            "${ROCM_HOME}/lib/cmake/hipcub" 
            "${ROCM_HOME}/lib/cmake/rccl"
            "${ROCM_HOME}/lib/cmake/composable_kernel" CACHE PATH "Path to which HIP has been installed")
        message(WARNING "System variable HIP_CMAKE_PATH is nonexist, defaults to ${HIP_CMAKE_PATH}")

        set(CMAKE_PREFIX_PATH "${ROCM_HOME};${ROCM_HOME}/lib/cmake/hip;${ROCSHMEM_CMAKE_PATH};${CMAKE_PREFIX_PATH}")
    else()
        set(HIP_CMAKE_PATH $ENV{HIP_CMAKE_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()

    set(CMAKE_MODULE_PATH "${HIP_CMAKE_PATH}" ${CMAKE_MODULE_PATH})

endif()

add_definitions(-Wall)
find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
    execute_process(COMMAND bash -c "/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*'"
      OUTPUT_VARIABLE CMAKE_HIP_ARCHITECTURES OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "CMAKE_HIP_ARCHITECTURES : ${CMAKE_HIP_ARCHITECTURES}")

    enable_language(HIP)

    add_definitions(-DUSE_ROCM=1)

    if (NOT DEFINED CMAKE_CXX_COMPILER)
        find_program(CMAKE_CXX_COMPILER hipcc PATHS /opt/rocm)
    endif()

    # NOTE (yiakwy) : modern way to include ROCM tools
    find_package(ROCmCMakeBuildTools PATHS /opt/rocm)
    include(ROCMCreatePackage)
    include(ROCMInstallTargets)
    include(ROCMCheckTargetIds)

    # NOTE (yiakwy) : include rocSHMEM
    if(NOT TARGET roc::rocshmem)
        # find_package(rocshmem REQUIRED)
        find_package(rocshmem REQUIRED PATHS /root) #${ROCSHMEM_LIB_DIR})
    endif()

else()
    message(WARNING "Could not find HIP. Ensure that ROCM SDK is either installed in /opt/rocm or the variable HIP_CMAKE_PATH is set to point to the right location.")
endif()


find_package(CUDA QUIET)
if (CUDA_FOUND)
    message(STATUS "FOUND CUDA: " ${CUDA_TOOLKIT_ROOT_DIR})

    execute_process(COMMAND bash -c "/usr/bin/nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -o -m1 '[0-9.]*'"
      OUTPUT_VARIABLE CUDA_ARCHITECTURES OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(CUDA_SUPPORTED_ARCHS "9.0")

    find_package(CUDAToolkit REQUIRED)
    set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    find_package(NVSHMEM REQUIRED)

    set(CMAKE_CUDA_ARCHITECTURES 90a CACHE STRING "CUDA architecture to target")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    find_package(NVSHMEM REQUIRED)
else()
    message(WARNING "Could not find CUDA.")
endif()

if (NOT (HIP_FOUND) AND NOT (CUDA_FOUND))
    message(FATAL "ROCM/CUDA SDK must be supported")
endif()


# === Configuration options ===
option(WITH_TESTS "Build tests" OFF)
option(WITH_BENCHMARKS "Build benchmarks" OFF)

# === CMake configuration ===
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb -O0")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE )

# === Dependencies ===
include(FetchContent)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include(cmake/py_helper.cmake)
append_torch_cmake_prefix_path()
find_package(Torch REQUIRED)

find_package(MPI REQUIRED)

if(WITH_TESTS)
    enable_testing()
    find_package(PkgConfig REQUIRED)

    if (HIP_FOUND)
        pkg_check_modules(RCCL rccl)
    else()
        pkg_check_modules(NCCL nccl)
    endif()
endif()

# Create imported target for PyTorch
add_library(torch_imported INTERFACE)
add_library(torch::py_limited ALIAS torch_imported)
target_include_directories(torch_imported SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
# NOTE(lequn): We don't link against all ${TORCH_LIBRARIES} because we use py_limited_api.
# See: https://github.com/pytorch/pytorch/blob/9017becf1d895999a1c819c9d35b8139c090e7a9/torch/utils/cpp_extension.py#L1256-L1270
target_link_libraries(torch_imported INTERFACE c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart)

# === Compiler and linker flags ===
add_compile_options(-Wno-deprecated-declarations)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_definitions(Py_LIMITED_API=0x03090000)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# === Library targets ===
if (HIP_FOUND)
    include(cmake/rocm_helper.cmake)
endif()
include(cmake/py_helper.cmake)
add_subdirectory(all_to_all)
add_subdirectory(core)

# Main shared library
# NOTE (yiakwy) : TODO
# add_library(pplx_kernels SHARED
#     bindings/all_to_all_ops.cpp
#     bindings/bindings.cpp
# )

# target_link_libraries(pplx_kernels PUBLIC
#     all_to_all_internode_lib
#     all_to_all_intranode_lib
#     core_lib
#     torch::py_limited
#     Python::Module

#     # CUDA::cuda_driver
#     # CUDA::cudart

#     roc::rocshmem

#     # nvshmem::nvshmem_host
#     # nvshmem::nvshmem_device
# )

# set_target_properties(pplx_kernels PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/pplx_kernels
#     CUDA_SEPARABLE_COMPILATION ON
# )
