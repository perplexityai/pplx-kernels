# Core library with common kernels.
set(core_srcs
    kernels.cu
    distributed.cpp
)

if (HIP_FOUND)
    set_source_files_properties(${core_srcs} PROPERTIES LANGUAGE HIP)

    hip_add_library(core_lib STATIC
        ${core_srcs}
    )

    set_target_properties(core_lib PROPERTIES HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
    target_link_libraries(core_lib PUBLIC
        hiprtc
        amdhip64
        hsa-runtime64::hsa-runtime64
        hip::device
        hip::host
        MPI::MPI_CXX
        roc::rocshmem
        -fgpu-rdc
    )
    target_include_directories(core_lib PRIVATE ${ROCSHMEM_INCLUDE_DIR})
    target_link_libraries(core_lib INTERFACE
        roc::rocshmem
    )
else()
    add_library(core_lib STATIC
        ${core_srcs}
    )
    target_link_libraries(core_lib PUBLIC
        MPI::MPI_CXX
        CUDA::cudart
    )
    target_include_directories(core_lib PRIVATE ${NVSHMEM_INCLUDE_DIR})
    target_link_libraries(core_lib INTERFACE
        nvshmem::nvshmem_host
    )
    
endif()

message(STATUS "ROCSHMEM incr : ${ROCSHMEM_INCLUDE_DIR}")