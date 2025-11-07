set(common_libs
    hiprtc
    amdhip64
    hsa-runtime64::hsa-runtime64
    hip::host
    hip::device
)

function(rocshmem_add_library lib_name srcs)
    cmake_parse_arguments(ARG "" "" "DEPS" ${ARGN})
    
    set_source_files_properties(${srcs} PROPERTIES LANGUAGE HIP)

    hip_add_library(${lib_name} STATIC ${srcs})
    
    set_target_properties(${lib_name} PROPERTIES HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
    target_compile_options(${lib_name} PRIVATE -fgpu-rdc)
    
    target_link_libraries(${lib_name}
        PRIVATE
            ${common_libs}
            MPI::MPI_CXX
            roc::rocshmem
            ${ARG_DEPS}
    )

    target_include_directories(${lib_name} PRIVATE ${ROCSHMEM_INCLUDE_DIR})
    # target_link_libraries(${lib_name} INTERFACE
    #     roc::rocshmem
    # )
endfunction()

function(rocshmem_add_executable exec_name srcs)
    cmake_parse_arguments(ARG "" "" "DEPS" ${ARGN})
    
    set(CMAKE_CXX_COMPIILER "/opt/rocm/bin/hipcc") 

    message(STATUS "exec_name : ${exec_name}")
    message(STATUS "srcs : ${srcs}")

    set_source_files_properties(${srcs} PROPERTIES LANGUAGE HIP)

    add_executable(${exec_name} ${srcs})
    
    # target_compile_options(${exec_name} PRIVATE -fgpu-rdc)
    
    target_link_libraries(${exec_name}
        PRIVATE
            ${common_libs}
            MPI::MPI_CXX
            roc::rocshmem
            ${ARG_DEPS}
    )
endfunction()