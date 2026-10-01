cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(batch_matmul_max_sum_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

if(DEFINED NPU_ARCH)
    set(SOC_ARCH ${NPU_ARCH})
else()
    set(SOC_ARCH "dav-2201")
endif()

add_executable(batch_matmul_max_sum_custom driver.asc)

target_link_libraries(batch_matmul_max_sum_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(batch_matmul_max_sum_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(batch_matmul_max_sum_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
)

# --- 本地构建补充：ASC 编译器需要 C++ 标准库路径（平台自带构建里没暴露）---
target_compile_options(batch_matmul_max_sum_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:-isystem/usr/include/c++/11>
    $<$<COMPILE_LANGUAGE:ASC>:-isystem/usr/include/x86_64-linux-gnu/c++/11>
    $<$<COMPILE_LANGUAGE:ASC>:-isystem/usr/include/c++/11/backward>
    $<$<COMPILE_LANGUAGE:ASC>:-isystem/usr/lib/gcc/x86_64-linux-gnu/11/include>
    $<$<COMPILE_LANGUAGE:ASC>:-isystem/usr/local/include>
    $<$<COMPILE_LANGUAGE:ASC>:-isystem/usr/include/x86_64-linux-gnu>
    $<$<COMPILE_LANGUAGE:ASC>:-isystem/usr/include>
)
