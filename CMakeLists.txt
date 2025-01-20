cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)

project(ppmp LANGUAGES CXX CUDA)

enable_language(CUDA)


add_executable(
        vec_add
        2_3_vec_add.cu
        utils.hpp
)

add_executable(
        color_to_grey
        3_2_color_to_grey.cu
        utils.hpp
)

add_executable(
        image_blur
        3_3_image_blur.cu
        random.hpp
        utils.hpp
)

add_executable(
        matrix_mul
        3_4_matrix_mul.cu
        utils.hpp
)

add_executable(
        query
        4_8_query.cu
        utils.hpp
)

add_executable(
        tiled_gemm
        5_4_tiled_matrix_multiplication_kernel.cu
        utils.hpp
)


add_executable(
        gemm_thread_coarsening
        6_3_matrix_multiplication_thread_coarsening.cu
        utils.hpp
)

add_executable(
        2d_conv_simple
        7_1_2d_conv_simple.cu
        utils.hpp
)