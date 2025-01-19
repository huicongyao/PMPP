#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include "utils.hpp"
#include "random.hpp"

__global__ void MatrixMulKernel(float *M, float *N,
                                float *P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

int main() {
    const int Width = 2;
    const int Matrix_Size = Width * Width;

    // 使用UnifiedPtr管理内存
    UnifiedPtr<float> M(Matrix_Size, true);  // 使用CUDA内存
    UnifiedPtr<float> N(Matrix_Size, true);  // 使用CUDA内存
    UnifiedPtr<float> K(Matrix_Size, true);  // 使用CUDA内存

    // 初始化矩阵M和N
    for (int i = 0; i < Matrix_Size; i++) {
        M[i] = rand() % 10;
        N[i] = rand() % 10;
    }

    // 打印矩阵M
    std::cout << "Matrix M:" << std::endl;
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            std::cout << M[i * Width + j] << " ";
        }
        std::cout << std::endl;
    }

    // 打印矩阵N
    std::cout << "Matrix N:" << std::endl;
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            std::cout << N[i * Width + j] << " ";
        }
        std::cout << std::endl;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((Width + blockSize.x - 1) / blockSize.x,
                  (Width + blockSize.y - 1) / blockSize.y);

    // 调用CUDA内核函数
    MatrixMulKernel<<<gridSize, blockSize>>>(M.get(), N.get(), K.get(), Width);

    // 等待CUDA内核执行完成
    cudaDeviceSynchronize();
    // 检查CUDA内核执行是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 打印结果矩阵K
    std::cout << "Matrix K (result):" << std::endl;
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            std::cout << K[i * Width + j] << " ";
        }
        std::cout << std::endl;
    }

    // UnifiedPtr会自动释放内存，无需手动调用cudaFree

    return 0;
}