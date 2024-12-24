#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
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
    RandomNumberGenerator &rng = RandomNumberGenerator::getInstance(0, 9);

    float M_h[Matrix_Size];
    float N_h[Matrix_Size];
    float K_h[Matrix_Size];

    for (int i = 0; i < Matrix_Size; i++) {
        M_h[i] = rng.getRandomInt();
        N_h[i] = rng.getRandomInt();
    }

    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            std::cout << M_h[i * Width + j] << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            std::cout << N_h[i * Width + j] << " ";
        }
        std::cout << std::endl;
    }

    float *M_d, *N_d, *K_d;
    const int size =  Matrix_Size * sizeof(float);
    cudaMalloc(&M_d, size);
    cudaMalloc(&N_d, size);
    cudaMalloc(&K_d, size);

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((Width + blockSize.x - 1) / blockSize.x,
                  (Width + blockSize.y - 1) / blockSize.y);

    MatrixMulKernel<<<gridSize, blockSize>>> (M_d, N_d, K_d, Width);

    cudaMemcpy(K_h, K_d, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            std::cout << K_h[i * Width + j] << " ";
        }
        std::cout << std::endl;
    }
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(K_d);

    return 0;
}