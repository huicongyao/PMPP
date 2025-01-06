#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "utils.hpp"

__global__ void matrixMulKernel(float *M, float *N,
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

#define TILE_WIDTH 16
__global__ void tiled_matrixMulKernel(float *M, float *N, float *P, int Width) {
    __shared__ float Mds [TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds [TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < ceil(Width / (float)TILE_WIDTH); ++ph) {
        // Collaborative loading of M and N tiles into shared memory
        if ( (Row < Width) && (ph * TILE_WIDTH + tx) < Width)
            Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        else Mds[ty][tx] = 0.0f;
        if ((ph * TILE_WIDTH + ty) < Width && Col < Width)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        else Nds[ty][tx] = 0.0f;
        // ensures that all threads have finished loading the tiles of M and N into `Mds` and `Nds`
        // before any of them can move forward
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if (Row < Width && Col < Width)
        P[Row*Width + Col] = Pvalue;
}

int main() {
    constexpr int Width = 400;
    constexpr int Matrix_Size = Width * Width;
//    RandomNumberGenerator &rng = RandomNumberGenerator::getInstance(0, 256);

    float M_h[Matrix_Size];
    float N_h[Matrix_Size];
    float K_h[Matrix_Size];
    float K_h_[Matrix_Size];

    for (int i = 0; i < Matrix_Size; i++) {
        M_h[i] = (rand() % 256) / 256.0;
        N_h[i] = (rand() % 256) / 256.0;
    }

    float *M_d, *N_d, *K_d, *K_d_;
    const int size =  Matrix_Size * sizeof(float);
    cudaMalloc(&M_d, size);
    cudaMalloc(&N_d, size);
    cudaMalloc(&K_d, size);
    cudaMalloc(&K_d_, size);

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((Width + blockSize.x - 1) / blockSize.x,
                  (Width + blockSize.y - 1) / blockSize.y);

    tiled_matrixMulKernel<<<gridSize, blockSize>>> (M_d, N_d, K_d, Width);

    cudaMemcpy(K_h, K_d, size, cudaMemcpyDeviceToHost);

    matrixMulKernel<<<gridSize, blockSize>>> (M_d, N_d, K_d_, Width);
    cudaMemcpy(K_h_, K_d_, size, cudaMemcpyDeviceToHost);


    float err = 0.0f;
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            err += abs(K_h[i * Width + j] - K_h_[i * Width + j]);
        }
    }
    std::cout << "Error: " << err << std::endl;

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(K_d);

    return 0;
}