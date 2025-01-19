#include <iostream>
#include <cassert>
#include "utils.hpp"

#define TILE_WIDTH      32
#define COARSE_FACTOR   4

__global__ void matrixMulKernel(float *M, float *N, float *P, int width) {
    __shared__ float Mds [TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds [TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // Initializa Pvalue for all output elements
    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        Pvalue[c] = 0.0f;
    }

    // Loop over the M and N tiles required to compute P element
    for (int ph = 0; ph < ceil(width / (float)TILE_WIDTH); ++ph) {

        // Collaborative loading of M tile into shared memory
        if ( (row < width) && (ph * TILE_WIDTH + tx) < width)
            Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        else Mds[ty][tx] = 0.0f;

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c * TILE_WIDTH;

            // Collaborative loading of N tile into shared memory
            if ((ph * TILE_WIDTH + ty) < width && col < width)
                Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
            else Nds[ty][tx] = 0.0f;
            __syncthreads();

            for(int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        if (row < width && col < width)
            P[row * width + col] = Pvalue[c];
    }
}

// CPU Reference Implementation for Matrix Multiplication
void matrixMulCPU(float *M, float *N, float *P, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float value = 0.0f;
            for (int k = 0; k < width; ++k) {
                value += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = value;
        }
    }
}

int main() {
    const int width = 2000; // 示例宽度
    const int size = width * width;

    UnifiedPtr<float> M(size, true);
    UnifiedPtr<float> N(size, true);
    UnifiedPtr<float> P(size, true);
    UnifiedPtr<float> P_ref(size);

    for (int i = 0; i < size; ++i) {
        M[i] = static_cast<float>(i % 10);
        N[i] = static_cast<float>((i + 1) % 10);
    }

    float* d_M = M.get();
    float* d_N = N.get();
    float* d_P = P.get();

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (width + blockSize.y - 1) / blockSize.y);

    matrixMulKernel<<<gridSize, blockSize>>>(d_M, d_N, d_P, width);

    cudaDeviceSynchronize();

    matrixMulCPU(M.get(), N.get(), P_ref.get(), width);

    // 验证结果
    for (int i = 0; i < size; ++i) {
        assert(fabs(P[i] - P_ref[i]) < 1e-3); // 允许小的浮动误差
    }

    std::cout << "Test passed successfully!" << std::endl;

    // 内存释放由 UnifiedPtr 自动处理，无需手动释放
    return 0;
}