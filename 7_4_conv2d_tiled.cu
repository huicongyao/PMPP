#include <iostream>
#include "utils.hpp"

constexpr int FILTER_RADIUS = 1;
constexpr int IN_TILE_DIM =  32;
constexpr int FILTER_SIZE =  (2 * (FILTER_RADIUS) + 1) * (2 * (FILTER_RADIUS) + 1);
constexpr int OUT_TILE_DIM =  ((IN_TILE_DIM) - 2 * (FILTER_RADIUS));

__constant__ float F_c[FILTER_SIZE];
__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P,
                                                      int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    // loading input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    // turing off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0
            && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Pvalue += F_c[fRow * (2 * FILTER_RADIUS + 1) + fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

// Host-side function to perform 2D convolution
void convolution_2D_host(const float * N, const float* F, float * P,
                         int width, int height) {
    int filterSize = 2 * FILTER_RADIUS + 1;
    for (int outRow = 0; outRow < height; ++outRow) {
        for (int outCol = 0; outCol < width; ++outCol) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < filterSize; ++fRow) {
                for (int fCol = 0; fCol < filterSize; ++fCol) {
                    int inRow = outRow - FILTER_RADIUS + fRow;
                    int inCol = outCol - FILTER_RADIUS + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Pvalue += F[fRow * filterSize + fCol] * N[inRow * width + inCol];
                    }
                }
            }

            P[outRow * width + outCol] = Pvalue;
        }
    }
}

int main() {
    constexpr int width = 200;
    constexpr int height = 200;
    constexpr int size = width * height;

//    float *N = new float[size];
    UnifiedPtr<float> N(size, true);
    float *F_h = new float[FILTER_SIZE];

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            N[i * width + j] = static_cast<float>(i * j % 10);
        }
    }

    for (int i = 0; i < (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1); i++) {
        F_h[i] = static_cast<float>(i % 10);
    }

    // Allocate memory for output matrix P (on hots)
    UnifiedPtr<float> P(size, true);
    float *P_cpu = new float[width * height];

    cudaMemcpyToSymbol(F_c, F_h, (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));


    dim3 blockSize(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    convolution_tiled_2D_const_mem_kernel<<<gridSize, blockSize>>> (N.get(), P.get(), width, height);

    cudaDeviceSynchronize();

    convolution_2D_host(N.get(), F_h, P_cpu, width, height);

    // Compare results
    bool pass = true;
    for (int i = 0; i < size; ++i) {
        if (std::fabs(P_cpu[i] - P[i]) > 1e-5) {
            pass = false;
            std::cout << "Mismatch at index " << i << ": Host=" << P_cpu[i]
                      << ", Device=" << P[i] << std::endl;
        }
    }

    if (pass) {
        std::cout << "Test PASSED!" << std::endl;
    } else {
        std::cout << "Test FAILED!" << std::endl;
    }

    delete [] F_h;
    delete [] P_cpu;

    return 0;
}
