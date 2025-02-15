#include <iostream>
#include "utils.hpp"
constexpr int FILTER_RADIUS = 1;
constexpr int TILE_DIM =  32;
constexpr int FILTER_DIM = (2 * (FILTER_RADIUS) + 1);
constexpr int FILTER_SIZE =  FILTER_DIM * FILTER_DIM;

__constant__ float F_c[FILTER_SIZE];

__global__ void convolution_cached_tiled_2D_const_mem_kernel( float *N,
                                                              float *P, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    // loading input tile
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    // Calculating output elements
    // turing off the threads at the edges of the block
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
            for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
                // shared mem
                if (static_cast<int>(threadIdx.x) - FILTER_RADIUS + fCol >= 0 &&
                    static_cast<int>(threadIdx.x) - FILTER_RADIUS + fCol < TILE_DIM &&
                    static_cast<int>(threadIdx.y) - FILTER_RADIUS + fRow >= 0 &&
                    static_cast<int>(threadIdx.y) - FILTER_RADIUS + fRow < TILE_DIM ) {
                    Pvalue += F_c[fRow * FILTER_DIM + fCol] * N_s[threadIdx.y - FILTER_RADIUS + fRow][threadIdx.x - FILTER_RADIUS + fCol];

                }
                    // use cache of global mem
                else {
                    if (row - FILTER_RADIUS + fRow >= 0 &&
                        row - FILTER_RADIUS + fRow < height &&
                        col - FILTER_RADIUS + fCol >= 0 &&
                        col - FILTER_RADIUS + fCol < width) {
                        Pvalue += F_c[fRow * FILTER_DIM + fCol] * \
                            N[(row - FILTER_RADIUS + fRow) * width + col - FILTER_RADIUS + fCol];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
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

    float *N = new float[size];
    float *F_h = new float[FILTER_SIZE];

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            N[i * width + j] = 1;
        }
    }

    for (int i = 0; i < FILTER_SIZE; i++) {
        F_h[i] = static_cast<float>(i % 10);
    }

    // Allocate memory for output matrix P (on hots)
    float *P_h = new float[width * height];
    float *P_cpu = new float[width * height];

    float *d_N, *d_P;
    cudaMalloc(&d_N, size * sizeof(float));
    cudaMalloc(&d_P, size * sizeof(float));

    cudaMemcpy(d_N, N, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_c, F_h, (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));


    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    convolution_cached_tiled_2D_const_mem_kernel<<<gridSize, blockSize>>> (d_N, d_P, width, height);

    cudaMemcpy(P_h, d_P, size * sizeof(float), cudaMemcpyDeviceToHost);

    convolution_2D_host(N, F_h, P_cpu, width, height);

    // Compare results
    bool pass = true;
    for (int i = 0; i < size; ++i) {
        if (std::fabs(P_cpu[i] - P_h[i]) > 1e-5) {
            pass = false;
            std::cout << "Mismatch at index " << i << ": Host=" << P_cpu[i]
                      << ", Device=" << P_h[i] << std::endl;
        }
    }

    if (pass) {
        std::cout << "Test PASSED!" << std::endl;
    } else {
        std::cout << "Test FAILED!" << std::endl;
    }

    delete [] N;
    delete [] F_h;
    delete [] P_h;
    delete [] P_cpu;

    cudaFree(d_N);
    cudaFree(d_P);
    return 0;
}
