#include <iostream>
#include <vector>
#include "utils.hpp"
#define FILTER_RADIUS 1
__constant__ float F[(2 * FILTER_RADIUS + 1)*(2 * FILTER_RADIUS + 1)];

__global__ void convolution_2D_const_mem_kernel(float *N, float *P,
                                                int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
//     printf("%f\n", Pvalue);
    if (outRow < height && outCol < width)
        P[outRow * width + outCol] = Pvalue;
}

// Host-side function to perform 2D convolution
void convolution_2D_host(float* N, float* F, float * P,
                         int r, int width, int height) {
    int filterSize = 2 * r + 1;
    for (int outRow = 0; outRow < height; ++outRow) {
        for (int outCol = 0; outCol < width; ++outCol) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < filterSize; ++fRow) {
                for (int fCol = 0; fCol < filterSize; ++fCol) {
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;
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
    // Input matrix dimensions
    const int width = 5;
    const int height = 5;
    const int r = 1; // Filter radius

    // Input matrix N and filter F (on host)
    UnifiedPtr<float> N ({
                                  1, 2, 3, 4, 5,
                                  6, 7, 8, 9, 10,
                                  11, 12, 13, 14, 15,
                                  16, 17, 18, 19, 20,
                                  21, 22, 23, 24, 25
                          }, true);

    UnifiedPtr<float> F_h({
        1, 0, -1,
                1, 0, -1,
                1, 0, -1
    }, false);

    // Allocate memory for output matrix P (on host)
    UnifiedPtr<float> P(width * height, 0, true);
    std::vector<float> P_host(width * height, 0);

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    cudaMemcpyToSymbol(F, F_h.get(), sizeof(float) * (2 * r + 1) * (2 * r + 1));
    convolution_2D_const_mem_kernel<<<gridDim, blockDim>>>(N.get(),  P.get(), r, width, height);
    cudaDeviceSynchronize();
    // Copy result back to host

    // Perform convolution on host for verification
    convolution_2D_host(N.get(), F_h.get(), P_host.data(), r, width, height);

    // Compare results
    bool pass = true;
    for (int i = 0; i < P_host.size(); ++i) {
        if (std::fabs(P_host[i] - P[i]) > 1e-5) {
            pass = false;
            std::cout << "Mismatch at index " << i << ": Host=" << P_host[i]
                      << ", Device=" << P[i] << std::endl;
        }
    }

    if (pass) {
        std::cout << "Test PASSED!" << std::endl;
    } else {
        std::cout << "Test FAILED!" << std::endl;
    }

    return 0;
}