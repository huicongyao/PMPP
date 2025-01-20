#include <iostream>
#include <vector>
#include "utils.hpp"

__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P,
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
    if (outRow < height && outCol < width)
        P[outRow * width + outCol] = Pvalue;
}

// Host-side function to perform 2D convolution
void convolution_2D_host(const float * N, const float* F, float* P,
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
    UnifiedPtr<float> N({
                                1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10,
                                11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20,
                                21, 22, 23, 24, 25
                        }, true);

    UnifiedPtr<float> F ({
                                 1, 0, -1,
                                 1, 0, -1,
                                 1, 0, -1
                         }, true);
    UnifiedPtr<float> P_device_output(width * height, true);

    // No need for host memory or cudaMemcpy now, as everything is unified.

    // Get the raw pointers for CUDA kernel
    float* d_N = N.get();
    float* d_F = F.get();
    float* d_P = P_device_output.get();

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA convolution kernel
    convolution_2D_basic_kernel<<<gridDim, blockDim>>>(d_N, d_F, d_P, r, width, height);

    // Synchronize the device to ensure the kernel finishes
    cudaDeviceSynchronize();

    // Perform convolution on host for verification
    UnifiedPtr<float> P_host(width * height, false);  // Host output matrix
    convolution_2D_host(N.get(), F.get(), P_host.get(), r, width, height);

    // Compare the results
    bool pass = true;
    for (int i = 0; i < P_host.size(); ++i) {
        if (std::fabs(P_host[i] - P_device_output[i]) > 1e-5) {
            pass = false;
            std::cout << "Mismatch at index " << i << ": Host=" << P_host[i]
                      << ", Device=" << P_device_output[i] << std::endl;
        }
    }

    if (pass) {
        std::cout << "Test PASSED!" << std::endl;
    } else {
        std::cout << "Test FAILED!" << std::endl;
    }

    // No need to manually free memory as UnifiedPtr will automatically clean up

    return 0;
}
