#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // For rand()

constexpr int ARRAY_SIZE = 512;  // Size of the array to test
constexpr int BLOCK_SIZE = 256;  // Must match SECTION_SIZE
constexpr int SECTION_SIZE = 512;

void sequential_scan(float *x, float *y, unsigned int N) {
    y[0] = x[0];
    for (unsigned int i = 1; i < N; ++i) {
        y[i] = y[i - 1] + x[i];
    }
}


__global__ void Koggle_Stone_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            XY[threadIdx.x] = temp;
        }
    }
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
}

__global__ void Brent_Kung_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) XY[threadIdx.x] = X[i];
    if (i + blockDim.x < N) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE) {
            XY[index] += XY[index - stride];
        }
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i < N) Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

void TestKoggleStoneScanKernel() {
    // Host input and output arrays
    float h_input[ARRAY_SIZE];
    float h_output[ARRAY_SIZE];
    float h_reference[ARRAY_SIZE];

    // Initialize host input with random values
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }

    // Compute the expected output (sequential scan) on the host
    sequential_scan(h_input, h_reference, ARRAY_SIZE);

    // Device input and output arrays
    float *d_input, *d_output;
    cudaMalloc(&d_input, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_output, ARRAY_SIZE * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int numBlocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE; // Ceil division
    Brent_Kung_scan_kernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, ARRAY_SIZE);

    // Copy output data back to host
    cudaMemcpy(h_output, d_output, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify correctness
    bool success = true;
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        if (std::abs(h_output[i] - h_reference[i]) > 1e-3) {
            success = false;
            std::cout << "idx: " << i << ", " << h_output[i] << ", " << h_reference[i] << "\n";
            break;
        }
    }

    if (success) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    TestKoggleStoneScanKernel();
    return 0;
}
