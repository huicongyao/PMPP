#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // For rand()

constexpr int ARRAY_SIZE = 2000;  // Size of the array to test
constexpr int SECTION_SIZE = 8;

void sequential_scan(float *x, float *y, unsigned int N) {
    y[0] = x[0];
    for (unsigned int i = 1; i < N; ++i) {
        y[i] = y[i - 1] + x[i];
    }
}

__global__ void Brent_Kung_scan_kernel_first_step(float *X, float *Y, float *d_Block_sum, const int array_size) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int N = (2 * blockIdx.x * blockDim.x + SECTION_SIZE);
    if (N > array_size) N = array_size;
    // printf("%d %d %d\n",N, i, ARRAY_SIZE);
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
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1 && d_Block_sum != nullptr) d_Block_sum[blockIdx.x] = XY[SECTION_SIZE - 1];
}

__global__ void Koggle_Stone_scan_kernel_second_step(float * input, const int array_size, const int section_size) {
    extern __shared__ float XY[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < array_size) {
        XY[threadIdx.x] = input[i];
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
    if (i < array_size) {
        input[i] = XY[threadIdx.x];
    }
    // printf("%d %f\n", i , input[i]);
}

__global__ void hierarchical_scan_final(float *d_output, float *d_block_sum, const int array_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x >= 1 && i < array_size)
        d_output[i] += d_block_sum[blockIdx.x - 1];
}

// run three kernel to perform hierarchical scan
void hierarchical_scan_for_arbitrary_inputs(float *h_input, float *h_output) {
    // Device input and output arrays
    int numBlocks = (ARRAY_SIZE + SECTION_SIZE - 1) / SECTION_SIZE; // Ceil division
    float *d_input, *d_output, *d_Block_sum;
    cudaMalloc(&d_Block_sum, numBlocks * sizeof(float));
    cudaMalloc(&d_input, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_output, ARRAY_SIZE * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    Brent_Kung_scan_kernel_first_step<<<numBlocks, SECTION_SIZE / 2>>>(d_input, d_output, d_Block_sum, ARRAY_SIZE);
    cudaDeviceSynchronize();
    Koggle_Stone_scan_kernel_second_step<<<1, numBlocks, numBlocks>>>(d_Block_sum, numBlocks, numBlocks);
    cudaDeviceSynchronize();
    hierarchical_scan_final<<<numBlocks, SECTION_SIZE>>> (d_output, d_Block_sum, ARRAY_SIZE);
    // Copy output data back to host
    cudaMemcpy(h_output, d_output, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_Block_sum);
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

    hierarchical_scan_for_arbitrary_inputs(h_input, h_output);

    // Verify correctness
    bool success = true;
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        if (std::abs(h_output[i] - h_reference[i]) > 1e-3) {
            success = false;
            std::cout << "idx: " << i << ", " << h_output[i] << ", " << h_reference[i] << "\n";
            // break;
        }
    }

    if (success) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }    
}

int main() {
    TestKoggleStoneScanKernel();
    return 0;
}
