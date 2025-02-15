#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // For rand()

// Define the size of the input array
constexpr int ARRAY_SIZE = 512;
constexpr int BLOCK_DIM = ARRAY_SIZE / 2;

__global__ void SimpleSumReductionKernel(float * input, float * output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

// a kernel with less control divergence and improved execution resource utilization efficiency
__global__ void ConvergentSumReductionKernel(float *input, float *output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

// a kernel that uses shared memory to reduce global memory accesses
__global__ void SharedMemorySumReductionKernel(float *input, float *output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

void TestSimpleSumReductionKernel() {
    // Host input and output
    float h_input[ARRAY_SIZE];
    float h_output = 0.0f;

    // Initialize host input with random values
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }

    // Print the input values
    // std::cout << "Input Array: ";
    // for (int i = 0; i < ARRAY_SIZE; ++i) {
    //     std::cout << h_input[i] << " ";
    // }
    // std::cout << "\n";

    // Device input and output
    float *d_input, *d_output;
    cudaMalloc(&d_input, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = ARRAY_SIZE / 2; // Assuming ARRAY_SIZE is a power of 2
    SharedMemorySumReductionKernel<<<1, threadsPerBlock>>>(d_input, d_output);

    // Copy output data from device to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute the expected result on the host
    float expected_output = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        expected_output += h_input[i];
    }

    // Print the results
    std::cout << "Kernel Output: " << h_output << "\n";
    std::cout << "Expected Output: " << expected_output << "\n";

    // Check if the result is correct
    if (std::abs(h_output - expected_output) < 1e-4) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    TestSimpleSumReductionKernel();
    return 0;
}
