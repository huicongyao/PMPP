#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // For rand()

// Define the size of the input array
constexpr int ARRAY_SIZE = 5120;
constexpr int BLOCK_DIM = 64;
constexpr int GRID_DIM = (ARRAY_SIZE + 2 * BLOCK_DIM - 1) / (2 * BLOCK_DIM);
__global__ void SegmentedSumReductionKernel(float * input, float *output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i];
    if (i + BLOCK_DIM < ARRAY_SIZE)
        input_s[t]  += input[i + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

void TestSimpleSumReductionKernel() {
    // Host input and output
    float h_input[ARRAY_SIZE];
    float h_output = 0.0f;

    // Initialize host input with random values
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        // h_input[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
        h_input[i] = static_cast<float>(i);
    }

    // Device input and output
    float *d_input, *d_output;
    cudaMalloc(&d_input, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    SegmentedSumReductionKernel<<<GRID_DIM, BLOCK_DIM>>>(d_input, d_output);

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
