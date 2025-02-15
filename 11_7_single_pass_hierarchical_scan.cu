#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // For rand()

constexpr int ARRAY_SIZE = 870;  // Size of the array to test
constexpr int SECTION_SIZE = 102;

void sequential_scan(float *x, float *y, unsigned int N) {
    y[0] = x[0];
    for (unsigned int i = 1; i < N; ++i) {
        y[i] = y[i - 1] + x[i];
    }
}

__device__ int block_Start_Counter = 0;
__device__ int block_finish_Counter = 0;

__global__ void Koggle_Stone_scan_kernel(float *X, float *Y, int *flags, unsigned int N) {
    __shared__ unsigned int bid_s;
    if (threadIdx.x == 0) {
        bid_s = atomicAdd(&block_Start_Counter, 1);
    }
    __syncthreads();
    unsigned int bid = bid_s;
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = bid * blockDim.x + threadIdx.x;
    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x == 0 && bid != 0) {
        // Wait for previous flag
        while(atomicAdd(&block_finish_Counter, 0) != bid) {}

        // Read previous partial sum
        XY[threadIdx.x] += Y[bid * blockDim.x - 1];
        // memory fence
        // __threadfence();
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
    // set flag
    // atomicAdd(&flags[bid + 1], 1);
    if (threadIdx.x == 0)
        atomicAdd(&block_finish_Counter, 1);
}


void TestHierarchicalScanKernel() {
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
    int *d_flags;
    cudaMalloc(&d_input, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_output, ARRAY_SIZE * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int numBlocks = (ARRAY_SIZE + SECTION_SIZE - 1) / SECTION_SIZE; // Ceil division
    cudaMalloc(&d_flags, numBlocks * sizeof(int));
    cudaMemset(&d_flags, 0, numBlocks * sizeof(int));
    Koggle_Stone_scan_kernel<<<numBlocks, SECTION_SIZE>>>(d_input, d_output, d_flags, ARRAY_SIZE);

    // Copy output data back to host
    cudaMemcpy(h_output, d_output, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_flags);
}

int main() {
    TestHierarchicalScanKernel();
    return 0;
}
