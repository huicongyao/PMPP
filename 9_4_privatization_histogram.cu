#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
// Number of bins (26 / 4 = 6.5, rounded to 7 bins)
constexpr int NUM_BINS = 7;

void histogram_sequential(char *data, unsigned int length, 
                          unsigned int *histo) {
    for (unsigned int i = 0; i < length; ++i) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            histo[alphabet_position / 4] ++;
        }
    }
}

__global__ void histo_private_kernel(char *data, unsigned int length,
                             unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[ blockIdx.x * NUM_BINS + alphabet_position / 4]), 1);
        }
    }
    if (blockIdx.x > 0) {
        __syncthreads();
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
            unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
            if (binValue > 0) {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}


// a privatized text histogram kernel using shared memory
__global__ void histo_private_kernel_shared_mem(char* data, unsigned int length,
                                                unsigned int* histo){
    // Initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();
    // Histogram
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position / 4]), 1);
        }
    }
    __syncthreads();
    // Commit to global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&(histo[bin]), binValue);
        }
    }
}


// Helper function to generate random lowercase letters
void generate_random_data(char *data, unsigned int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis('a', 'z');

    for (unsigned int i = 0; i < length; ++i) {
        data[i] = static_cast<char>(dis(gen));
    }
}

// Helper function to print the histogram
void print_histogram(unsigned int *histo) {
    for (int i = 0; i < NUM_BINS; ++i) {
        std::cout << "Bin " << i << ": " << histo[i] << std::endl;
    }
}

int main() {
    const unsigned int data_length = 1 << 22; // 1M elements
    char *data;
    unsigned int *histo_seq, *histo_gpu;
    // Parallel computation using CUDA
    const unsigned int threads_per_block = 256;
    const unsigned int blocks = (data_length + threads_per_block - 1) / threads_per_block;

    // Unified memory allocation using cudaMallocManaged
    cudaMallocManaged(&data, data_length * sizeof(char));
    cudaMallocManaged(&histo_seq, NUM_BINS * sizeof(unsigned int));
    cudaMallocManaged(&histo_gpu, NUM_BINS * sizeof(unsigned int) * blocks);

    // Initialize data and histograms
    generate_random_data(data, data_length);
    std::fill(histo_seq, histo_seq + NUM_BINS, 0);
    std::fill(histo_gpu, histo_gpu + NUM_BINS * blocks, 0);

    // Sequential computation
    auto start_seq = std::chrono::high_resolution_clock::now();
    histogram_sequential(data, data_length, histo_seq);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;
    std::cout << "Sequential execution time: " << duration_seq.count() << " seconds\n";

    auto start_gpu = std::chrono::high_resolution_clock::now();
    histo_private_kernel<<<blocks, threads_per_block>>>(data, data_length, histo_gpu);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu = end_gpu - start_gpu;
    std::cout << "Parallel execution time: " << duration_gpu.count() << " seconds\n";

    // Print and compare results
    std::cout << "Sequential Histogram:\n";
    print_histogram(histo_seq);
    std::cout << "GPU Histogram:\n";
    print_histogram(histo_gpu);

    bool correct = true;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (histo_seq[i] != histo_gpu[i]) {
            std::cout << "Mismatch at bin " << i << ": " << histo_seq[i] << " != " << histo_gpu[i] << std::endl;
            correct = false;
        }
    }

    if (correct) {
        std::cout << "Results match!\n";
    } else {
        std::cout << "Results do not match.\n";
    }

    // Free unified memory
    cudaFree(data);
    cudaFree(histo_seq);
    cudaFree(histo_gpu);
    return 0;
}