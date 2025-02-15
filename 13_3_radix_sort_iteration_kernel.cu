#include <iostream>
#include <cstdlib> // For rand()
#include <cuda_runtime.h>
#include <algorithm>
#include <bitset>
#include "utils.hpp"

constexpr int SECTION_SIZE = 128;
constexpr unsigned int array_size = 2000;

int* block_sync_flag;
// int* block_finish_Counter;

__global__ void radix_sort_iter(unsigned int* input, unsigned int* output,
            unsigned int* bits, unsigned int N, unsigned int iter, 
            int* block_sync_flag) {
    // if (threadIdx.x == 0)
    // printf("%d %d \n", *block_Start_Counter, *block_finish_Counter);
    __shared__ unsigned int bid_s;
    if (threadIdx.x == 0) {
        bid_s = atomicAdd(block_sync_flag, 1);
    }
    __syncthreads();
    unsigned int bid = bid_s;
    unsigned int i = bid * blockDim.x + threadIdx.x;
    unsigned int key, bit;
    if (i < N) {
        key = input[i];
        bit = (key >> iter) & 1;
        bits[i + 1] = bit;
    }
    __syncthreads();
    // perform complex hierarchical exclusive scan
    {
        __shared__ float XY[SECTION_SIZE];
        // unsigned int i = bid * blockDim.x + threadIdx.x;
        if (i < N) {
            XY[threadIdx.x] = bits[i + 1];
        } else {
            XY[threadIdx.x] = 0.0f;
        }
        if (threadIdx.x == 0 && bid != 0) {
            while(atomicAdd(block_sync_flag + 1, 0) != bid) {} // Wait for previous flag
            // Read previous partial sum
            XY[threadIdx.x] += bits[bid * blockDim.x];
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
            bits[i + 1] = XY[threadIdx.x]; // exclusive scan
        }
        // set flag
        if (threadIdx.x == 0)
            atomicAdd(block_sync_flag + 1, 1);
    }
    // __syncthreads();
    while(atomicAdd(block_sync_flag + 1, 0) != gridDim.x) {} // wait all blocks finish hierarchical scan
    // if (i == 0) {
    //     printf("itr: %d \n ", iter);
    //     for (int i = 0; i <= N; i++) {
    //         printf("%d ", bits[i]);
    //     }
    //     printf("\n");
    // }
    unsigned int numOnesTotal = bits[N];
    if (i < N) {
        unsigned int numOnesBefore = bits[i];
        unsigned int dst = (bit == 0) ? (i - numOnesBefore) 
                                      : (N - numOnesTotal + numOnesBefore);
        // if (key == 6 && iter == 1) {
        //     printf("i: %d, dst: %d, numOnesBefore: %d, numOnesTotal: %d\n",i, dst, numOnesBefore, numOnesTotal);
        // }
        output[dst] = key;
    }
}

int main() {
    
    constexpr int numBlocks = (array_size + SECTION_SIZE - 1) / SECTION_SIZE; // Ceil division
    auto input = UnifiedPtr<unsigned int>(array_size, true);
    auto output = UnifiedPtr<unsigned int>(array_size, true);
    auto bits = UnifiedPtr<unsigned int>(array_size + 1, true);
    cudaMallocManaged(&block_sync_flag, 2 * sizeof(int));
    
    for (int i = 0; i < array_size; i++) {
        input[i] = rand() % 32;
        output[i] = UINT32_MAX ;
        bits[i] = 0;
    }

    for (int itr = 0; itr < 8; itr++) {
        *block_sync_flag = 0;
        *(block_sync_flag + 1) = 0;
        // printf("itr: %d %d %d\n", itr, *block_Start_Counter, *block_finish_Counter);
        if (itr == 0) radix_sort_iter<<<numBlocks, SECTION_SIZE >>> (input.get(), output.get(), bits.get(), array_size, itr, block_sync_flag);
        else radix_sort_iter<<<numBlocks, SECTION_SIZE >>> (output.get(), output.get(), bits.get(), array_size, itr, block_sync_flag);
        cudaDeviceSynchronize();
        // for (int i = 0; i < array_size; i++) {
        //     std::cout << "itr: " << itr << "\t" << std::bitset<10>(output[i]) << "\n";
        // }
    }

    std::sort(input.get(), input.get() + array_size);

    // Verify correctness
    bool success = true;
    for (int i = 0; i < array_size; ++i) {
        if (input[i] != output[i]) {
            success = false;
            std::cout << "idx: " << i << "\t\t, " << std::bitset<32>(input[i]) << ", " << std::bitset<32>(output[i]) << "\n";
            // break;
        }
    }

    if (success) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }  

}