#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <climits> // For UINT_MAX
#include "utils.hpp"

struct CSRGraph {
    unsigned int *srcPtrs;
    unsigned int *dst;
};

__global__ void bfs_kernel_frontiers(CSRGraph csrGraph, unsigned int* level, 
                unsigned int* prevFrontier, unsigned int* currFrontier,
                unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
                unsigned int currLevel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX) {
                unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                currFrontier[currFrontierIdx] = neighbor;
            }
        }
    }
}

constexpr unsigned int LOCAL_FRONTIER_CAPACITY = 10;
__global__ void bfs_kernel_frontiers_privatization(CSRGraph csrGraph, unsigned int* level,
                unsigned int* prevFrontier, unsigned int* currFrontier,
                unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
                unsigned int currLevel) {
    // Initialize privatized frontier
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    // Perform BFS
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX) {
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if (currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
                    unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    // Allocate in global frontier
    __shared__ unsigned int currFrontierStartIdx;
    if(threadIdx.x == 0) {
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();

    // Commit to global frontiers
    for (unsigned int currFrontierIdx_s = threadIdx.x; 
            currFrontierIdx_s < numCurrFrontier_s; currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

int main() {
    constexpr int V = 9;
    // constexpr int E = 15;

    UnifiedPtr<unsigned int> srcPtrs({0,2,4,7,9,11,12,13,15,15}, true);
    UnifiedPtr<unsigned int> dst({1,2,3,4,5,6,7,4,8,5,8,6,8,0,6}, true);

    UnifiedPtr<unsigned int> level(V, true, UINT_MAX);
    level[0] = 0;
    UnifiedPtr<unsigned int> prevFrontier(V, true, UINT_MAX);
    prevFrontier[0] = 0;
    UnifiedPtr<unsigned int> currFrontier(V, true, UINT_MAX);
    UnifiedPtr<unsigned int> numCurrFrontier(1, true, 0);
    unsigned int currLevel = 1;
    unsigned int numPrevFrontier = 1;
    // unsigned int block_dim = 32;
    while (numPrevFrontier > 0) { 
        unsigned int block_dim = 32;
        unsigned int grid_dim = (block_dim + numPrevFrontier - 1) / block_dim;
        // std::cout << "curr level: " << currLevel << std::endl;
        // bfs_kernel_frontiers<<<grid_dim, block_dim>>>(
        //     {srcPtrs.get(), dst.get()}, 
        //     level.get(),
        //     prevFrontier.get(), currFrontier.get(),
        //     numPrevFrontier, numCurrFrontier.get(),
        //     currLevel);
        bfs_kernel_frontiers_privatization<<<grid_dim, block_dim>>>(
            {srcPtrs.get(), dst.get()}, 
            level.get(),
            prevFrontier.get(), currFrontier.get(),
            numPrevFrontier, numCurrFrontier.get(),
            currLevel);
        cudaDeviceSynchronize();
        // printf("%d \n", numCurrFrontier[0]);
        currLevel += 1;
        for (size_t i = 0; i < numCurrFrontier[0]; i++) {
            prevFrontier[i] = currFrontier[i];
        }
        numPrevFrontier = numCurrFrontier[0];
        numCurrFrontier[0] = 0;
    }

    // Output the levels
    std::cout << "\nBFS Levels:\n";
    for (int i = 0; i < V; ++i) {
        std::cout << "Node " << i << ": Level " << level[i] << std::endl;
    }

    return 0;
}