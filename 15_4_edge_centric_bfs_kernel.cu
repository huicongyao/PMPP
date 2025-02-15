
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <climits>
#include "utils.hpp"

#define NUM_THREADS 32

struct COOGraph {
    unsigned int *src;
    unsigned int *dst;
    unsigned int numEdges;
};

__global__ void bfs_kernel (COOGraph cooGraph, unsigned int *level,
        unsigned int* newVertexVisited, unsigned int currLevel) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < cooGraph.numEdges) {
        unsigned int vertex = cooGraph.src[edge];
        if (level[vertex] == currLevel - 1) {
            unsigned int neighbor = cooGraph.dst[edge];
            if (level[neighbor] == UINT_MAX) {
                level[neighbor] = currLevel;
                *newVertexVisited = 1;
            }
        }
    }
}



// Host function to check BFS levels
void verifyBFS(UnifiedPtr<unsigned int>& level) {
    std::vector<unsigned int> expected = {0, 1, 1, 2, 2, 2, 2, 2, 3};
    for (size_t i = 0; i < level.size(); ++i) {
        if (level[i] != expected[i]) {
            std::cerr << "Mismatch at vertex " << i << ": Expected "
                      << expected[i] << ", Got " << level[i] << std::endl;
            return;
        }
    }
    std::cout << "BFS levels verified successfully!" << std::endl;
}

int main() {
    // Input graph as shown in the photo
    const unsigned int numVertices = 9;
    const unsigned int numEdges = 15;

    // Edge list (source -> destination)
    std::vector<unsigned int> src = {0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7};
    std::vector<unsigned int> dst = {1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6};

    UnifiedPtr<unsigned int> d_src ( std::initializer_list<unsigned int>{0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7}, true);
    UnifiedPtr<unsigned int> d_dst ( std::initializer_list<unsigned int>{1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6}, true);
    COOGraph cooGraph = {d_src.get(), d_dst.get(), numEdges};

    UnifiedPtr<unsigned int> d_newVertexVisited (1, true);
    UnifiedPtr<unsigned int> d_level (numVertices, true);
    for (int i = 1; i < numVertices; ++i) d_level[i] = UINT_MAX;
    d_level[0] = 0;

    // Launch BFS kernel iteratively
    unsigned int currLevel = 1;
    while (true) {
        // cudaMemset(d_newVertexVisited, 0, sizeof(unsigned int));
        d_newVertexVisited[0] = 0;
        unsigned int blocks = (numEdges + NUM_THREADS - 1) / NUM_THREADS;
        bfs_kernel<<<blocks, NUM_THREADS>>>(cooGraph, d_level.get(), d_newVertexVisited.get(), currLevel);
        cudaDeviceSynchronize();
        // cudaMemcpy(newVertexVisited, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (d_newVertexVisited[0]== 0) break; // No new vertices visited, stop BFS
        currLevel++;
    }

    // Verify BFS results
    verifyBFS(d_level);

    return 0;
}
