#include <iostream>
#include <climits>
#include <cuda_runtime.h>

// Your struct and kernels from the question
struct CSRGraph {
    unsigned int numVertices;
    unsigned int *srcPtrs;
    unsigned int *dst;
};

struct CSCGraph {
    unsigned int numVertices;
    unsigned int *src;
    unsigned int *dstPtrs;
};

__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level, 
                            unsigned int *newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < csrGraph.numVertices) {
        if (level[vertex] == currLevel - 1) {
            for (unsigned int edge = csrGraph.srcPtrs[vertex]; 
                 edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if (level[neighbor] == UINT_MAX) { // Neighbor not visited
                    level[neighbor] = currLevel;
                    *newVertexVisited = 1;
                }
            }
        }
    }
}

__global__ void bfs_kernel_(CSCGraph cscGraph, unsigned int *level, 
                            unsigned int *newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < cscGraph.numVertices) {
        if (level[vertex] == UINT_MAX) { // vertex not yet visited
            for (unsigned int edge = cscGraph.dstPtrs[vertex]; 
                 edge < cscGraph.dstPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = cscGraph.src[edge];
                if (level[neighbor] == currLevel - 1) {
                    level[vertex] = currLevel;
                    *newVertexVisited = 1;
                    break;
                }
            }

        }
    }
}

int main() {
    unsigned int numVertices = 9;
    unsigned int edges = 15;
    unsigned int h_srcPtrs[10] = {0,2,4,7,9,11,12,13,15,15};

    unsigned int h_dst[15] = {1,2,3,4,5,6,7,4,8,5,8,6,8,0,6};

    // src array in CSC is the list of "sources" grouped by "dest"
    unsigned int h_cscSrc[15] = {
       7,0,0,1,1,3,2,4,2,5,7,2,3,4,6
    };
    unsigned int h_cscDstPtrs[] = {0,1,2,3,4,6,8,11,12,15};


    // Allocate device memory
    unsigned int *d_srcPtrs, *d_dst;
    unsigned int *d_cscSrc, *d_cscDstPtrs;
    unsigned int *d_level, *d_newVertexVisited;

    cudaMalloc((void**)&d_srcPtrs, (numVertices+1)*sizeof(unsigned int));
    cudaMalloc((void**)&d_dst, edges*sizeof(unsigned int));

    cudaMalloc((void**)&d_cscSrc, edges*sizeof(unsigned int));
    cudaMalloc((void**)&d_cscDstPtrs, (numVertices+1)*sizeof(unsigned int));

    cudaMemcpy(d_srcPtrs, h_srcPtrs, (numVertices+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, edges*sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_cscSrc, h_cscSrc, edges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscDstPtrs, h_cscDstPtrs, (numVertices+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Initialize level array
    unsigned int *h_level = new unsigned int[numVertices];
    for (unsigned int i = 0; i < numVertices; i++) {
        h_level[i] = UINT_MAX;
    }
    h_level[0] = 0; // start BFS at vertex 0

    cudaMalloc((void**)&d_level, numVertices*sizeof(unsigned int));
    cudaMemcpy(d_level, h_level, numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_newVertexVisited, sizeof(unsigned int));

    // Setup graph structs
    CSRGraph d_csrGraph;
    d_csrGraph.numVertices = numVertices;
    d_csrGraph.srcPtrs = d_srcPtrs;
    d_csrGraph.dst = d_dst;

    CSCGraph d_cscGraph;
    d_cscGraph.numVertices = numVertices;
    d_cscGraph.src = d_cscSrc;
    d_cscGraph.dstPtrs = d_cscDstPtrs;

    // BFS parameters
    unsigned int currLevel = 1;
    unsigned int blockSize = 128;
    unsigned int gridSize = (numVertices + blockSize - 1) / blockSize;

    // Run BFS using CSR graph
    // Loop until no new vertices visited in a frontier
    bool done = false;
    while (!done) {
        unsigned int h_newVisited = 0;
        cudaMemcpy(d_newVertexVisited, &h_newVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);

        bfs_kernel<<<gridSize, blockSize>>>(d_csrGraph, d_level, d_newVertexVisited, currLevel);

        cudaMemcpy(&h_newVisited, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (h_newVisited == 0) {
            done = true;
        } else {
            currLevel++;
        }
        cudaDeviceSynchronize();
    }

    // Copy back the levels
    cudaMemcpy(h_level, d_level, numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "BFS Levels (CSR):\n";
    for (unsigned int i = 0; i < numVertices; i++) {
        if (h_level[i] == UINT_MAX) {
            std::cout << i << ": INF\n";
        } else {
            std::cout << i << ": " << h_level[i] << "\n";
        }
    }

    // Reset for CSC test
    for (unsigned int i = 0; i < numVertices; i++) {
        h_level[i] = UINT_MAX;
    }
    h_level[0] = 0;
    cudaMemcpy(d_level, h_level, numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice);

    currLevel = 1;
    done = false;
    while (!done) {
        unsigned int h_newVisited = 0;
        cudaMemcpy(d_newVertexVisited, &h_newVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);

        bfs_kernel_<<<gridSize, blockSize>>>(d_cscGraph, d_level, d_newVertexVisited, currLevel);

        cudaMemcpy(&h_newVisited, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (h_newVisited == 0) {
            done = true;
        } else {
            currLevel++;
        }
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_level, d_level, numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "\nBFS Levels (CSC):\n";
    for (unsigned int i = 0; i < numVertices; i++) {
        if (h_level[i] == UINT_MAX) {
            std::cout << i << ": INF\n";
        } else {
            std::cout << i << ": " << h_level[i] << "\n";
        }
    }

    // Cleanup
    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_cscSrc);
    cudaFree(d_cscDstPtrs);
    cudaFree(d_level);
    cudaFree(d_newVertexVisited);
    delete[] h_level;

    return 0;
}
