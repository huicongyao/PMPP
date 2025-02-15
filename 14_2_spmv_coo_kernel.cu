#include <iostream>
#include <cuda_runtime.h>

// Define the COOMatrix structure
struct COOMatrix {
    unsigned int* rowIdx; // Array of row indices
    unsigned int* colIdx; // Array of column indices
    float* value;         // Array of values
    unsigned int numNonzeros; // Number of non-zero elements
};


__global__ void spmv_coo_kernel(COOMatrix cooMatrix, float* x, float *y) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cooMatrix.numNonzeros) {
        unsigned int row = cooMatrix.rowIdx[i];
        unsigned int col = cooMatrix.colIdx[i];
        float value = cooMatrix.value[i];
        atomicAdd(&y[row], x[col]*value);
    }
}



int main() {
    // Example COO data from the figure
    unsigned int h_rowIdx[] = {1, 3, 2, 2, 0, 1, 1, 0};
    unsigned int h_colIdx[] = {3, 3, 1, 2, 0, 0, 2, 1};
    float h_value[] = {9, 6, 2, 8, 1, 5, 3, 7};
    float h_x[] = {1, 2, 3, 4}; // Example vector
    float h_y[4] = {0};        // Result vector initialized to zero

    unsigned int numNonzeros = 8; // Number of non-zero elements
    unsigned int numRows = 4;     // Number of rows in the matrix

    // Allocate device memory
    unsigned int *d_rowIdx, *d_colIdx;
    float *d_value, *d_x, *d_y;

    cudaMalloc((void**)&d_rowIdx, numNonzeros * sizeof(unsigned int));
    cudaMalloc((void**)&d_colIdx, numNonzeros * sizeof(unsigned int));
    cudaMalloc((void**)&d_value, numNonzeros * sizeof(float));
    cudaMalloc((void**)&d_x, numRows * sizeof(float));
    cudaMalloc((void**)&d_y, numRows * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_rowIdx, h_rowIdx, numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value, numNonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, numRows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, numRows * sizeof(float), cudaMemcpyHostToDevice);

    // Create COOMatrix struct on host and device
    COOMatrix cooMatrix = {d_rowIdx, d_colIdx, d_value, numNonzeros};
    // COOMatrix* d_cooMatrix;
    // cudaMalloc((void**)&d_cooMatrix, sizeof(COOMatrix));
    // cudaMemcpy(d_cooMatrix, &h_cooMatrix, sizeof(COOMatrix), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numNonzeros + threadsPerBlock - 1) / threadsPerBlock;
    spmv_coo_kernel<<<blocksPerGrid, threadsPerBlock>>>(cooMatrix, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < numRows; i++) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_rowIdx);
    cudaFree(d_colIdx);
    cudaFree(d_value);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}