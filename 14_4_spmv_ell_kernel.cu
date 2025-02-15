#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include "utils.hpp"


struct ELLMatrix {
    unsigned int numRows;
    unsigned int * nnzPerRow;
    unsigned int * colIdx;
    float * value;
};

__global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float *x, float *y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < ellMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < ellMatrix.nnzPerRow[row]; ++t) {
            unsigned int i = t * ellMatrix.numRows + row;
            unsigned int col = ellMatrix.colIdx[i];
            float value = ellMatrix.value[i];
            sum += x[col] * value;
        }
        y[row] = sum;
    }
}

int main() {
    unsigned int numRows = 4;
    unsigned int nnzPerRow[] = {2, 3, 2, 1};
    unsigned int colIdx[] = {0, 0, 1, 3, 1, 2, 2, 999, 999, 3, 999, 999};
    float h_value[] = {1, 5, 2, 6, 7, 3, 8, 0, 0, 9, 0, 0}; // Non-zero values
    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f}; // Input vector
    float h_y[] = {0.0f, 0.0f, 0.0f, 0.0f}; // Output vector (initialized to zero)

    // Device-site memory allocation
    unsigned int *d_nnzPerRow, *d_colIdx;
    float *d_value, *d_x, *d_y;
    cudaMalloc(&d_nnzPerRow, sizeof(nnzPerRow));
    cudaMalloc(&d_colIdx, sizeof(colIdx));
    cudaMalloc(&d_value, sizeof (h_value));
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMalloc(&d_y, sizeof(h_y));

    cudaMemcpy(d_nnzPerRow, nnzPerRow, sizeof(nnzPerRow), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx, sizeof(colIdx), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value, sizeof(h_value), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    ELLMatrix d_ellMatrix = {numRows, d_nnzPerRow, d_colIdx, d_value};

    dim3 blockDim(32);  
    dim3 gridDim((numRows + blockDim.x - 1) / blockDim.x);

    spmv_ell_kernel<<<gridDim, blockDim>>>(d_ellMatrix, d_x, d_y);

    cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost);

    float expected_y[] = {15.0f, 50.0f, 28.0f, 24.0f}; 
    for (unsigned int i = 0; i < numRows; ++i) {  
        if (fabs(h_y[i] - expected_y[i]) > 1e-3) {
            printf("%f %f\n", h_y[i], expected_y[i]);
            std::cerr << "Test failed!\n";
            // Free device memory
            cudaFree(d_nnzPerRow);  
            cudaFree(d_colIdx);  
            cudaFree(d_value);  
            cudaFree(d_x);  
            cudaFree(d_y);
            return ;
        }
    }  
    std::cout << "Test passed!" << std::endl;

    cudaFree(d_nnzPerRow);  
    cudaFree(d_colIdx);  
    cudaFree(d_value);  
    cudaFree(d_x);  
    cudaFree(d_y);

    return 0;  
}