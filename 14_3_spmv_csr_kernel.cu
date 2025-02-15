#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

struct CSRMatrix {
    unsigned int numRows;
    unsigned int * rowPtrs;
    unsigned int * colIdx;
    float * value;
};

__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float *x, float *y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < csrMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i) {
            unsigned int col = csrMatrix.colIdx[i];
            float value = csrMatrix.value[i];
            sum += x[col] * value;
        }
        y[row] += sum;
    }
}


void test_spmv_csr_kernel() {
    // Host-side CSR matrix representation
    unsigned int h_rowPtrs[] = {0, 2, 4, 7}; // Pointers to start of rows
    unsigned int h_colIdx[] = {0, 1, 0, 2, 1, 2, 3}; // Column indices
    float h_value[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f}; // Non-zero values
    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f}; // Input vector
    float h_y[] = {0.0f, 0.0f, 0.0f}; // Output vector (initialized to zero)

    unsigned int numRows = 3; // Number of rows in CSR matrix

    // Device-side memory allocation
    unsigned int *d_rowPtrs, *d_colIdx;
    float *d_value, *d_x, *d_y;

    cudaMalloc((void **)&d_rowPtrs, sizeof(h_rowPtrs));
    cudaMalloc((void **)&d_colIdx, sizeof(h_colIdx));
    cudaMalloc((void **)&d_value, sizeof(h_value));
    cudaMalloc((void **)&d_x, sizeof(h_x));
    cudaMalloc((void **)&d_y, sizeof(h_y));

    // Copy data from host to device
    cudaMemcpy(d_rowPtrs, h_rowPtrs, sizeof(h_rowPtrs), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, sizeof(h_colIdx), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value, sizeof(h_value), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice);

    // CSRMatrix on the device
    CSRMatrix d_csrMatrix = {numRows, d_rowPtrs, d_colIdx, d_value};

    // Kernel execution parameters
    dim3 blockDim(128);
    dim3 gridDim((numRows + blockDim.x - 1) / blockDim.x);

    // Launch kernel
    spmv_csr_kernel<<<gridDim, blockDim>>>(d_csrMatrix, d_x, d_y);

    // Copy results back to host
    cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost);

    // Expected results (manually computed for validation)
    float expected_y[] = {50.0f, 150.0f, 560.0f};

    // Validate results
    for (unsigned int i = 0; i < numRows; ++i) {
        if (fabs(h_y[i] - expected_y[i]) > 1e-3) {
            printf("%f %f\n", h_y[i], expected_y[i]);
            std::cerr << "Test failed!\n";
            // Free device memory
            cudaFree(d_rowPtrs);
            cudaFree(d_colIdx);
            cudaFree(d_value);
            cudaFree(d_x);
            cudaFree(d_y);
            return ;
        }
    }

    std::cout << "Test passed!" << std::endl;

    // Free device memory
    cudaFree(d_rowPtrs);
    cudaFree(d_colIdx);
    cudaFree(d_value);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    test_spmv_csr_kernel();
    return 0;
}
