#include <iostream>
#include <cstdlib> // For rand()
#include <cuda_runtime.h>
#include <algorithm>
#include "utils.hpp"

__host__ __device__ void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0; // Index into A
    int j = 0; // Index into B
    int k = 0; // Index into C
    while((i < m) && (j < n)) { // Handle start of A[] and B[]  
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == m) {
        while( j < n) C[k++] = B[j++];
    } else {
        while (i < m) C[k++] = A[i++];
    }
}

__host__ __device__ int co_rank(int k, int *A, int m, int *B, int n) {
    int i = k < m ? k : m; // i = min(k, m)
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n; // i_low = max(0, k - n)
    int j_low = 0 > (k - m) ? 0 : k - m; // i_low = max(0, k - m)
    int delta;
    bool active = true;
    while( active ) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = ((i - i_low + 1) >> 1); // ceil((i - i_low) / 2)
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j-1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

__global__ void merge_basic_kernel(int *A, int m, int *B, int n, int *C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int elementsPerThread = (m + n + total_threads - 1) / total_threads;
    int k_curr = tid * elementsPerThread; // start output index
    int k_next = min((tid + 1) * elementsPerThread, m + n); // end output index
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    // printf("i_curr = %d, i_next = %d, j_curr = %d, j_next = %d, k_curr = %d, k_next = %d\n", i_curr, i_next, j_curr, j_next, k_curr, k_next);
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}


int main() {
    constexpr int m = 37;
    constexpr int n = 47;
    auto A = UnifiedPtr<int>(m, true);
    auto B = UnifiedPtr<int>(n, true);
    auto C = UnifiedPtr<int>(m + n, true);
    auto C_h = UnifiedPtr<int>(m + n);

    for (int i = 0; i < m; i++) {
        A[i] =  rand() % 100;
    }
    for (int i = 0; i < n; i++) {
        B[i] =  rand() % 100;
    }
    for (int i = 0; i < m + n; i++) {
        C[i] =  -1;
    }

    std::sort(A.get() , A.get() + m);
    std::sort(B.get() , B.get() + n);

    merge_basic_kernel<<<3, 5>>>(A.get(), m, B.get(), n, C.get());
    cudaDeviceSynchronize();
    merge_sequential(A.get(), m, B.get(), n, C_h.get());

    for (int i = 0; i < m + n; i++) {
        if (C[i] != C_h[i]) {
            std::cout << "test failed!\n";
            return;
        }
    }
    std::cout << "test succeess!\n";
    return;
}   