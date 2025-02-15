#include <iostream>
#include <cstdlib> // For rand()
#include <cuda_runtime.h>
#include <algorithm>
#include "utils.hpp"

__host__ __device__ void merge_sequential(int *A, int m, int *B, int n, int *C) {
    // printf("%d %d %d\n", m, n, C[0]);
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


__global__ void merge_tiled_kernel(int *A, int m, int *B, int n, int *C, int tile_size) {
    /* shared memory allocation */
    extern __shared__ int sharedAB[];
    int * A_S = &sharedAB[0];                   // shareA is first half of shareAB
    int * B_S = &sharedAB[tile_size];           // shareB is second half of shareAB
    int C_curr = blockIdx.x * ((m + n + gridDim.x - 1) / gridDim.x ); // Start point of block's C subarray
    int C_next = min((blockIdx.x + 1) * ((m + n + gridDim.x - 1) / gridDim.x ), (m + n));
    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);   // Make block-level co-rank values visible
        A_S[1] = co_rank(C_next, A, m, B, n);   // to other threads in the block
    }
    __syncthreads();
    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();
    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = _ceil(C_length, tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    while(counter < total_iteration) {
        /* loading tile-size A and B elements into shared memory */
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed) {
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed) {
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();
        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
        /* find co-rank for c_curr and c_next */
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed), 
                                B_S, min(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed), 
                                B_S, min(tile_size, B_length - B_consumed));
        int b_next = c_next - a_next;
        /* All threads call the sequential merge function */
        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr,
            C + C_curr + C_completed + c_curr);
        /* Update the number of A and B elements that have been consumed thus far */
        counter++;
        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}

int main() {
    constexpr int m = 33000;
    constexpr int n = 31000;
    auto A = UnifiedPtr<int>(m, true);
    auto B = UnifiedPtr<int>(n, true);
    auto C = UnifiedPtr<int>(m + n, true);
    auto C_h = UnifiedPtr<int>(m + n);

    for (int i = 0; i < m; i++) {
        A[i] =  rand() ;
    }
    for (int i = 0; i < n; i++) {
        B[i] =  rand() ;
    }
    for (int i = 0; i < m + n; i++) {
        C[i] =  -1;
    }

    std::sort(A.get() , A.get() + m);
    std::sort(B.get() , B.get() + n);
    constexpr int grid_dim = 128, block_dim = 64, tile_size = 128;
    merge_tiled_kernel<<<grid_dim, block_dim, tile_size * 2>>>(A.get(), m, B.get(), n, C.get(), tile_size);
    cudaDeviceSynchronize();
    merge_sequential(A.get(), m, B.get(), n, C_h.get());

    for (int i = 0; i < m + n; i++) {
        if (C[i] != C_h[i] ) {
            std::cout << "index: " << i << ", " << C[i] << ", " << C_h[i] << std::endl;
            std::cout << "test failed!\n";
            return;
        }
    }
    std::cout << "test succeess!\n";
    return;
}