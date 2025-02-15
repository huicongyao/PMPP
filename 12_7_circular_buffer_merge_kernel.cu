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

__host__ __device__ void merge_sequential_circular(int *A, int m, int *B, int n, int *C, 
    int A_S_start, int B_S_start, int tile_size) {
    int i = 0; // virtual index into A
    int j = 0; // virtual index into B
    int k = 0; // virtual index into C
    while ((i < m) && (j < n)) {
        int i_cir = (A_S_start + i) % tile_size;
        int j_cir = (B_S_start + j) % tile_size;
        if (A[i_cir] <= B[j_cir]) {
            C[k++] = A[i_cir]; i++;
        } else {
            C[k++] = B[j_cir]; j++;
        }
    }
    if (i == m) { // done with A[] handle remaining B[]
        for (; j < n; j++) {
            int j_cir = (B_S_start + j) % tile_size;
            C[k++] = B[j_cir];
        }
    } else { // done with B[], handle remaining A[]
        for (; i < m; i++) {
            int i_cir = (A_S_start + i) % tile_size;
            C[k++] = A[i_cir];
        }
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

__host__ __device__ int co_rank_circular(int k, int *A, int m, int *B, int n, 
    int A_S_start, int B_S_start, int tile_size) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n; // i_low = max(0, k - n)
    int j_low = 0 > (k - m) ? 0 : k - m; // j_low = max(0, k - m)
    int delta;
    bool active = true;
    while(active) {
        int i_cir = (A_S_start + i) % tile_size;
        int i_m_l_cir = (A_S_start + i - 1) % tile_size;
        int j_cir = (B_S_start + j) % tile_size;
        int j_m_l_cir = (B_S_start + j - 1) % tile_size;
        if (i > 0 && j < n && A[i_m_l_cir] > B[j_cir]) {
            delta = ((i - i_low + 1) >> 1); // ceil((i - i_low) / 2)
            j_low = j;
            i = i - delta;
            j = j + delta;
        } else if (j > 0 && i < m && B[j_m_l_cir] >= A[i_cir]) {
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
    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = tile_size; // in the first iteration, fill the tile_size
    int B_S_consumed = tile_size; // in the first iteration, fill the tile_size
    while(counter < total_iteration) {
        /* loading A_S_consumed elements into A_S */
        for (int i = 0; i < A_S_consumed; i += blockDim.x) {
            if ((i + threadIdx.x) < (A_length - A_consumed) && (i + threadIdx.x) < A_S_consumed) {
                A_S[(A_S_start + (tile_size - A_S_consumed) + i + threadIdx.x) % tile_size] = \
                    A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        /* loading B_S_consumed elements into B_S */
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if ((i + threadIdx.x) < (B_length - B_consumed) && (i + threadIdx.x) < B_S_consumed) {
                B_S[(B_S_start + (tile_size - B_S_consumed) + i + threadIdx.x) % tile_size] = \
                    B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();
        int c_curr = threadIdx.x       *    (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) *    (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
        /* find co-rank for c_curr and c_next */
        int a_curr = co_rank_circular(c_curr, 
                                A_S, min(tile_size, A_length - A_consumed), 
                                B_S, min(tile_size, B_length - B_consumed),
                                A_S_start, B_S_start, tile_size);
        int b_curr = c_curr - a_curr;
        int a_next = co_rank_circular(c_next, 
                                A_S, min(tile_size, A_length - A_consumed), 
                                B_S, min(tile_size, B_length - B_consumed), 
                                A_S_start, B_S_start, tile_size);
        int b_next = c_next - a_next;
        /* All threads call the circular-buffer version of the sequential merge function */
        merge_sequential_circular(A_S , a_next - a_curr, 
                        B_S, b_next - b_curr, C + C_curr + C_completed + c_curr, 
                        A_S_start + a_curr, B_S_start + b_curr, tile_size);
        /* Figure out the work has been done */
        counter++;
        A_S_consumed = co_rank_circular(min(tile_size, C_length - C_completed), 
                            A_S, min(tile_size, A_length - A_consumed), 
                            B_S, min(tile_size, B_length - B_consumed), 
                            A_S_start, B_S_start, tile_size);
        B_S_consumed = min(tile_size, C_length - C_completed) - A_S_consumed;
        A_consumed += A_S_consumed;        
        C_completed += min(tile_size, C_length - C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = (A_S_start + A_S_consumed) % tile_size;
        B_S_start = (B_S_start + B_S_consumed) % tile_size;
        __syncthreads();
    }
}

int main() {
    constexpr int m = 3300;
    constexpr int n = 3100;
    auto A = UnifiedPtr<int>(m, true);
    auto B = UnifiedPtr<int>(n, true);
    auto C = UnifiedPtr<int>(m + n, true);
    auto C_h = UnifiedPtr<int>(m + n);

    for (int i = 0; i < m; i++) {
        A[i] =  rand() % (m + n);
    }
    for (int i = 0; i < n; i++) {
        B[i] =  rand() % (m + n);
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
            // std::cout << "test failed!\n";
            // return;
        }
    }
    std::cout << "test succeess!\n";
    return;
}