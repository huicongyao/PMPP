#include <iostream>
#include <cstdlib> // For rand()
#include <cuda_runtime.h>
#include <algorithm>
#include <bitset>
#include "utils.hpp"

constexpr unsigned int array_size = 10270;

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

__host__ __device__ void _swap(int *arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

__host__ __device__ void reverse(int *arr, int st, int ed) {
    int i = st;
    int j = ed - 1;
    while (i < j) {
        _swap(arr, i, j);
        i++;
        j--;
    }
}

__host__ __device__ void exchange(int * arr, int st, int mid, int ed) {
    reverse(arr, st, mid);
    reverse(arr, mid, ed);
    reverse(arr, st, ed);
}

__host__ __device__ void implace_merge_sequential(int *arr, int st, int mid, int ed) {
    int i = st;
    int j = mid;
    int k = ed - 1;
    while (i < j && j < ed) {
        int step = 0;
        while (i < j && arr[i] <= arr[j]) {
            i++;
        }
        while (j <= k && arr[j] < arr[i]) {
            j++;
            step++;
        }
        exchange(arr, i, j - step, j);
    }
}

// Function to perform Shell Sort
__host__ __device__ void shellsort(int *arr, int st, int ed) {
    int n = ed - st + 1;
    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = st + gap; i <= ed; i++) {
            int temp = arr[i];
            int j;
            for (j = i; j >= st + gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

__host__ __device__ void quick_sort(int *arr, int left, int right) {
    if (left >= right) return;

    int pivot = arr[(left + right) >> 1];
    int i = left - 1;
    int j = right + 1;
    while (i < j) {
        do i++; while (arr[i] < pivot);
        do j--; while (arr[j] > pivot);
        if (i < j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    quick_sort(arr, left, j);
    quick_sort(arr, j + 1, right);
}


__global__ void parallel_merge_sort(int *input, int *output, int array_size, int minimum_sort, int *block_sync_flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = 1;
    int cnt = 0;
    while (stride < array_size) {
        cnt ++;
        if ((2 * i + 1) * stride <= array_size) {
            int st_1 = 2 * i * stride;
            int st_2 = (2 * i + 1) * stride;
            int ed = min((2 * i + 2) * stride , array_size);
            if (stride == 1)
                merge_sequential(input + st_1, stride, input + st_2, ed - st_2, output + st_1);
            else
                implace_merge_sequential(output ,st_1, st_2, ed);
            // printf("thread: %d, stride %d, merge [%d/%d] and [%d/%d]\n", i, stride, st_1, st_1 + stride, st_2, st_2 + stride);
            
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(block_sync_flag, 1);
            // printf("%d %d\n", stride, *block_sync_flag);
        }

        while(atomicAdd(block_sync_flag, 0) < gridDim.x * cnt ) {}
        // if (i == 0) *block_sync_flag = 0;
        // if (i == 0) {
        //     printf("stride = %d: ", stride);
        //     for (int i = 0; i < array_size; i++) {
        //         printf(" %d", output[i]);
        //     }
        //     printf("\n");
        // }
        stride *= 2;
    }
}

void test(UnifiedPtr<int> & in1, UnifiedPtr<int> &in2) {
    // Verify correctness
    bool success = true;
    for (int i = 0; i < array_size; ++i) {
        if (in1[i] != in2[i]) {
            success = false;
            std::cout << "idx: " << i << ", " << in1[i] << ", " << in2[i] << "\n";
            // break;
        }
    }

    if (success) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    } 
}

int main() {
    UnifiedPtr<int> input(array_size, true);
    UnifiedPtr<int> input_2(array_size, true);
    UnifiedPtr<int> output(array_size, true);
    UnifiedPtr<int> block_sync_flag(1, true);
    for (int i = 0; i < array_size; i++) {
        input[i] = rand() % 10000;
        input_2[i] = input[i];
        output[i] = -1;
    }
    int minimum_sort = 16;
    int block_dim = 32;
    // int grid_dim = ((array_size + minimum_sort - 1) / minimum_sort + block_dim - 1) / block_dim;
    int grid_dim = (array_size + block_dim - 1) / block_dim;
    std::cout << grid_dim << " " << block_dim << std::endl;

    // for (int i = 0; i < array_size; i++) {
    //     printf(" %d", input[i]);
    // }
    // printf("\n");

    parallel_merge_sort<<<grid_dim, block_dim>>>(input.get(), output.get(), array_size, minimum_sort, block_sync_flag.get());
    cudaDeviceSynchronize();
    
    std::sort(input.get(), input.get() + array_size);
    shellsort(input_2.get(), 0, array_size - 1);
    test(input, input_2);
    test(input, output);
    return 0;
}