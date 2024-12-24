#include <stdio.h>
#include <stdlib.h>
#include "utils.hpp"
#include <unistd.h>
#include <time.h>

void vecadd(float *A_h, float *B_h, float *C_h, int n) {
    for (int i = 0; i < n; i++) {

        C_h[i] = A_h[i] + B_h[i];
    }
}

__global__
void vecAddKernel(UnifiedPtr<float> A, UnifiedPtr<float> B, UnifiedPtr<float> C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n ) {
        C[i] = A[i] + B[i];
//        printf("thread: %d, %f, %f, %f\n", A[i], B[i], C[i]);
    }
}

int main() {
    int n = 10;
    UnifiedPtr<float>  A(n, true);
    UnifiedPtr<float>  B(n, true);
    UnifiedPtr<float>  C(n, true);

    for (int i = 0; i < n; i++) A[i] = 1;
    for (int i = 0; i < n; i++) B[i] = 2;

    // kernel invocation code
    vecAddKernel<<<ceil(n/256.), 256>>>(A, B, C, n);
    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++) {
        printf("%f\t", C[i]);
    }
    printf("\n");
}