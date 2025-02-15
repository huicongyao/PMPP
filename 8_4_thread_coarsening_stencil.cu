#include <iostream>
#include <ctime>
#include <unistd.h>
constexpr int OUT_TILE_DIM = 6;
constexpr int IN_TILE_DIM = OUT_TILE_DIM + 2;

constexpr float c0 = 0.5;
constexpr float c1 = 0.5;
constexpr float c2 = 0.5;
constexpr float c3 = 0.5;
constexpr float c4 = 0.5;
constexpr float c5 = 0.5;
constexpr float c6 = 0.5;


__global__ void stencil_kernel_mem_tiling_thread_coarsening(float *in, float *out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 
                && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] = \
                                             c0 * inCurr_s[threadIdx.y][threadIdx.x]
                                           + c1 * inCurr_s[threadIdx.y][threadIdx.x - 1]
                                           + c2 * inCurr_s[threadIdx.y][threadIdx.x + 1]
                                           + c3 * inCurr_s[threadIdx.y + 1][threadIdx.x]
                                           + c4 * inCurr_s[threadIdx.y - 1][threadIdx.x]
                                           + c5 * inPrev_s[threadIdx.y][threadIdx.x]
                                           + c6 * inNext_s[threadIdx.y][threadIdx.x];
                // if (inPrev_s[threadIdx.y][threadIdx.x] != in[(i - 1) * N * N + j * N + k]) {
                //     printf("%d %d %d\n %f %f \n", threadIdx.z, j ,k, 
                //     inPrev_s[threadIdx.y][threadIdx.x], 
                //     in[(i - 1) * N * N + j * N + k]);
                // }
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        // __syncthreads();
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
        // __syncthreads();
    }
}

constexpr float TOL = 1e-3;

// Host function to verify the result
void verify_result(float *in, float *out, unsigned int N) {
    for (unsigned int i = 1; i < N - 1; ++i) {
        for (unsigned int j = 1; j < N - 1; ++j) {
            for (unsigned int k = 1; k < N - 1; ++k) {
                unsigned int idx = i * N * N + j * N + k;

                float expected = 0.5f * in[idx] +
                                 0.5f * in[idx - 1] +   // k-1
                                 0.5f * in[idx + 1] +   // k+1
                                 0.5f * in[idx - N] +   // j-1
                                 0.5f * in[idx + N] +   // j+1
                                 0.5f * in[idx - N * N] + // i-1
                                 0.5f * in[idx + N * N]; // i+1

                // Check stencil_kernel_mem_tiling output
                if (fabs(out[idx] - expected) > TOL) {
                    std::cerr << "Mismatch in stencil_kernel_mem_tiling at index " << idx
                              << ", : " << out[idx] << " != " << expected << std::endl;
                    printf("%d %d %d\n %f %f %f %f %f %f %f\n ", i, j ,k, 
                        in[i * N * N + j * N + k], 
                        in[i * N * N + j * N + k - 1], 
                        in[i * N * N + j * N + k + 1], 
                        in[i * N * N + (j + 1) * N + k], 
                        in[i * N * N + (j - 1) * N + k], 
                        in[(i - 1) * N * N + j * N + k], 
                        in[(i + 1) * N * N + j * N + k]);
                    return;
                }
            }
        }
    }
    std::cout << "All tests passed!" << std::endl;
}

int main() {
    constexpr unsigned int N = 41; // Small grid size for testing
    constexpr unsigned int NUM_ELEMENTS = N * N * N;

    // Allocate unified memory
    float *in, *out;
    cudaMallocManaged(&in, NUM_ELEMENTS * sizeof(float));
    cudaMallocManaged(&out, NUM_ELEMENTS * sizeof(float));

    // Initialize input array with some values
    for (unsigned int i = 0; i < NUM_ELEMENTS; ++i) {
        // in[i] = static_cast<float>(rand() % 10); // Simple modulo to fill array
        in[i] = static_cast<float>(i % 10); // Simple modulo to fill array

    }

    // Zero out output arrays
    std::fill(out, out + NUM_ELEMENTS, 0.0f);
    // Launch stencil_kernel_mem_tiling
    dim3 tiledThreadsPerBlock(IN_TILE_DIM, 
                            // IN_TILE_DIM, 
                            IN_TILE_DIM); // only use two block dimensions
    dim3 tiledBlocksPerGrid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                            (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                            (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    stencil_kernel_mem_tiling_thread_coarsening<<<tiledBlocksPerGrid, tiledThreadsPerBlock>>>(in, out, N);
    cudaDeviceSynchronize();

    // Verify the results
    verify_result(in, out, N);

    // Free unified memory
    cudaFree(in);
    cudaFree(out);

    return 0;
}