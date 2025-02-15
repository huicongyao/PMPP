#include <iostream>
#include <thread>
#include <chrono>
constexpr float c0 = 0.5;
constexpr float c1 = 0.5;
constexpr float c2 = 0.5;
constexpr float c3 = 0.5;
constexpr float c4 = 0.5;
constexpr float c5 = 0.5;
constexpr float c6 = 0.5;


__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] 
                                    + c1 * in[i * N * N + j * N + (k - 1)]
                                    + c2 * in[i * N * N + j * N + (k + 1)]
                                    + c3 * in[i * N * N + (j - 1) * N + k]
                                    + c4 * in[i * N * N + (j + 1) * N + k]
                                    + c5 * in[(i - 1) * N * N + j * N + k]
                                    + c6 * in[(i + 1) * N * N + j * N + k];
    }
}

constexpr int OUT_TILE_DIM = 4;
constexpr int IN_TILE_DIM = OUT_TILE_DIM + 2;
__global__ void stencil_kernel_mem_tiling(float *in, float *out, unsigned int N) {
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && 
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && 
            threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1 ) {
            if (i == 1 && j == 1 && k == 1) {
                printf("%d, %d, %d \n%d, %d, %d \n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x);
            }
            out[i * N * N + j * N + k] = c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                                        + c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x-1]
                                        + c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x+1]
                                        + c3 * in_s[threadIdx.z][threadIdx.y-1][threadIdx.x]
                                        + c4 * in_s[threadIdx.z][threadIdx.y+1][threadIdx.x]
                                        + c5 * in_s[threadIdx.z-1][threadIdx.y][threadIdx.x]
                                        + c6 * in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
            
        }
    }
}

constexpr float TOL = 1e-5;

// Host function to verify the result
void verify_result(float *in, float *out1, float *out2, unsigned int N) {
    bool is_success_1 = true;
    bool is_success_2 = true;
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

                // Check stencil_kernel output
                if (fabs(out1[idx] - expected) > TOL) {
                    is_success_1 = false;
                    std::cerr << "Mismatch in stencil_kernel at index " << idx
                              << ": " << out1[idx] << " != " << expected << std::endl;
                    
                }

                // Check stencil_kernel_mem_tiling output
                if (fabs(out2[idx] - expected) > TOL) {
                    is_success_2 = false;
                    std::cerr << "Mismatch in stencil_kernel_mem_tiling at index " << idx
                              << ", : " << out2[idx] << " != " << expected << std::endl;
                    printf("%d %d %d\n %f %f %f %f %f %f %f\n ", i, j ,k, 
                        in[i * N * N + j * N + k], 
                        in[i * N * N + j * N + k - 1], 
                        in[i * N * N + j * N + k + 1], 
                        in[i * N * N + (j - 1) * N + k], 
                        in[i * N * N + (j + 1) * N + k], 
                        in[(i - 1) * N * N + j * N + k], 
                        in[(i + 1) * N * N + j * N + k]);
                }
                if (!is_success_1 || !is_success_2) break;
            }
            if (!is_success_1 || !is_success_2) break;

        }
        if (!is_success_1 || !is_success_2) break;
    }
    if (is_success_1) 
        std::cout << "stencil simple kernel passed!\n";
    else std::cout << "stencil simple kernel failed, found mismatches\n";
    if (is_success_2) 
        std::cout << "stencil mem tiling kernel passed!\n";
    else std::cout << "stencil mem tiling kernel failed, found mismatches\n";
}

int main() {
    constexpr unsigned int N = 500; // Small grid size for testing
    constexpr unsigned int NUM_ELEMENTS = N * N * N;

    // Allocate unified memory
    float *in, *out1, *out2;
    cudaMallocManaged(&in, NUM_ELEMENTS * sizeof(float));
    cudaMallocManaged(&out1, NUM_ELEMENTS * sizeof(float));
    cudaMallocManaged(&out2, NUM_ELEMENTS * sizeof(float));

    // Initialize input array with some values
    for (unsigned int i = 0; i < NUM_ELEMENTS; ++i) {
        in[i] = static_cast<float>(rand() % 10); // Simple modulo to fill array
    }

    // Zero out output arrays
    std::fill(out1, out1 + NUM_ELEMENTS, 0.0f);
    std::fill(out2, out2 + NUM_ELEMENTS, 0.0f);

    // Configure kernel launch parameters
    dim3 threadsPerBlock(10, 10, 10);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Launch stencil_kernel
    stencil_kernel<<<blocksPerGrid, threadsPerBlock>>>(in, out1, N);
    cudaDeviceSynchronize();

    // Launch stencil_kernel_mem_tiling
    dim3 tiledThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 tiledBlocksPerGrid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                            (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                            (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    stencil_kernel_mem_tiling<<<tiledBlocksPerGrid, tiledThreadsPerBlock>>>(in, out2, N);
    cudaDeviceSynchronize();

    // Verify the results
    verify_result(in, out1, out2, N);

    // Free unified memory
    cudaFree(in);
    cudaFree(out1);
    cudaFree(out2);
    return 0;
}