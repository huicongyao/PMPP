/* 
    this file should be compiled with `-rdc=true`
 */


#include "utils.hpp"
__device__ void doSomeWork(unsigned int in) {}

__device__ void doMoreWork(unsigned int in) {}

__global__ void kernel(unsigned int* start, unsigned int* end, 
    float* someData, float* moreData) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    doSomeWork(someData[i]);
    for (unsigned int j = start[i]; j < end[i]; ++j) {
        doMoreWork(moreData[j]);
    }
}

__global__ void kernel_child(unsigned int start, unsigned int end, 
    float* moreData) {
    unsigned int j = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (j < end) {
        doMoreWork(moreData[j]);
    }
}

__global__ void kernel_parent(unsigned int * start, unsigned int * end, 
    float *someData, float * moreData) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    doSomeWork(someData[i]);
    kernel_child<<<_ceil(end[i] - start[i], 256), 256>>> (start[i], end[i], moreData);
}



int main() {
    constexpr int num_elements = 1024;
    constexpr int more_data_size = 4096;

    UnifiedPtr<unsigned int> start(num_elements, true);
    UnifiedPtr<unsigned int> end(num_elements, true);
    UnifiedPtr<float> some_Data(num_elements, true);
    UnifiedPtr<float> more_Data(more_data_size, true);

    // Initialize input data
    for (int i = 0; i < num_elements; ++i) {
        start[i] = i * 4; // Start indices
        end[i] = start[i] + 4; // End indices
        some_Data[i] = static_cast<float>(i); // Some data
    }
    for (int i = 0; i < more_data_size; ++i) {
        more_Data[i] = static_cast<float>(i); // More data
    }

    // Launch the `kernel` function
    constexpr int block_size = 256;
    constexpr int num_blocks = (num_elements + block_size - 1) / block_size;

    std::cout << "Launching kernel..." << std::endl;
    kernel<<<num_blocks, block_size>>>(start.get(), end.get(), some_Data.get(), more_Data.get());
    cudaDeviceSynchronize();

    // Launch the `kernel_parent` function
    std::cout << "Launching kernel_parent..." << std::endl;
    kernel_parent<<<num_blocks, block_size>>>(start.get(), end.get(), some_Data.get(), more_Data.get());
    cudaDeviceSynchronize();

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}