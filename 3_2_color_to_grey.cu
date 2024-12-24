#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <random>
#include <cstdlib>
#include "utils.hpp"

// The input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels (RGB)
__global__
void colortoGrayscaleConvertion(UnifiedPtr<unsigned char> Pout,
                                UnifiedPtr<unsigned char> Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int grayOffset = row * width + col;
        // One can think of the RGB image having CHANNEL
        // times more columns than the gray scale image
        int rgbOffset = grayOffset * 3;
        unsigned char r = Pin[rgbOffset    ]; // Red value
        unsigned char g = Pin[rgbOffset + 1]; // Green value
        unsigned char b = Pin[rgbOffset + 2]; // Blue value
        // Perform the rescaling and store it
        // We multiply by floating point constants
        Pout[grayOffset] =  0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void checkCudaError(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << msg << " - "
                  << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Host function to convert RGB to grayscale (for verification)
unsigned char rgbToGray(unsigned char r, unsigned char g, unsigned char b) {
    return static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
}

int main() {
    // Define a small 3x2 RGB image (6 pixels, 18 bytes total)
    constexpr int width = 62;
    constexpr int height = 76;
    constexpr int channels = 3;
    constexpr int rgbImageSize = width * height * channels; // 18 bytes
    constexpr int grayImageSize = width * height;           // 6 bytes

    // Host memory allocation for input (RGB) and output (grayscale) images
    UnifiedPtr<unsigned char> input(rgbImageSize, true);
    for (int i = 0; i < rgbImageSize; i++) {
        input[i] = rand() % 256;
    }
    UnifiedPtr<unsigned char> output(grayImageSize, true);
    for (int i = 0; i < grayImageSize; i++) {
        output[i] = rand() % 256;
    }
    // Define grid and block dimensions
    dim3 blockSize(16, 16);  // Block size of 16x16 threads
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    colortoGrayscaleConvertion<<<gridSize, blockSize>>>(output, input, width, height);
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Copy the result back to the host

    // Verify the result
    bool testPassed = true;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int grayOffset = i * width + j;
            int rgbOffset = grayOffset * channels;

            unsigned char expectedGray = rgbToGray(
                    input[rgbOffset], input[rgbOffset + 1], input[rgbOffset + 2]);

            if (output[grayOffset] != expectedGray) {
                std::cerr << "Test failed at pixel (" << i << ", " << j << "): "
                          << "Expected " << (int)expectedGray << ", but got "
                          << (int)output[grayOffset] << std::endl;
                testPassed = false;
            }
        }
    }

    if (testPassed) {
        std::cout << "Test passed: All grayscale values match!" << std::endl;
    } else {
        std::cerr << "Test failed: Mismatch in grayscale values." << std::endl;
    }
    return testPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}