#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include "random.hpp"

__device__ int  BLUR_SIZE = 3;

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
            // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++ blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                    // Verify we have a valid image pixel
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }
        // Write our new pixel value out
        out[row*w + col] = (unsigned char) (pixVal / pixels);
    }
}

int main() {
    RandomNumberGenerator &rng = RandomNumberGenerator::getInstance(0, 255);

    const int width = 224;
    const int height = 224;
    const int image_size = width * height;

    unsigned char h_input[image_size];
    for (int i = 0; i < image_size; i++ ){
        h_input[i] = rng.getRandomInt();
    }

    unsigned char h_output[image_size];

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);

    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    blurKernel<<<gridSize, blockSize>>> (d_output, d_input, width, height);
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}