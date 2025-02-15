#include <iostream>
#include <type_traits>
#include <cmath>

template <typename T>
T sigmoid(T value) {
    // Static assertion to ensure the type is a floating-point type
    static_assert(std::is_floating_point<T>::value, "Template argument must be a floating-point type.");

    return 1 / (1 + std::exp(-value));
}

void convLayer_forward(int M, int C, int H, int W, int K, 
    float *X, float *Weights, float *Y) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    
    for (int m = 0; m < M; m++) {                       // for each output feature map
        for (int h = 0; h < H_out; h++) {               // for each output element
            for (int w = 0; w < W_out; w++) {           
                Y[m, h, w] = 0;
                for (int c = 0; c < C; c++) {
                    for (int p = 0; p < K; p++) {
                        for (int q = 0; q < K; q++) {
                            Y[m, h, w] += X[c, h + p, w + q] * Weights[m, c, p, q];
                        }
                    }
                }
            }
        }
    }
}

void subsamplingLayer_forward(int M, int H, int W, int K, float *Y, float *S, float *b) {
    for (int m = 0; m < M; m++) {
        for (int h = 0; h < H / K; h++) {
            for (int w = 0; w < W / K; w++) {
                S[m, h, w] = 0.;
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        S[m, h, w] += Y[m, h * K + p, w * K + q] / (K * K);
                    }
                }
                S[m, h, w] = sigmoid(S[m, h, w] + b[m]);
            }
        }
    }
}

// void convLayer_backward_x_grad(int M, int C, int H_in, int W_in, int K,
//                 float* dE_dY, float * W, float* dE_dX) {
//     int H_out = H_in - K + 1;
//     int W_out = W_in - K + 1;
//     for (int c = 0; c < C; c++) 
//         for (int h = 0; h < H_in; h++)
//             for (int w = 0; w < W_in; w++)
//                 dE_dX[c, h, w] = 0;
//     for (int m = 0; m < M; m++)
//         for (int h = 0; h < H - 1; h++)
//             for (int w = 0; w < W - 1; w++)
//                 for (int c = 0; c < C; c++) 
//                     for (int p = 0; p < K; p++) 
//                         for (int q = 0; q < K; q++)
//                             if (h - p >= 0 && w-p >= 0 && h-p < H_out )
// }

__global__ void 
unroll_Kernel(int C, int H, int W, int K, float* X, float* X_unroll) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;  
    int H_out = H - K + 1;  
    int W_out = W - K + 1;  
    // Width of the unrolled input feature matrix  
    int W_unroll = H_out * W_out;  
    if (t < C * W_unroll) {  
        // Channel of the input feature map being collected by the thread  
        int c = t / W_unroll;  
        // Column index of the unrolled matrix to write a strip of  
        // input elements into (also, the linearized index of the output  
        // element for which the thread is collecting input elements)  
        int w_unroll = t % W_unroll;  
        // Horizontal and vertical indices of the output element
        int h_out = w_unroll / W_out;  
        int w_out = w_unroll % W_out;  
        // Starting row index for the unrolled matrix section for channel c  
        int w_base = c * K * K;  
        for (int p = 0; p < K; p++) {  
            for (int q = 0; q < K; q++) {  
                // Row index of the unrolled matrix for the thread to write  
                // the input element into for the current iteration  
                int h_unroll = w_base + p * K + q;  
                X_unroll[h_unroll, w_unroll] = X[c, h_out + p, w_out + q];  
            }  
        }  
    }  
}
