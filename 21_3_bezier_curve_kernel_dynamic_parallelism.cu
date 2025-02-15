/* 
    this file should be compiled with `-rdc=true`
 */


#include "utils.hpp"

constexpr int MAX_TESS_POINTS = 32;

// A structure containing all parameters needed to tessellate a Bezier line
struct BezierLine {
    float2 CP[3];
    float2* vertexPos;
    int nVertices;
};

__device__ float dist(float2 a, float2 b) {
    // printf("a.x = %f, a.y = %f, b.x = %f, b.y = %f\n", a.x, a.y, b.x, b.y);
    return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) *  (a.y - b.y));
}

__device__ float computeCurvature(BezierLine bLine) {
    float a = dist(bLine.CP[1], bLine.CP[0]);
    float b = dist(bLine.CP[1], bLine.CP[2]);
    float c = dist(bLine.CP[2], bLine.CP[0]);
    // printf("a = %f, b = %f, c = %f\n", a, b, c);
    float s = (a + b + c) / 2;
    float area = sqrtf(s * (s - a) * (s - b) * (s - c));
    return 4 * area / (a * b * c);
}

__global__ void computeBezierLine_child(int lidx, BezierLine* bLines, 
    int nTessPoints) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nTessPoints) {
        float u = (float)idx / (float)(nTessPoints - 1);
        float omu = 1.0f - u;
        float B3u[3];
        B3u[0] = omu * omu;
        B3u[1] = 2.0f*u*omu;
        B3u[2] = u*u;
        float2 position = {0, 0};
        for (int i = 0; i < 3; i++) {
            // position = position + B3u[i] * bLines[lidx].CP[i];
            position.x = position.x + B3u[i] * bLines[lidx].CP[i].x;
            position.y = position.y + B3u[i] * bLines[lidx].CP[i].y;
        }
        bLines[lidx].vertexPos[idx] = position;
    }
}

__global__ void computeBezierLines_parent(BezierLine *bLines, int nLines) {
    // Compute a unique index for each Bezier line
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lidx < nLines) {
        // Compute the curvation of the line
        float curvature = computeCurvature(bLines[lidx]);
        
        // From the curvature, compute the number of tessellation points
        bLines[lidx].nVertices = min(max((int)(curvature * 16.0f), 32), MAX_TESS_POINTS);
        printf("%f\n", curvature);
        cudaMalloc((void**)&bLines[lidx].vertexPos, 
            bLines[lidx].nVertices*sizeof(float2));
        
        // Call the child kernel to compute the tessellated points for each line
        computeBezierLine_child<<<ceilf((float)bLines[lidx].nVertices/32.0f), 32>>>
            (lidx, bLines, bLines[lidx].nVertices);
    }
}

__global__ void freeVertexMem(BezierLine *bLines, int nLines) {
    // Compute a unique for each Bezier line
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lidx < nLines) {
        cudaFree(bLines[lidx].vertexPos);
    }
}

void initializeBezierLines(UnifiedPtr<BezierLine> & bLines, int nLines) {
    for (int i = 0; i < nLines; i++) {
        // Initialize control points for the Bezier line
        bLines[i].CP[0] = {0.0f, 0.0f};
        bLines[i].CP[1] = {0.5f, static_cast<float>(i+1)};
        bLines[i].CP[2] = {1.0f, 0.0f};
        bLines[i].nVertices = 0;
    }
}

void printResults(UnifiedPtr<BezierLine> & bLines, int nLines) {
    for (int i = 0; i < nLines; i++) {
        std::cout << "Bezier Line " << i << ":\n";
        std::cout << "  nVertices: " << bLines[i].nVertices << "\n";
        // for (int j = 0; j < bLines[i].nVertices; j++) {
        //     std::cout << "  Vertex " << j << ": (" << bLines[i].vertexPos[j].x << ", " << bLines[i].vertexPos[j].y << ")\n";
        // }
    }
}

int main() {
    constexpr int nLines = 32;
    UnifiedPtr<BezierLine> bLines(nLines, true);
    initializeBezierLines(bLines, nLines);
    int threadsPerBlock = 32;
    int blocksPerGrid = (nLines + threadsPerBlock - 1) / threadsPerBlock;
    computeBezierLines_parent<<<blocksPerGrid, threadsPerBlock>>>(bLines.get(), nLines);
    cudaDeviceSynchronize();
    printResults(bLines, nLines);
    cudaDeviceSynchronize();
    freeVertexMem<<<blocksPerGrid, threadsPerBlock>>>(bLines.get(), nLines);
}