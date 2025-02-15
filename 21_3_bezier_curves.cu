/* 
    this file should be compiled with `-rdc=true`
 */


#include "utils.hpp"

constexpr int MAX_TESS_POINTS = 32;

// A structure containing all parameters needed to tessellate a Bezier line
// __host__ __device__
struct BezierLine {
    float2 CP[3];
    float2 vertexPos[MAX_TESS_POINTS];
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

__global__ void computeBezierLines(BezierLine *bLines, int nLines) {
    int bidx = blockIdx.x;
    if (bidx < nLines) {
        // Compute the curvature of the line
        float curvature = computeCurvature(bLines[bidx]);

        // From the curvature, compute the number of the tessellation points
        int nTessPoints = min(max((int)(curvature * 16.0f), 4), 32);
        bLines[bidx].nVertices = nTessPoints;
        // if (threadIdx.x == 0)
        // printf("block idx: %d, num vertices: %d, curvature: %f\n", bidx, bLines[bidx].nVertices, curvature);
        // Loop through vertices to be tessellated, incrementing by blockDim.x;
        for (int inc = 0; inc < nTessPoints; inc += blockDim.x) {
            int idx = inc + threadIdx.x;
            if (idx < nTessPoints) {
                float u = (float) idx / (float)(nTessPoints - 1);
                float omu = 1.0f - u;
                float B3u[3];
                B3u[0] = omu * omu;
                B3u[1] = 2.0f * u * omu;
                B3u[2] = u * u;
                float2 position = {0, 0};
                for (int i = 0; i < 3; i++) {
                    // position = position + B3u[i] * bLines[bidx].CP[i];
                    position.x = position.x + B3u[i] * bLines[bidx].CP[i].x;
                    position.y = position.y + B3u[i] * bLines[bidx].CP[i].y;
                }
                bLines[bidx].vertexPos[idx] = position;
            }
        }
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
        for (int j = 0; j < bLines[i].nVertices; j++) {
            std::cout << "  Vertex " << j << ": (" << bLines[i].vertexPos[j].x << ", " << bLines[i].vertexPos[j].y << ")\n";
        }
    }
}

int main() {
    constexpr int nLines = 4;
    // UnifiedPtr<BezierLine> bLines(nLines, true);
    // initializeBezierLines(bLines, nLines);
    // int threadsPerBlock = 32;
    // int blocksPerGrid = nLines;
    // computeBezierLines<<<blocksPerGrid, threadsPerBlock>>>(bLines.get(), nLines);
    // cudaDeviceSynchronize();
    // printResults(bLines, nLines);
    UnifiedPtr<int> a(nLines, true);
}