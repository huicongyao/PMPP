/* 
    this file should be compiled with `-rdc=true`
 */


#include "utils.hpp"

// Check the number of points and its depth
__device__ bool check_num_points_and_depth(Quadtree_node &node, Points *points, 
                                            int num_points, Parameters params) {
    if(params.depth >= params.max_depth || num_points <= params.min_points_per_node) {
        // Stop the recursion here. Make the points[0] cntains all the points
        if (params.point_selector == 1) {
            int it = node.points_begin(), end = node.points_end();
            for (it += threadIdx.x; it < end; it += blockDim.x) {
                if (it < end) 
                    points[0].set_point(it, points[1].get_point(it));
            }
        }
        return true;
    }
    return false;
}

// Count the number of points in each quadrant
__device__ void count_points_in_children(const Points &in_points, int *smem,
    int range_begin, int range_end, float2 center) {
    // Initialize shared memory
    if (threadIdx.x < 4) smem[threadIdx.x] = 0;
    __syncthreads();
    // Compute the number of points
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = in_points.get_point(iter);
        if (p.x < center.x && p.y >= center.y) 
            atomicAdd(&smem[0], 1);
        if (p.x >= center.x && p.y >= center.y) 
            atomicAdd(&smem[1], 1);
        if (p.x < center.x && p.y < center.y) 
            atomicAdd(&smem[2], 1);
        if (p.x >= center.x && p.y < center.y) 
            atomicAdd(&smem[3], 1);
    }
    __syncthreads();
}

// Scan quadrants' results to obtain reording offset
__device__ void scan_for_offsets(int node_points_begin, int *smem) {
    int *smem2 = &smem[4];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; i++)
            smem2[i] = i == 0 ? 0 : smem2[i - 1] + smem[i - 1]; // Sequential scan
        for (int i = 0; i < 4; i++)
            smem2[i] += node_points_begin; // Global offset
    }
    __syncthreads();
}

// Reorder points in order to group the points in each quadrant
__device__ void reorder_points (Points& out_points, const Points& in_points, int* smem,
    int range_begin, int range_end, float2 center)
{
    int* smem2 = &smem[4];
    // Reorder points
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        int dest;
        float2 p = in_points.get_point(iter); // Load the coordinates of the point
        if (p.x < center.x && p.y >= center.y)
            dest = atomicAdd(&smem2[0], 1); // Top-left point?
        else if (p.x >= center.x && p.y >= center.y)
            dest = atomicAdd(&smem2[1], 1); // Top-right point?
        else if (p.x < center.x && p.y < center.y)
            dest = atomicAdd(&smem2[2], 1); // Bottom-left point?
        else if (p.x >= center.x && p.y < center.y)
            dest = atomicAdd(&smem2[3], 1); // Bottom-right point?
        // Move point
        out_points.set_point(dest, p);
    }
    __syncthreads();
}


// Prepare children launch
__device__ void prepare_children(Quadtree_node* children, Quadtree_node& node,
    const Bounding_box& bbox, int* smem)
{
    int child_offset = 4 * node.id(); // The offsets of the children at their level

    // Set IDs
    children[child_offset + 0].set_id(4 * node.id() + 0);
    children[child_offset + 1].set_id(4 * node.id() + 4);
    children[child_offset + 2].set_id(4 * node.id() + 8);
    children[child_offset + 3].set_id(4 * node.id() + 12);

    // Points of the bounding-box
    const float2& p_min = bbox.get_min();
    const float2& p_max = bbox.get_max();

    // Set the bounding boxes of the children
    children[child_offset + 0].set_bounding_box(
        p_min.x, center.y, center.x, p_max.y);   // Top-left
    children[child_offset + 1].set_bounding_box(
        center.x, center.y, p_max.x, p_max.y);   // Top-right
    children[child_offset + 2].set_bounding_box(
        p_min.x, p_min.y, center.x, center.y);   // Bottom-left
    children[child_offset + 3].set_bounding_box(
        center.x, p_min.y, p_max.x, center.y);   // Bottom-right

    // Set the ranges of the children
    children[child_offset + 0].set_range(node.points_begin(), smem[4 + 0]);
    children[child_offset + 1].set_range(smem[4 + 0], smem[4 + 1]);
    children[child_offset + 2].set_range(smem[4 + 1], smem[4 + 2]);
    children[child_offset + 3].set_range(smem[4 + 2], smem[4 + 3]);
}

__global__ void build_quadtree_kernel
        (Quadtree_node *nodes, Points *points, Parameters params) {
    __shared__ int smem[8];

    // The current node in the quadtree
    Quadtree_node &node = nodes[blockIdx.x];
    node.set_id(node.id() + blockIdx.x);
    int num_points = node.num_points(); // The number of points in the node

    // Check the number of points and its depth
    bool exit = check_num_points_and_depth(node, points, num_points, params);
    if (exit) return;

    // Compute the center of the bounding box of the points
    const Bounding_box &bbox = node.bounding_box();
    float2 center;
    bbox.compute_center(center);

    // Range of points
    int range_begin = node.points_begin();
    int range_end = node.points_end();
    const Points &in_points = points[params.point_selector]; // Input points
    Points &out_points = points[(params.point_selector + 1) % 2];

    // Count the number of points in each child
    count_points_in_children(in_points, smem, range_begin, range_end, center);

    // Scan the quadrants' results to know the reordering offset
    scan_for_offsets(node.points_begin(), smem);

    // Move points 
    reorder_points(out_points, in_points, smem, range_begin, range_end, center);

    // Launch new blocks
    if (threadIdx.x == blockDim.x - 1) {
        // The children
        Quadtree_node *children = &nodes[params.num_nodes_at_this_level];

        // Prepare children launch
        prepare_children(children, node, bbox, smem);

        // Launch 4 children
        build_quadtree_kernel<<<4, blockDim.x, 8 * sizeof(int)>>>
            (children, points, Parameters(params, true));
    }
}
