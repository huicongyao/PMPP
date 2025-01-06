#pragma once
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include <atomic>

__device__ int _ceil(int x, int y) {
    return (x + y - 1) / y;
}

template <typename  T>
struct ControlBlock {
    T* ptr;
    bool isCuda;
    size_t size;
    std::atomic<int> ref_count;

    ControlBlock(T *p, bool cuda, size_t s)
        : ptr(p), isCuda(cuda), size(s), ref_count(1) {}
};


// Custom template pointer class
template <typename T>
class UnifiedPtr {
private:
    ControlBlock<T> *control;

    void release() {
        if (--control->ref_count == 0) {
            if (control->isCuda) {
                cudaFree(control->ptr);
            } else {
                delete[] control->ptr;
            }
            delete control;
        }
        control = nullptr;
    }

public:

    // Constructor: Initializes memory based on whether CUDA memory is being used
    __host__
    UnifiedPtr(size_t _size, bool useCuda = false) : control(nullptr) {
        T* p = nullptr;
        if (useCuda) {
            cudaError_t err = cudaMallocManaged(&p, _size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMallocManaged failed: " + std::string(cudaGetErrorString(err)));
            }
        } else {
            p = new T[_size];
        }
        control = new ControlBlock<T>(p, useCuda, _size);
    }

    // Constructor: Initializes memory based on whether CUDA memory is being used
    __host__
    UnifiedPtr(size_t _size, T val, bool useCuda = false) : control(nullptr) {
        T* p = nullptr;
        if (useCuda) {
            cudaError_t err = cudaMallocManaged(&p, _size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMallocManaged failed: " + std::string(cudaGetErrorString(err)));
            }
        } else {
            p = new T[_size];
        }
        // 初始化数据
        for (size_t i = 0; i < _size; ++i) {
            p[i] = val;
        }
        control = new ControlBlock<T>(p, useCuda, _size);
    }

    // Constructor using an initializer_list
    __host__
    UnifiedPtr(std::initializer_list<T> initList, bool useCuda = false) : control(nullptr){
        T* p = nullptr;
        if (useCuda) {
            cudaError_t err = cudaMallocManaged(&p, initList.size() * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMallocManaged failed: " + std::string(cudaGetErrorString(err)));
            }
        } else {
            p = new T[initList.size()];
        }
        // 复制数据
        size_t i = 0;
        for (const auto& value : initList) {
            p[i++] = value;
        }
        control = new ControlBlock<T>(p, useCuda, initList.size());
    }

    UnifiedPtr(const UnifiedPtr& other) : control(other.control) {
        if (control) {
            ++control->ref_count;
        }
    }

    UnifiedPtr& operator=(const UnifiedPtr& other) {
        if (this != &other) {
            // 释放当前资源
            release();
            // 复制新的控制块
            control = other.control;
            if (control) {
                ++control->ref_count;
            }
        }
        return *this;
    }

    UnifiedPtr(UnifiedPtr&& other) noexcept : control(other.control) {
        other.control = nullptr;
    }

    UnifiedPtr& operator=(UnifiedPtr&& other) noexcept {
        if (this != &other) {
            // 释放当前资源
            release();
            // 转移控制块指针
            control = other.control;
            other.control = nullptr;
        }
        return *this;
    }


    // Destructor: Releases resources
    ~UnifiedPtr() {
        release();
    }

    // Overloaded * operator to support access like a raw pointer
    __host__ __device__
    T& operator*() const {
        return *(control->ptr);
    }

    // Overloaded -> operator to support access like a raw pointer
    __host__ __device__
    T* operator->() const {
        return control->ptr;
    }

    // Overloaded [] operator to support index-based access
    __host__ __device__
    T& operator[](size_t index) const {
        assert(index < control->size && "Index out of range");
        return control->ptr[index];
    }

    // Interface to return the internal raw pointer
    __host__ __device__
    T* get() const {
        return control ? control->ptr : nullptr;
    }

    // Returns the number of elements
    __host__ __device__
    size_t size() const {
        return control ? control->size : 0;
    }

    int use_count() const {
        return control ? control->ref_count.load() : 0;
    }

    bool unique() const {
        return use_count() == 1;
    }
};


// A21.1 Support code for quadtree example


// A structure of 2D points
class Points {
    float *m_x;
    float *m_y;

public:
    // constructor
    __host__ __device__ Points() : m_x(NULL), m_y(NULL) {}

    // Constructor
    __host__ __device__ Points(float *x, float *y) : m_x(x), m_y(y) {}

    // Get a point
    __host__ __device__ __forceinline__ float2 get_point(int idx) const {
        return make_float2(m_x[idx], m_y[idx]);
    }

    // Set a point
    __host__ __device__ __forceinline__ void set_point(int idx, const float2 &p) {
        m_x[idx] = p.x;
        m_y[idx] = p.y;
    }

    // Set the pointers
    __host__ __device__ __forceinline__ void set(float *x, float *y) {
        m_x = x;
        m_y = y;
    }
};

// A 2D bounding box
class Bounding_box {
    // Extreme points of the bounding box
    float2 m_p_min;
    float2 m_p_max;

public:
    // Constructor. Create a unit box
    __host__ __device__ Bounding_box() {
        m_p_min = make_float2(0.0f, 0.0f);
        m_p_max = make_float2(1.0f, 1.0f);
    }

    // Compute the center of the bounding-box
    __host__ __device__ void compute_center(float2 &center) const {
        center.x = 0.5f * (m_p_min.x + m_p_max.x);
        center.y = 0.5f * (m_p_min.y + m_p_max.y);
    }

    // The points of the box
    __host__ __device__ __forceinline__ const float2 &get_max() const {
        return m_p_max;
    }

    __host__ __device__ __forceinline__ const float2 &get_min() const {
        return m_p_min;
    }

    // Does a box contain a point
    __host__ __device__ bool contains(const float2 &p) const {
        return p.x >= m_p_min.x && p.x < m_p_max.x && p.y >= m_p_min.y && p.y < m_p_max.y;
    }

    // Define the bounding box
    __host__ __device__ void set(float min_x, float min_y, float max_x, float max_y) {
        m_p_min.x = min_x;
        m_p_min.y = min_y;
        m_p_max.x = max_x;
        m_p_max.y = max_y;
    }
};

// A node of a quadree
class Quadtree_node {
    // The identifier of the node
    int m_id;
    // The bounding box of the tree;
    Bounding_box m_bounding_box;
    // The range of points
    int m_begin, m_end;

public:
    // Constructor
    __host__ __device__ Quadtree_node() : m_id(0), m_begin(0), m_end(0) {}

    // The ID of a node at its level
    __host__ __device__ int id() const {
        return m_id;
    }

    // The ID of a node at its level
    __host__ __device__ void set_id(int new_id) {
        m_id = new_id;
    }

    // The bounding box
    __host__ __device__ __forceinline__ const Bounding_box &bounding_box() const {
        return m_bounding_box;
    }

    // Set the bounding box
    __host__ __device__ __forceinline__ void set_bounding_box(float min_x,
                                                              float min_y, float max_x, float max_y) {
        m_bounding_box.set(min_x, min_y, max_x, max_y);
    }

    // The number of points in the tree
    __host__ __device__ __forceinline__ int num_points() const {
        return m_end - m_begin;
    }

    // The range of points in the tree
    __host__ __device__ __forceinline__ int points_begin() const {
        return m_begin;
    }

    __host__ __device__ __forceinline__ int points_end() const {
        return m_end;
    }

    // Define the range for that node
    __host__ __device__ __forceinline__ void set_range(int begin, int end) {
        m_begin = begin;
        m_end = end;
    }
};

// Algorithm parameters
struct Parameters {
    // Choose the right set of points to use as in/out
    int point_selector;
    // The number of nodes at a given level (2^k for level k)
    int num_nodes_at_this_level;
    // The recursion depth
    int depth;
    // The max value for depth
    const int max_depth;
    // The minimum number of points in a node to stop recursion
    const int min_points_per_node;

    // Constructor set to default values.
    __host__ __device__ Parameters(int max_depth, int min_points_per_node) :
            point_selector(0),
            num_nodes_at_this_level(1),
            depth(0),
            max_depth(max_depth),
            min_points_per_node(min_points_per_node) {}

    // Copy constructor. Changes the value for next iteration
    __host__ __device__ Parameters(const Parameters & params, bool) :
            point_selector((params.point_selector + 1) % 2),
            num_nodes_at_this_level(4 * params.num_nodes_at_this_level),
            depth(params.depth+1),
            max_depth(params.max_depth),
            min_points_per_node(params.min_points_per_node) {}
};