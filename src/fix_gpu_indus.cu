#include "cuda_utils.cuh"
#include "fix_gpu_indus.cuh"
#include "image.hh"


#include <cub/cub.cuh>

#include <raft/core/nvtx.hpp>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include "kernels/map_fix_pixels.cuh"
#include "kernels/scatter.cuh"

struct is_not_garbage {
    const int garbage_val = -27;
    
    __host__ __device__
    bool operator()(const int x) const {
        return (x != garbage_val);
    }
};

struct map_functor {
    const int m[4] = {1, -5, 3, -8};

    __host__ __device__
    int operator()(const int x, const int idx) const {
        return x + m[idx % 4];
    }
};

void fix_image_gpu_indus(rmm::device_uvector<int>& to_fix, unsigned long image_size, const raft::handle_t handle)
{
    // Build predicate vector

    raft::common::nvtx::range fix_image_gpu("Fix image GPU");

    const int actual_size = to_fix.size();
    rmm::device_uvector<int> predicate(actual_size, handle.get_stream());
    

    //copy tofix in buffer_copy
    thrust::device_vector<int> buffer_copy(to_fix.begin(), to_fix.end());
    raft::common::nvtx::push_range("Set predicate");
    thrust::transform(thrust::cuda::par.on(handle.get_stream()),
                      buffer_copy.begin(),
                      buffer_copy.end(),
                      predicate.begin(),
                      is_not_garbage());
    raft::common::nvtx::pop_range();

    //! Compute the exclusive sum of the predicate
    raft::common::nvtx::push_range("Exclusive scan");
    thrust::exclusive_scan(thrust::cuda::par.on(handle.get_stream()),
                           predicate.begin(),
                           predicate.end(),
                           predicate.begin());
    raft::common::nvtx::pop_range();

    // Scatter to the corresponding addresses
    raft::common::nvtx::push_range("Scatter");
    thrust::scatter_if(
        thrust::cuda::par.on(handle.get_stream()),
        buffer_copy.begin(),        // Beginning of the sequence of values to scatter. 
        buffer_copy.end(),          // End of the sequence of values to scatter. 
        predicate.begin(),     // Beginning of the sequence of output indices. 
        buffer_copy.begin(),        // Beginning of the sequence of predicate values. 
        to_fix.begin(),        // Beginning of the destination range. 
        is_not_garbage()       // Predicate to apply to the stencil values.
    );
    raft::common::nvtx::pop_range();

    // #2 Apply map to fix pixels
    raft::common::nvtx::push_range("Map fix pixels");
    thrust::counting_iterator<int> idx_begin(0);
    thrust::transform(thrust::cuda::par.on(handle.get_stream()),
                      to_fix.begin(),
                      to_fix.end(),
                      idx_begin,
                      to_fix.begin(),
                      map_functor());
    raft::common::nvtx::pop_range();

    // // #3 Histogram equalization

    raft::common::nvtx::push_range("Compute histogram");
    //! Does not work
    // void*    d_temp_storage = nullptr;
    // size_t   temp_storage_bytes = 0;
    // int* d_histo = nullptr;

    // cub::DeviceHistogram::HistogramEven(
    //     d_temp_storage, temp_storage_bytes,
    //     to_fix.data(), d_histo, 256,
    //     0, 256, image_size
    // );

    // CUDA_CHECK_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // cub::DeviceHistogram::HistogramEven(
    //     d_temp_storage, temp_storage_bytes,
    //     to_fix.data(), d_histo, 256,
    //     0, 256, image_size
    // );

    raft::common::nvtx::pop_range();

    // // Compute the inclusive sum scan of the histogram
    // raft::common::nvtx::push_range("Inclusive scan");
    // inclusive_scan(raft::device_span<int>(histo.data(), 256),
    //     handle.get_stream());
    // raft::common::nvtx::pop_range();

    
    // // Histogram equalization of the image
    // raft::common::nvtx::push_range("Equalize histogram");
    // equalize_histogram(raft::device_span<int>(to_fix.data(), image_size),
    //     raft::device_span<int>(histo.data(), 256),
    //     handle.get_stream());
    // raft::common::nvtx::pop_range();
}