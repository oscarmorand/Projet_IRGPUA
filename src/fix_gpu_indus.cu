#include "cuda_utils.cuh"
#include "fix_gpu_indus.cuh"
#include "image.hh"

#include <cub/cub.cuh>

#include <raft/core/nvtx.hpp>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/find.h>

#include "kernels/map_fix_pixels.cuh"
#include "kernels/scatter.cuh"

struct is_not_garbage
{
    const int garbage_val = -27;

    __device__ bool operator()(const int x) const
    {
        return (x != garbage_val);
    }
};

struct map_functor
{
    const int m[4] = {1, -5, 3, -8};

    __device__ int operator()(const int x, const int idx) const
    {
        return x + m[idx % 4];
    }
};

struct histogram_equalization
{
    raft::device_span<int> d_histo_;
    unsigned long image_size_;
    int* cdf_min_;

    histogram_equalization(raft::device_span<int> d_histo, unsigned long image_size, int* cdf_min)
        : d_histo_(d_histo)
        , image_size_(image_size)
        , cdf_min_(cdf_min)
    {}

    __device__ int operator()(const int value) const
    {
        return static_cast<int>(((d_histo_[value] - *cdf_min_) / static_cast<float>(image_size_ - *cdf_min_)) * 255.0f);
    }
};

void fix_image_gpu_indus(rmm::device_uvector<int> &to_fix, unsigned long image_size, const raft::handle_t handle)
{
    // Build predicate vector

    raft::common::nvtx::range fix_image_gpu("Fix image GPU");

    const int actual_size = to_fix.size();

    // replace hand made compact with copy_if ?

    raft::common::nvtx::push_range("Set predicate");
    rmm::device_uvector<int> predicate(actual_size, handle.get_stream());
    thrust::transform(thrust::cuda::par.on(handle.get_stream()),
                      to_fix.begin(),
                      to_fix.end(),
                      predicate.begin(),
                      is_not_garbage());
    raft::common::nvtx::pop_range();

    //! Compute the exclusive sum of the predicate
    raft::common::nvtx::push_range("Exclusive scan");
    rmm::device_uvector<int> predicate_indices(actual_size, handle.get_stream());
    thrust::exclusive_scan(thrust::cuda::par.on(handle.get_stream()),
                           predicate.begin(),
                           predicate.end(),
                           predicate_indices.begin());
    raft::common::nvtx::pop_range();

    // Scatter to the corresponding addresses
    raft::common::nvtx::push_range("Scatter");
    rmm::device_uvector<int> fixed_image(image_size, handle.get_stream());
    thrust::scatter_if(
        thrust::cuda::par.on(handle.get_stream()),
        to_fix.begin(), // Beginning of the sequence of values to scatter.
        to_fix.end(),   // End of the sequence of values to scatter.
        predicate_indices.begin(),   // Beginning of the sequence of output indices.
        predicate.begin(), // Beginning of the sequence of predicate values.
        fixed_image.begin()      // Beginning of the destination range.
    );
    raft::common::nvtx::pop_range();

    // #2 Apply map to fix pixels
    raft::common::nvtx::push_range("Map fix pixels");
    thrust::counting_iterator<int> idx_begin(0);
    thrust::transform(thrust::cuda::par.on(handle.get_stream()),
                      fixed_image.begin(),
                      fixed_image.end(),
                      idx_begin,
                      fixed_image.begin(),
                      map_functor());
    raft::common::nvtx::pop_range();

    // #3 Histogram equalization

    const int HISTO_SIZE = 256;

    raft::common::nvtx::push_range("Compute histogram");
    size_t temp_storage_bytes = 0;

    rmm::device_uvector<int> d_histo(HISTO_SIZE, handle.get_stream());
    cub::DeviceHistogram::HistogramEven(
        nullptr, temp_storage_bytes,
        fixed_image.data(), d_histo.data(), HISTO_SIZE + 1,
        0, HISTO_SIZE, image_size, handle.get_stream());

    rmm::device_uvector<int> d_temp_storage(temp_storage_bytes / sizeof(int), handle.get_stream());

    cub::DeviceHistogram::HistogramEven(
        d_temp_storage.data(), temp_storage_bytes,
        fixed_image.data(), d_histo.data(), HISTO_SIZE + 1,
        0, HISTO_SIZE, image_size, handle.get_stream());
    raft::common::nvtx::pop_range();

    // Compute the inclusive sum scan of the histogram
    raft::common::nvtx::push_range("Inclusive scan");
    thrust::inclusive_scan(thrust::cuda::par.on(handle.get_stream()),
                           d_histo.begin(),
                           d_histo.end(),
                           d_histo.begin());
    raft::common::nvtx::pop_range();

    // Histogram equalization of the image
    raft::common::nvtx::push_range("Equalize histogram");
    auto found_iter = thrust::find_if(thrust::cuda::par.on(handle.get_stream()), 
                                      d_histo.begin(),
                                      d_histo.end(),
                                      [] __device__ (int x) { return x != 0; });
    int* cdf_min = thrust::raw_pointer_cast(&(*found_iter));

    histogram_equalization histo_equal(raft::device_span<int>(d_histo.data(), HISTO_SIZE), image_size, cdf_min);

    thrust::transform(thrust::cuda::par.on(handle.get_stream()),
                      fixed_image.begin(),
                      fixed_image.end(),
                      fixed_image.begin(),
                      histo_equal);
    raft::common::nvtx::pop_range();

    raft::copy(to_fix.data(), fixed_image.data(), image_size, handle.get_stream());

    raft::common::nvtx::pop_range();
}