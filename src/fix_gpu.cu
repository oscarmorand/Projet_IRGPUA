#include "cuda_utils.cuh"
#include "fix_gpu.cuh"
#include "image.hh"
#include "kernels/map_fix_pixels.cuh"
#include "kernels/set_predicate.cuh"
#include "kernels/scatter.cuh"
#include "kernels/scan.cuh"
#include "kernels/histogram.cuh"
#include "kernels/equalization.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

void fix_image_gpu(rmm::device_uvector<int>& to_fix, unsigned long image_size, const raft::handle_t handle)
{
    // Build predicate vector

    rmm::device_uvector<int> predicate(image_size, handle.get_stream());

    set_predicate(
        raft::device_span<int>(to_fix.data(), image_size),
        raft::device_span<int>(predicate.data(), image_size),
        -27,
        handle.get_stream());

    // Compute the exclusive sum of the predicate

    exclusive_scan(
        raft::device_span<int>(predicate.data(), image_size),
        handle.get_stream());

    // Scatter to the corresponding addresses

    scatter(raft::device_span<int>(to_fix.data(), image_size),
        raft::device_span<int>(predicate.data(), image_size),
        27,
        handle.get_stream());

    // #2 Apply map to fix pixels

    map_fix_pixels(raft::device_span<int>(to_fix.data(), image_size),
        handle.get_stream());

    // #3 Histogram equalization

    rmm::device_uvector<int> histo(256, handle.get_stream());

    compute_histogram(raft::device_span<int>(to_fix.data(), image_size),
        raft::device_span<int>(histo.data(), 256),
        handle.get_stream());

    // Compute the inclusive sum scan of the histogram

    inclusive_scan(raft::device_span<int>(histo.data(), 256),
        handle.get_stream());

    // Histogram equalization of the image

    equalize_histogram(raft::device_span<int>(to_fix.data(), image_size),
        raft::device_span<int>(histo.data(), 256),
        handle.get_stream());
}