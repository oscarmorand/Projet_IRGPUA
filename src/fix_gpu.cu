#include "cuda_utils.cuh"
#include "fix_gpu.cuh"
#include "image.hh"
#include "kernels/map_fix_pixels.cuh"
#include "kernels/set_predicate.cuh"
#include "kernels/scatter.cuh"
#include "kernels/scan.cuh"
#include "kernels/histogram.cuh"
#include "kernels/equalization.cuh"
#include "kernels/reduce.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <raft/core/nvtx.hpp>

//!!!!!!!!!!!!!!!!!!!!!11
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <string>

void write_uvector_to_file(rmm::device_uvector<int>& d_vector, std::string m = "", int size = 0) {
    auto filename = "outgpu.txt";
    // Create a thrust host vector to copy the device vector into
    std::ofstream file("outgpu.txt", std::ios::app);
    file << "New image data:\n";
    file << m << std::endl;
    thrust::host_vector<int> h_vector(d_vector.size());

    // Copy data from device to host
    thrust::copy(thrust::device_ptr<int>(d_vector.data()), 
                 thrust::device_ptr<int>(d_vector.data() + d_vector.size()), 
                 h_vector.begin());

    // Open file for writing
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write data to file
    if (size == 0) {
        for (int i = 0; i < h_vector.size(); i++) {
            file << h_vector[i] << " ";
        }
    } else {
        for (int i = size; i < d_vector.size(); i++) {
            file << h_vector[i] << " ";
        }
    }
    file << std::endl;
    // Close the file
    file.close();
}
//!!!!!!!!!!!!!!!!!!!!!11

void fix_image_gpu(rmm::device_uvector<int>& to_fix, unsigned long image_size, rmm::device_scalar<int>& total, const raft::handle_t handle)
{
    // Build predicate vector

    raft::common::nvtx::range fix_image_gpu("Fix image GPU");

    const int actual_size = to_fix.size();

    const int padded_size = (std::ceil(static_cast<float>(actual_size) / 64) * 64);

    // std::cout << "Actual size: " << actual_size << std::endl;
    // std::cout << "Padded size: " << padded_size << std::endl;

    rmm::device_uvector<int> predicate(padded_size, handle.get_stream());
    raft::common::nvtx::push_range("Set predicate");
    set_predicate(
        raft::device_span<int>(to_fix.data(), actual_size),
        raft::device_span<int>(predicate.data(), padded_size),
        -27,
        padded_size,
        handle.get_stream());
    raft::common::nvtx::pop_range();

    // Compute the exclusive sum of the predicate
    raft::common::nvtx::push_range("Exclusive scan");
    exclusive_scan(
        raft::device_span<int>(predicate.data(), padded_size),
        handle.get_stream());

    // rmm::device_uvector<int> predicate_scanned(padded_size, handle.get_stream());
    // thrust::exclusive_scan(thrust::cuda::par.on(handle.get_stream()), predicate.begin(), predicate.end(), predicate_scanned.begin());
    
    raft::common::nvtx::pop_range();
    

    // Scatter to the corresponding addresses
    raft::common::nvtx::push_range("Scatter");
    rmm::device_uvector<int> res_image_buffer(image_size, handle.get_stream());

    scatter(raft::device_span<int>(to_fix.data(), actual_size),
        // raft::device_span<int>(predicate_scanned.data(), padded_size),
        raft::device_span<int>(predicate.data(), padded_size),
        raft::device_span<int>(res_image_buffer.data(), image_size),
        -27,
        handle.get_stream());
    
    // thrust::scatter_if(thrust::cuda::par.on(handle.get_stream()), to_fix.begin(), to_fix.end(), predicate_scanned.begin(), predicate.begin(), res_image_buffer.begin());

    raft::common::nvtx::pop_range();

    // #2 Apply map to fix pixels
    raft::common::nvtx::push_range("Map fix pixels");
    map_fix_pixels(raft::device_span<int>(res_image_buffer.data(), image_size),
        handle.get_stream());
    raft::common::nvtx::pop_range();

    // #3 Histogram equalization
    rmm::device_uvector<int> histo(256, handle.get_stream());
    raft::common::nvtx::push_range("Compute histogram");
    compute_histogram(raft::device_span<int>(res_image_buffer.data(), image_size),
        raft::device_span<int>(histo.data(), 256),
        handle.get_stream());
    raft::common::nvtx::pop_range();

    // Compute the inclusive sum scan of the histogram
    raft::common::nvtx::push_range("Inclusive scan");
    inclusive_scan(raft::device_span<int>(histo.data(), 256),
        handle.get_stream());
    // thrust::inclusive_scan(thrust::cuda::par.on(handle.get_stream()), histo.begin(), histo.end(), histo.begin());
    raft::common::nvtx::pop_range();
    
    // // Histogram equalization of the image
    raft::common::nvtx::push_range("Equalize histogram");
    equalize_histogram(raft::device_span<int>(res_image_buffer.data(), image_size),
        raft::device_span<int>(histo.data(), 256),
        handle.get_stream());
    raft::common::nvtx::pop_range();

    // Compute the sum of the image to sort it later
    raft::common::nvtx::push_range("Reduce sum");
    reduce_sum(raft::device_span<int>(res_image_buffer.data(), image_size),
        raft::device_span<int>(total.data(), 1),
        handle.get_stream());
    raft::common::nvtx::pop_range();

    raft::copy(to_fix.data(), res_image_buffer.data(), image_size, handle.get_stream()); // FIXME passer par ref direct pour pas avoir de raft copy à faire ?
}