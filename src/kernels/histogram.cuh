#pragma once

#include "cuda_utils.cuh"

void compute_histogram(raft::device_span<int> buffer, raft::device_span<int> histo, cudaStream_t stream);