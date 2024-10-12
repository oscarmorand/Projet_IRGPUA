#pragma once

#include "cuda_utils.cuh"

void equalize_histogram(raft::device_span<int> buffer, raft::device_span<int> histo, cudaStream_t stream);