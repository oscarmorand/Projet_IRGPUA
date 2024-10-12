#pragma once

#include "cuda_utils.cuh"

void inclusive_scan(raft::device_span<int> buffer, cudaStream_t stream);

void exclusive_scan(raft::device_span<int> buffer, cudaStream_t stream);

void get_new_size(raft::device_span<int> buffer, raft::device_span<int> new_size, cudaStream_t stream);