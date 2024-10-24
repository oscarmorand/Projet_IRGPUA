#pragma once

#include "cuda_utils.cuh"

void reduce_sum(raft::device_span<int> buffer, raft::device_span<int> total, cudaStream_t stream);