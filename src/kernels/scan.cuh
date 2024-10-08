#pragma once

#include "cuda_utils.cuh"

void scan(raft::device_span<int> buffer, cudaStream_t stream);