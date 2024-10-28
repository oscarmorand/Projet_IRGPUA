#pragma once

#include <raft/core/device_span.hpp>

void set_zeros(raft::device_span<int> buffer, int block_size);