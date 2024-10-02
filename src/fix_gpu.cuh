#pragma once

#include <raft/core/handle.hpp>

#include "image.hh"

void fix_image_gpu(Image& to_fix, const raft::handle_t handle);