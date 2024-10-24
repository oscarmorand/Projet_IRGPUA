#include "reduce.cuh"

__inline__ __device__
int warp_reduce(int val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(~0, val, offset);
    }
    return val;
}

template <typename T, int BLOCK_SIZE>
__global__
void reduce_sum_kernel(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    const unsigned int tid = threadIdx.x;
    unsigned int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    
    int sum = 0;
    while (id < buffer.size() / 4) {
        int4 val = reinterpret_cast<const int4*>(buffer.data())[id];
        sum += val.x + val.y + val.z + val.w;
        id += BLOCK_SIZE * gridDim.x;
    }
    id *= 4;
    while (id < buffer.size()) {
        sum += buffer[id];
        id += 1;
    }

    sum = warp_reduce(sum);

    if (tid % 32 == 0) atomicAdd(total.data(), sum);
}

void reduce_sum(raft::device_span<int> buffer,
                raft::device_span<int> total,
                cudaStream_t stream)
{
    constexpr int blocksize = 64;
    const int gridsize = (((buffer.size() + blocksize - 1) / blocksize) + 1) / 2;

	reduce_sum_kernel<int, blocksize><<<gridsize, blocksize, blocksize * sizeof(int), stream>>>(buffer, total);

    CUDA_CHECK_ERROR(cudaGetLastError());
}