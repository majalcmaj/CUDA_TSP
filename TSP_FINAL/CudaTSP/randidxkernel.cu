#include<stdio.h>
#include "tspcommon.h"
#include<cuda_runtime.h>

__device__ void generateIndexPermutations(int* indexes, int indexes_count, int tid, double rand);

__global__ void
populateMemory(int* idx_holder, int locations_count, double* rands)
{
	extern __shared__ int indexes[];
	int tid = threadIdx.x;
	int tid_g = blockIdx.x * blockDim.x + tid;
	if(tid < locations_count) {
		indexes[tid] = tid;
	}
	generateIndexPermutations(indexes, locations_count, tid, rands[tid_g]);
	int* indexes_d = &idx_holder[blockIdx.x * locations_count];
	if(tid < locations_count) {
		indexes_d[tid] = indexes[tid];
	}
}

__device__ int binlog(int bits)
{
    int log = 0;
    if( ( bits & 0xffff0000 ) != 0 ) { bits >>= 16; log = 16; }
    if( bits >= 256 ) { bits >>= 8; log += 8; }
    if( bits >= 16  ) { bits >>= 4; log += 4; }
    if( bits >= 4   ) { bits >>= 2; log += 2; }
    return log + ( bits >> 1 );
}

__device__ void generateIndexPermutations(int* indexes, int indexes_count, int tid, double rand)
{
	unsigned int random = (unsigned int)(rand * 0xFFFFFFFFu);

	int steps = binlog(indexes_count);
	for(int i = 0; i < steps ; i++)
	{
		// if conditions: first - randomness, second - checks if approprite bit on tid is off - then a step is made
		// thrird - if element to swap with is less than elem count.
		if(!test_bit(random, i) && !test_bit(tid, i) && (tid + (1 << i)) < indexes_count)
		{
			int tmp = indexes[tid];
			indexes[tid] = indexes[tid + (1 << i)];
			indexes[tid + (1 << i)] = tmp;
		}
		__syncthreads();
	}
}
