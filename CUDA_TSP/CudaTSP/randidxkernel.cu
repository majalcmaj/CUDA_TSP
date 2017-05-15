#include<cuda_runtime.h>
#include<stdio.h>
#include "tspcommon.h"
__device__ int binlog(int a);
__device__ void generateIndexes(int* indexes, int tid, int locations_count);
__device__ void generateIndexPermutations(int* indexes, int indexes_count, int tid, double rand);

__device__ int binlog(int a)
{
	// Opt p. Draszawki
	int log = 0;
	while(a >>= 1) log++;
	return log;
}

__device__ void generateIndexes(int* indexes, int tid, int locations_count)
{
	if(tid < locations_count) {
		indexes[tid] = tid;
	}
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

__global__ void
populateMemory(int* idx_holder, int locations_count, double* rands)
{
	int tid = threadIdx.x;
	int tid_g = blockIdx.x * blockDim.x + tid;
	generateIndexes(&idx_holder[blockIdx.x * locations_count], tid, locations_count);
	int* indexes = &idx_holder[blockIdx.x * locations_count];
	generateIndexPermutations(indexes, locations_count, tid, rands[tid_g]);
}
