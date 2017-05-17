#include"../common.h"
#include<cuda_runtime.h>
#include"tspcommon.h"

#define MUTATION_PROBABILITY 0.1

__device__ void
generateRandomMutations(int* indexes, int indexes_count, int tid, double random);

__global__ void
mutateGenomes(int* indexes, double* rands)
{
	int tid = threadIdx.x;
	int tid_g = blockIdx.x * blockDim.x + tid;
	int locations_count = blockDim.x;
	int* my_idxs = &indexes[blockIdx.x * locations_count];
	generateRandomMutations(my_idxs, locations_count, tid, rands[tid_g]);
}

__device__ void generateRandomMutations(int* indexes, int indexes_count, int tid, double random)
{
	int steps = binlog(indexes_count);
	for(int i = 0; i < steps ; i++)
	{
		// if conditions: first - randomness, second - checks if approprite bit on tid is off - then a step is made
		// thrird - if element to swap with is less than elem count.
		if(random <= MUTATION_PROBABILITY && !test_bit(tid, i) && (tid + (1 << i)) < indexes_count)
		{
			int tmp = indexes[tid];
			indexes[tid] = indexes[tid + (1 << i)];
			indexes[tid + (1 << i)] = tmp;
		}

		// with double precision this may be done 15 times - currently program
		// works for max 512 records. (only 9 loop iterations needed
		random = 10 * random;
		random -= (int)random;
		__syncthreads();
	}
}
