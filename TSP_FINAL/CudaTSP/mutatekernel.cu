#include"../common.h"
#include<cuda_runtime.h>
#include"tspcommon.h"

#define MUTATION_PROBABILITY 0.003
#define MUT_PROB_GROWTH_FACTOR 1.71
__device__ void
generateRandomMutations(int* indexes, int indexes_count, int tid, double random);

__global__ void
mutateGenomes(dist_idx_t* sorted_distances, int* indexes, double* rands)
{
	if(sorted_distances[0].index != blockIdx.x)
	{
		int tid = threadIdx.x;
		int tid_g = blockIdx.x * blockDim.x + tid;
		int locations_count = blockDim.x;
		int* my_idxs = &indexes[blockIdx.x * locations_count];
		generateRandomMutations(my_idxs, locations_count, tid, rands[tid_g]);
	}
}

#include<cuda_runtime.h>
__device__ int binlog1(int a)
{
	// Opt p. Draszawki
	int log = 0;
	while(a >>= 1) log++;
	return log;
}

__device__ void generateRandomMutations(int* indexes, int indexes_count, int tid, double random)
{

	int steps = binlog1(indexes_count);
	double mutation_probability = MUTATION_PROBABILITY;

	for(int i = steps -1 ; i  >= 0  ; i--)
	{
		// if conditions: first - randomness, second - checks if approprite bit on tid is off - then a step is made
		// thrird - if element to swap with is less than elem count.
		if(random <= mutation_probability && !test_bit(tid, i) && (tid + (1 << i)) < indexes_count)
		{
			int tmp = indexes[tid];
			indexes[tid] = indexes[tid + (1 << i)];
			indexes[tid + (1 << i)] = tmp;
			mutation_probability *= MUT_PROB_GROWTH_FACTOR;
		}

		// with double precision this may be done 15 times - currently program
		// works for max 512 records. (only 9 loop iterations needed
		random = 117 * random;
		random -= (int)random;
		__syncthreads();
	}
}
