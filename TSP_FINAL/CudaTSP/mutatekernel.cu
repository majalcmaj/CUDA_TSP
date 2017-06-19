#include"../common.h"
#include<cuda_runtime.h>
#include"tspcommon.h"

#define MUTATOR_1_PROB 0.25
#define MUTATOR_2_PROB 0.4

#define MUTATION_PROBABILITY_MUT1 0.004
#define MUT_PROB_GROWTH_FACTOR_MUT1 1.71

#define MUTATION_PROBABILITY_MUT2 0.05
#define MAX_MUTATIONS_MUT2 5
#define MAX_CHUNK_SIZE_MUT2 16

#define MUTATION_PROBABILITY_MUT3 0.1
#define MAX_MUTATIONS_MUT3 1
#define MAX_CHUNK_SIZE_MUT3(count) count / 2
__device__ void
generateRandomMutations1(int* indexes, int indexes_count, int tid, double random);

__device__ void
generateRandomMutations2(int* indexes, int indexes_count, int tid, double* random);

__device__ void
generateRandomMutations3(int* indexes, int indexes_count, int tid, double* random);

__global__ void
mutateGenomes(dist_idx_t* sorted_distances, int* all_indexes, double* rands)
{
	if(sorted_distances[0].index != blockIdx.x)
	{
		extern __shared__ int indexes[];
		int tid = threadIdx.x;
		int tid_g = blockIdx.x * blockDim.x + tid;
		int locations_count = blockDim.x;
		int* my_idxs = &all_indexes[blockIdx.x * locations_count];
		double decision_rand = *(rands++);
		indexes[tid] = my_idxs[tid];
		if(decision_rand < MUTATOR_1_PROB) {
			generateRandomMutations1(indexes, locations_count, tid, rands[tid_g]);
		}else if(decision_rand < MUTATOR_1_PROB + MUTATOR_2_PROB) {
			generateRandomMutations2(indexes, locations_count, tid, rands);
		}else {
			generateRandomMutations3(indexes, locations_count, tid, rands);
		}
		my_idxs[tid] = indexes[tid];
	}
}

__device__ int binlog1(int bits)
{
    int log = 0;
    if( ( bits & 0xffff0000 ) != 0 ) { bits >>= 16; log = 16; }
    if( bits >= 256 ) { bits >>= 8; log += 8; }
    if( bits >= 16  ) { bits >>= 4; log += 4; }
    if( bits >= 4   ) { bits >>= 2; log += 2; }
    return log + ( bits >> 1 );
}

__device__ void generateRandomMutations1(int* indexes, int indexes_count, int tid, double random)
{

	int steps = binlog1(indexes_count);
	double mutation_probability = MUTATION_PROBABILITY_MUT1;

	for(int i = steps -1 ; i  >= 0  ; i--)
	{
		// if conditions: first - randomness, second - checks if approprite bit on tid is off - then a step is made
		// thrird - if element to swap with is less than elem count.
		if(random <= mutation_probability && !test_bit(tid, i) && (tid + (1 << i)) < indexes_count)
		{
			int tmp = indexes[tid];
			indexes[tid] = indexes[tid + (1 << i)];
			indexes[tid + (1 << i)] = tmp;
			mutation_probability *= MUT_PROB_GROWTH_FACTOR_MUT1;
		}

		// with double precision this may be done 15 times - currently program
		// works for max 512 records. (only 9 loop iterations needed
		random = 117 * random;
		random -= (int)random;
		__syncthreads();
	}
}

__device__ void
generateRandomMutations2(int* indexes, int indexes_count, int tid, double* random)
{
	for(int i = 0 ; i < MAX_MUTATIONS_MUT2 ; i ++)
	{
		if(*(random++) < MUTATION_PROBABILITY_MUT2)
		{
			int chunk_size = *(random++) * MAX_CHUNK_SIZE_MUT2;
			int chunk_start = *(random++) * (indexes_count - chunk_size);
			int other_start = *(random++) * (indexes_count - chunk_size);
			if(chunk_start > other_start) { // other should start always later
				int tmp = chunk_start;
				chunk_start = other_start;
				other_start = tmp;
			}
			if(chunk_start + chunk_size > other_start) // if chunks are overlapping - take only the unique parts.
			{
				int old_other_start = other_start;
				other_start = chunk_start + chunk_size;
				chunk_size = old_other_start - chunk_start;
			}
			int idx, other;
			if(tid < chunk_size)
			{
				idx = indexes[chunk_start + tid];
				other = indexes[other_start + tid];
			}
			__syncthreads();
			if(tid < chunk_size)
			{
				indexes[chunk_start + tid] = other;
				indexes[other_start + tid] = idx;
			}
			__syncthreads();
		}
	}
}

// reverse
__device__ void
generateRandomMutations3(int* indexes, int indexes_count, int tid, double* random)
{
	for(int i = 0 ; i < MAX_MUTATIONS_MUT3 ; i ++)
	{
		if(*(random++) < MUTATION_PROBABILITY_MUT3)
		{
			int chunk_size = *(random++) * MAX_CHUNK_SIZE_MUT3(indexes_count);
			int chunk_start = *(random++) * (indexes_count - chunk_size);

			if(tid < chunk_size / 2)
			{
				int tmp = indexes[chunk_start + tid];
				indexes[chunk_start + tid] = indexes[chunk_start + chunk_size - tid -1];
				indexes[chunk_start + chunk_size - tid -1] = tmp;
			}
			__syncthreads();
		}
	}
}
