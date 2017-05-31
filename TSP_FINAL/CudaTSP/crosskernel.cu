#include"../common.h"
#include"tspcommon.h"
#include<cuda_runtime.h>

#define CHUNK_SIZE(original_size) (int)(original_size >> 1)

__global__ void
crossoverTheFittest(dist_idx_t* distances, int* indexes, double* rands)  // Todo calculate needed random arr size, shared size, blocks_count = len(distances) / 2
{
	extern __shared__ int new_chunk[];

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int locations_count = blockDim.x;
	int blocks_count = gridDim.x;

	int* old_chunk = &new_chunk[locations_count];

	int out_idx = distances[blocks_count + bid].index * locations_count + tid;
	int cross_with = (bid + 1 + (int)(rands[bid] * (blocks_count - 1 ))) % blocks_count; // now both blocks get a different indices to do crossover with

	rands += blocks_count; // set new start of rands array
	int* my_idxs = &indexes[distances[bid].index * locations_count];
	int* cross_with_idxs = &indexes[distances[cross_with].index * locations_count];

	int chunk_size = CHUNK_SIZE(locations_count);
	int crossover_chunk_start = (int)(rands[bid]  * (locations_count - chunk_size ));
	int crossover_chunk_end = crossover_chunk_start + chunk_size;
	int my_idx;
	if(tid >= crossover_chunk_start && tid < crossover_chunk_end)
	{
		new_chunk[tid - crossover_chunk_start] = my_idx = cross_with_idxs[tid];
		old_chunk[tid - crossover_chunk_start] = my_idxs[tid];
	}
	__syncthreads();
	if(! (tid >= crossover_chunk_start && tid < crossover_chunk_end))
	{
		my_idx = my_idxs[tid];
restart_loop:
		for(int i = 0 ; i < chunk_size ; i++)
		{
			if(new_chunk[i] == my_idx)
			{
				my_idx = old_chunk[i];
				goto restart_loop;
			}
		}
	}
	indexes[out_idx] = my_idx;
}


