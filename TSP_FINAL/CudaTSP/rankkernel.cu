#include"../common.h"
#include<cuda_runtime.h>
#include"tspcommon.h"

__global__ void
rankPermutations(dist_idx_t* distances)
{
	__shared__ extern dist_idx_t dists[];
	unsigned int tid = threadIdx.x;
	int tc = blockDim.x;
	dists[tid] = distances[tid];
	dist_idx_t* current = dists;
	dist_idx_t* other = dists + tc;
	dist_idx_t*  tmp;

	for (unsigned int s = 0; 1 << s < blockDim.x; s ++)
	{
		if (tid % (1 << (s + 1)) == 0)
		{
			int step = 1 << s;
			int residx = tid, i1 = 0 , i2 = step;
			while(i1 < step && i2 < 2 * step)
			{
				if(current[tid + i1].distance < current[tid + i2].distance)
				{
					other[residx] = current[tid + i1];
					i1++;
				}else {
					other[residx] = current[tid + i2];
					i2 ++;
				}
				residx++;
			}
			for(;i1  < step; i1 ++, residx++)
				other[residx] = current[tid + i1];

			for(;i2  < 2 * step; i2 ++, residx++)
				other[residx] = current[tid + i2];

		}
		tmp = current;
		current = other;
		other = tmp;
		__syncthreads();
	}

	// printf("KERNEL: %lf\n", current[tid].distance);

	distances[tid] = current[tid];
}
