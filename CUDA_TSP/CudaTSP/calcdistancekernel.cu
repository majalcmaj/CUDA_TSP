#include<cuda_runtime.h>
#include"../common.h"
#define _USE_MATH_DEFINES
#include<math.h>
#include"tspcommon.h"

#define EARTH_R 6371

__device__ float deg_to_rad(float deg);
__device__ float calculateDistance(location_t* loc1, location_t* loc2);

__device__ float calculateDistance(location_t* loc1, location_t* loc2) {
	float dlat = deg_to_rad(loc1->latitude - loc2->latitude);
	float dlon = deg_to_rad(loc1->longitude - loc2->longitude);
	float a = sinf(dlat / 2) * sinf(dlat / 2)
			+ cosf(deg_to_rad(loc1->latitude)) * cosf(deg_to_rad(loc2->latitude))
					* sinf(dlon / 2) * sinf(dlon / 2);
	float c = 2 * atan2f(sqrtf(a), sqrtf(1 - a));
	float d = EARTH_R * c;
	return d;
}

__device__ float deg_to_rad(float deg) {
	return deg / 180.0 * M_PI;
}

__global__ void
calcDistanceForEachPermutation(location_t* locations, int* idx_holder, dist_idx_t * distances) {
	extern __shared__ double aggregators[]; // shared memory declaration
	int* indexes = &idx_holder[blockIdx.x * blockDim.x];
	unsigned int tid = threadIdx.x;

// each thread its element to 0 in shared memory
	aggregators[tid] = calculateDistance(&locations[indexes[tid]], &locations[indexes[(tid + 1) % blockDim.x]]);
	__syncthreads();
// reduction
	for (unsigned int s = 0; tid + (1 << s)  < blockDim.x; s ++) {
		if (!test_bit(tid, s)) {
			aggregators[tid] += aggregators[tid + (1 << s)];
		}
		__syncthreads();
	}
// transfer of the result to global memory
	if (tid == 0) {
		distances[blockIdx.x].distance = aggregators[0];
		distances[blockIdx.x].index = blockIdx.x;
	}
}
