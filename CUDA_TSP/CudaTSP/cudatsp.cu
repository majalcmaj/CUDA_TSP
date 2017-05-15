#include"cudatsp.h"
#include"tspcommon.h"
#include <stdio.h>
// CUDA runtime
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include<curand.h>

#include<time.h>

#define BLOCKS_COUNT 16

// DEBUG
#include"../DataIO/dataio.h"

__global__ void
calcDistanceForEachPermutation(location_t* locations, int* idx_holder, dist_idx_t* distances);

__global__ void
populateMemory(int* idx_holder, int locations_count, double* rands);

__global__ void
rankPermutations(dist_idx_t* distances);

int run_cuda_tsp(location_t* locations, int locations_count)
{
	double* rands;
	int* idx_holder;
	dim3 gridDim(BLOCKS_COUNT);
	dim3 blockDim(locations_count);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	CudaCheckError();
	CudaSafeCall(cudaMalloc(&rands, sizeof(double) * locations_count * BLOCKS_COUNT));
	curandGenerateUniformDouble(gen, rands, locations_count * BLOCKS_COUNT);
	CudaCheckError();

	CudaSafeCall(cudaMalloc(&idx_holder, sizeof(int) * BLOCKS_COUNT * locations_count));
	populateMemory<<<gridDim, blockDim>>>(idx_holder, locations_count, rands);
	CudaCheckError();

	int* idx = (int*)malloc(sizeof(int) * locations_count * BLOCKS_COUNT);

	location_t* locd;
	int locsize = sizeof(location_t) * locations_count;
	cudaMalloc(&locd, locsize);
	cudaMemcpy(locd, locations, locsize, cudaMemcpyHostToDevice);

	dist_idx_t* distances = (dist_idx_t*)malloc(BLOCKS_COUNT * sizeof(dist_idx_t));
	dist_idx_t* distances_d;
	cudaMalloc(&distances_d, 2 * BLOCKS_COUNT * sizeof(dist_idx_t));
	calcDistanceForEachPermutation<<<gridDim, blockDim, sizeof(double) * locations_count>>>(locd, idx_holder, distances_d);
	CudaCheckError();

	rankPermutations<<<1, BLOCKS_COUNT, 2 * sizeof(dist_idx_t) * BLOCKS_COUNT>>>(distances_d);
	CudaCheckError();
	// DEBUG
	cudaMemcpy(distances, distances_d, BLOCKS_COUNT * sizeof(dist_idx_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(idx, idx_holder, sizeof(int) * BLOCKS_COUNT * locations_count, cudaMemcpyDeviceToHost);
	for(int i = 0 ; i < BLOCKS_COUNT; i++)
	{
		printf("Distance for block %lf with index %d\n", distances[i].distance, distances[i].index);
//		for(int j = 0 ; j < locations_count ; j++)
//		{
//			printf("%d ", idx[i * locations_count + j]);
//		}
//		printf("\n");
	}
	return 0;
}
