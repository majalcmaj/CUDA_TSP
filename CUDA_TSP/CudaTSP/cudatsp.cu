#include"cudatsp.h"
#include"tspcommon.h"
#include <stdio.h>
// CUDA runtime
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include<curand.h>

#include<time.h>

#define BLOCKS_COUNT 2
#define ITERATIONS 10

// DEBUG
#include"../DataIO/dataio.h"

__global__ void
calcDistanceForEachPermutation(location_t* locations, int* idx_holder, dist_idx_t* distances);

__global__ void
populateMemory(int* idx_holder, int locations_count, double* rands);

__global__ void
rankPermutations(dist_idx_t* distances);

__global__ void
crossoverTheFittest(dist_idx_t* distances, int* indexes, double* rands);  // Todo calculate needed random arr size, shared size, blocks_count = len(distances) / 2

__global__ void
mutateGenomes(int* indexes, double* rands);

void generate_randoms(curandGenerator_t* gen, double* rands, int count);
void init_curand(curandGenerator_t* gen, double* rands, size_t alloc_size);
void print_best(int number, int* indexes_d, dist_idx_t* distances_d, int locations_count);

int run_cuda_tsp(location_t* locations, int locations_count)
{
	// Host variables
	int locsize;
	curandGenerator_t curand_generator;
	int i;

	// Device variables
	int* indexes_d;
	double* rands_d;
	location_t* locations_d;
	dist_idx_t* distances_d;

	// Initializaion
	init_curand(&curand_generator, rands_d, 2 * sizeof(double) * locations_count * BLOCKS_COUNT);
	CudaSafeCall(cudaMalloc(&indexes_d, sizeof(int) * BLOCKS_COUNT * locations_count));

	locsize = sizeof(location_t) * locations_count;
	cudaMalloc(&locations_d, locsize);
	cudaMemcpy(locations_d, locations, locsize, cudaMemcpyHostToDevice);

	cudaMalloc(&distances_d, 2 * BLOCKS_COUNT * sizeof(dist_idx_t));

	// Logic
	populateMemory<<<gridDim, blockDim>>>(indexes_d, locations_count, rands_d);
	CudaCheckError();

	for(i = 0 ; i < ITERATIONS ; i ++)
	{
		calcDistanceForEachPermutation<<<BLOCKS_COUNT, locations_count, sizeof(double) * locations_count>>>(locations_d, indexes_d, distances_d);
		CudaCheckError();

		rankPermutations<<<1, BLOCKS_COUNT, 2 * sizeof(dist_idx_t) * BLOCKS_COUNT>>>(distances_d);
		CudaCheckError();

		print_best(i, indexes_d, distances_d, locations_count);

		generate_randoms(&curand_generator, rands_d, 2 * sizeof(double) * BLOCKS_COUNT);
		crossoverTheFittest<<<BLOCKS_COUNT / 2, locations_count, locations_count * sizeof(int)>>>(distances_d, indexes_d, rands_d);

		generate_randoms(&curand_generator, rands_d, BLOCKS_COUNT * locations_count * 2);
		mutateGenomes<<<BLOCKS_COUNT * 2, locations_count>>>(indexes_d, rands_d);
	}

	calcDistanceForEachPermutation<<<BLOCKS_COUNT, locations_count, sizeof(double) * locations_count>>>(locations_d, indexes_d, distances_d);
	CudaCheckError();

	rankPermutations<<<1, BLOCKS_COUNT, 2 * sizeof(dist_idx_t) * BLOCKS_COUNT>>>(distances_d);
	CudaCheckError();

	print_best(i, indexes_d, distances_d, locations_count);

	// Cleanup
	cudaFree(rands_d);
	cudaFree(locations_d);
	cudaFree(locations_d);
	cudaFree(distances_d);

	curandDestroyGenerator(curand_generator);

	return 0;
}

void init_curand(curandGenerator_t* gen, double* rands, size_t alloc_size)
{
	curandCreateGenerator(gen, CURAND_RNG_PSEUDO_MTGP32);
	CudaCheckError();
	CudaSafeCall(cudaMalloc(&rands, alloc_size));
}

void generate_randoms(curandGenerator_t* gen, double* rands, int count)
{
	curandGenerateUniformDouble(*gen, rands, count);
	CudaCheckError();
}

void print_best(int number, int* indexes_d, dist_idx_t* distances_d, int locations_count)
{
	dist_idx_t best_distance;
	int* indexes = (int*)malloc(sizeof(int) * locations_count);
	CudaSafeCall( cudaMemcpy(&best_distance, indexes_d, sizeof(dist_idx_t), cudaMemcpyDeviceToHost) );
	CudaSafeCall( cudaMemcpy(&indexes, indexes_d, sizeof(int) * locations_count, cudaMemcpyDeviceToHost) );
	//	cudaMemcpy(indexes, indexes_d, LENGTH * 4 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Iteration %d\nBest distance: %lf\nIndexes:\n");
	for(int i = 0 ; i < locations_count; i++)
	{
		printf("%d ", indexes);
	}
	free(indexes);
}

// DEBUG
//#define LENGTH 16
//	dist_idx_t distances[4];
//	distances[0].index = 0;
//	distances[1].index = 1;
//	distances[2].index = 2;
//	distances[3].index = 3;
//
//	dist_idx_t* distances_d;
//	cudaMalloc(&distances_d, 4 * sizeof(dist_idx_t));
//	cudaMemcpy(distances_d, distances, 4 * sizeof(dist_idx_t), cudaMemcpyHostToDevice);
//	int indexes[LENGTH * 4] = {
//			5, 0xa, 1, 0xb, 2, 4, 3, 0xc, 0, 6, 8, 7, 0xf, 9, 0xe, 0xd,
//			2, 0, 7, 4, 0xc, 1, 6, 0xb, 3, 0xa, 0xd, 5, 0xe, 0xf, 9, 8
//	};
//	int* indexes_d;
//	cudaMalloc(&indexes_d, LENGTH * 4 * sizeof(int));
//	cudaMemcpy(indexes_d, indexes, LENGTH * 4 * sizeof(int), cudaMemcpyHostToDevice);
//
//	double* rands;
//	curandGenerator_t gen;
//	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
//	CudaCheckError();
//	CudaSafeCall(cudaMalloc(&rands, 2 * sizeof(double) * BLOCKS_COUNT));
//	curandGenerateUniformDouble(gen, rands, 2 * BLOCKS_COUNT);
//	CudaCheckError();
//
//	cudaMemcpy(indexes, indexes_d, LENGTH * 4 * sizeof(int), cudaMemcpyDeviceToHost);
//	for(int i = 0 ; i < 4 * LENGTH; i++)
//	{
//		printf("%5X", indexes[i]);
//		if(i % LENGTH == LENGTH-1)
//			printf("\n");
//	}

//printf("MUTATING\n\n");
//cudaMemcpy(indexes, indexes_d, LENGTH * 4 * sizeof(int), cudaMemcpyDeviceToHost);
//for(int i = 0 ; i < 4 * LENGTH; i++)
//{
//	printf("%5X", indexes[i]);
//	if(i % LENGTH == LENGTH-1)
//		printf("\n");
//}
