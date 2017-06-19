#include"cudatsp.h"
#include"tspcommon.h"
#include <stdio.h>
#include<stdlib.h>
// CUDA runtime
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include<curand.h>

#include<time.h>

#define BLOCKS_COUNT 256
#define ITERATIONS 10000

// DEBUG
#include"../DataIO/dataio.h"

__global__ void
populateMemory(int* idx_holder, int locations_count, double* rands);

__global__ void
calcDistanceForEachPermutation(int* idx_holder, dist_idx_t* distances);

__global__ void
rankPermutations(dist_idx_t* distances);

__global__ void
crossoverTheFittest(dist_idx_t* distances, int* indexes, double* rands);  // Todo calculate needed random arr size, shared size, blocks_count = len(distances) / 2

__global__ void
mutateGenomes(dist_idx_t* sorted_distances, int* indexes, double* rands);

void copyLocationsToDevice(location_t* locations, int locations_count);
void generate_randoms(curandGenerator_t* gen, double* rands, int count);
void init_curand(curandGenerator_t* gen, double** rands, size_t alloc_size);
void print_best(int number, int* indexes_d, dist_idx_t* distances_d, int locations_count, char print_indexes);

int run_cuda_tsp(location_t* locations, int locations_count, int on_device)
{
	// Host variables
	curandGenerator_t curand_generator;
	int i;

	// Device variables
	int* indexes_d;
	double* rands_d;
	dist_idx_t* distances_d;

	CudaSafeCall( cudaSetDevice(on_device) );


	// Initializaion
	init_curand(&curand_generator, &rands_d, 2 * sizeof(double) * locations_count * BLOCKS_COUNT);
	CudaSafeCall(cudaMalloc(&indexes_d, sizeof(int) * BLOCKS_COUNT * locations_count));


	copyLocationsToDevice(locations, locations_count);

	cudaMalloc(&distances_d, 2 * BLOCKS_COUNT * sizeof(dist_idx_t));

	// Logic
	populateMemory<<<BLOCKS_COUNT, locations_count, sizeof(int) * locations_count>>>(indexes_d, locations_count, rands_d);
	CudaCheckError();

	for(i = 0 ; i < ITERATIONS ; i ++)
	{
		calcDistanceForEachPermutation<<<BLOCKS_COUNT, locations_count, sizeof(double) * locations_count>>>(indexes_d, distances_d);
		CudaCheckError();

		rankPermutations<<<1, BLOCKS_COUNT, 2 * sizeof(dist_idx_t) * BLOCKS_COUNT>>>(distances_d);
		CudaCheckError();

		if(i % 1000 == 0 )
			print_best(i, indexes_d, distances_d, locations_count, 0);

		generate_randoms(&curand_generator, rands_d, 2 * sizeof(double) * BLOCKS_COUNT);
		crossoverTheFittest<<<BLOCKS_COUNT / 2, locations_count, 2 * locations_count * sizeof(int)>>>(distances_d, indexes_d, rands_d);

		generate_randoms(&curand_generator, rands_d, BLOCKS_COUNT * locations_count * 2);
		mutateGenomes<<<BLOCKS_COUNT, locations_count, sizeof(int) * locations_count>>>(distances_d, indexes_d, rands_d);
	}

	calcDistanceForEachPermutation<<<BLOCKS_COUNT, locations_count, sizeof(double) * locations_count>>>(indexes_d, distances_d);
	CudaCheckError();

	rankPermutations<<<1, BLOCKS_COUNT, 2 * sizeof(dist_idx_t) * BLOCKS_COUNT>>>(distances_d);
	CudaCheckError();

	print_best(i, indexes_d, distances_d, locations_count, 0);

	// Cleanup
	cudaFree(rands_d);
	cudaFree(distances_d);

	curandDestroyGenerator(curand_generator);

	return 0;
}

void init_curand(curandGenerator_t* gen, double** rands, size_t alloc_size)
{
	curandCreateGenerator(gen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(*gen, time(NULL));
	CudaCheckError();
	CudaSafeCall(cudaMalloc(rands, alloc_size));
}

void generate_randoms(curandGenerator_t* gen, double* rands, int count)
{
	curandGenerateUniformDouble(*gen, rands, count);
	CudaCheckError();
}

void print_best(int number, int* indexes_d, dist_idx_t* distances_d, int locations_count, char print_indexes)
{
	dist_idx_t best_distance;
	int* indexes = (int*)malloc(sizeof(int) * locations_count);
	CudaSafeCall( cudaMemcpy(&best_distance, distances_d, sizeof(dist_idx_t), cudaMemcpyDeviceToHost) );
	printf("%lf\n", best_distance.distance);

	if(print_indexes)
	{
		cudaMemcpy(indexes, &indexes_d[best_distance.index * locations_count], locations_count * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Indexes: \n");
		for(int i = 0 ; i < locations_count; i++)
		{
			printf("%d ", indexes[i]);
		}
		printf("\n\n");
	}
	free(indexes);
}
