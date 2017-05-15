
#include <stdio.h>
#include"common.h"
#include "DataIO/dataio.h"
#include "CudaTSP/cudatsp.h"

int main()
{
	int errno = 0;
	location_t* locations = NULL;
	int locations_length;

	errno = read_csv("../resources/CitiesLocationsSmall.csv", &locations, &locations_length);
	if(errno != 0)
	{
		fprintf(stderr, "Reading city locations from file failed. with errno=%d\n", errno);
		return 1;
	}

	run_cuda_tsp(locations, locations_length);

	free(locations);
	return 0;
}

//// System includes
//#include <stdio.h>
//// CUDA runtime
//#include <cuda_runtime.h>
//#include<device_launch_parameters.h>
//
//int main()
//{
//	int count;
//	cudaGetDeviceCount(&count);
//
//	printf("Available devices: %d\n", count);
//	cudaDeviceProp prop;
//
//	for(int i = 0; i < count; i++)
//	{
//		cudaGetDeviceProperties(&prop, i);
//		printf("Device: %d: %s\n", i, prop.name);
//		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
//		printf("Max grid dims: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
//		printf("Max block dims: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
//		printf("Shared mem per block %d\n", prop.sharedMemPerBlock);
//		printf("Max threads per block %d\n", prop.maxThreadsPerBlock);
//	}
//	return 0;
//}
