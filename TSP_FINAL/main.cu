
#include <stdio.h>
#include"common.h"
#include "DataIO/dataio.h"
#include "CudaTSP/cudatsp.h"
#include <sys/time.h>

#define DEVICE_0 0
#define DEVICE_1 1

int main()
{
	int errno = 0;
	int i;
	struct timespec start, end;
	location_t* locations = NULL;
	int locations_length;

	errno = read_csv("../resources/CitiesLocations64.csv", &locations, &locations_length);
	if(errno != 0)
	{
		fprintf(stderr, "Reading city locations from file failed. with errno=%d\n", errno);
		return 1;
	}

	for(i = 0 ; i < 5 ; i ++)
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		run_cuda_tsp(locations, locations_length, DEVICE_1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);

		double delta_us = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1000000000;
		printf("Time\n%lf\n", delta_us);
	}
	free(locations);
	return 0;
}
