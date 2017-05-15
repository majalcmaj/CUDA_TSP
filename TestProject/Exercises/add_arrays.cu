/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void addArrays(int* a, int* b, int* c )
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int add_arrays()
{
	const int count = 5;
	const int size = count * sizeof(int);
	int ha[] = { 1, 2, 3, 4, 5 };
	int hb[] = { 10, 20, 30, 40, 50 };
	int hc[count];

	int *da, *db, *dc;
	cudaMalloc(&da, size);
	cudaMalloc(&db, size);
	cudaMalloc(&dc, size);

	cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

	addArrays<<<1, count>>>(da, db, dc);

	cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

	for(int i = 0 ; i < count ; i++) {
		printf("%d ", hc[i]);
	}
}
