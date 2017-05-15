// System includes
#include <stdio.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include<curand.h>
#define _USE_MATH_DEFINES
#include<math.h>


__global__ void sumSingleBlock(int *d)
{
	int tid = threadIdx.x;
	int myIdx = tid * 2;
	int diff = 1;
	while(myIdx + diff < 2 * blockDim.x) {
		d[myIdx] += d[myIdx + diff];
		diff <<= 1;
		__syncthreads();
	}
}
int main()
{
	const int count = 512;
	const int size = count * sizeof(int);

	int h[count];
	for(int i = 0 ; i < count; i++) {
		h[i] = 1 + i;
	}
	int *d;
	cudaMalloc(&d, size);
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
	sumSingleBlock<<<1, count/2>>>(d);

	int result;
	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Result: %d\n", result);
//	cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
//	for(int i = 0 ; i < count ; i++) {
//		printf("element %d: %d\n", i, h[i]);
//	}
	cudaFree(d);
	return 0;
}
