#include <stdio.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void scan(int *d)
{
	int myIdx = threadIdx.x;
	int diff=1;
	while(myIdx + diff < blockDim.x + 1)
	{
		d[myIdx + diff] += d[myIdx];
		diff <<= 1;
		__syncthreads();
	}
}

int main()
{
	const int count = 16;
	const int size = count * sizeof(int);

	int h[16];
	for(int i = 0; i < count ; i++) {
		h[i] = i + 1;
	}
	int *d;
	cudaMalloc(&d, size);
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	scan<<<1, count-1>>>(d);

	cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
	for(int i = 0 ; i < count ; i++) {
		printf("%d: %d\n", i, h[i]);
	}
}
