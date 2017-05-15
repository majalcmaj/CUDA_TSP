#include <stdio.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void scan(int *d)
{
	extern __shared__ int dcopy[];
	int myIdx = threadIdx.x;
	int diff=1;
	dcopy[myIdx] = d[myIdx];
	while(myIdx + diff < blockDim.x)
	{
		dcopy[myIdx + diff] += dcopy[myIdx];
		diff <<= 1;
		__syncthreads();
	}
	d[myIdx] = dcopy[myIdx];
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

	scan<<<1, count, size>>>(d);

	cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
	for(int i = 0 ; i < count ; i++) {
		printf("%d: %d\n", i, h[i]);
	}
}
