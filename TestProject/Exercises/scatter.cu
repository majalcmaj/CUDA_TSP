// System includes
#include <stdio.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include<curand.h>
#define _USE_MATH_DEFINES
#include<math.h>

__device__ __host__ __inline__  float N(float x)
{
	return 0.5 + 0.5 * erf(x * M_SQRT1_2);
}

__device__ __host__ void price(float k, float s, float t, float r, float v, float *c, float* p) {
	float srt = v * sqrtf(t);
	float d1 = (logf(s/k) + (r + 0.5 * v * v) * t) /srt;
	float d2 = d1 - srt;
	float kert = k * expf(-r * t);
	*c = N(d1) * s - N(d2) * kert;
	*p = kert - s + *c;
}

__global__ void price(float* k, float* s, float* t, float* r, float *v, float* c, float *p)
{
	int idx = threadIdx.x;
	price(k[idx], s[idx], t[idx], r[idx], v[idx], c + idx, p + idx);
}

int scatter()
{
	const int count = 512;
	float *args[5];
	const int size = count * sizeof(float);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	for(int i = 0 ; i < 5 ; i ++)
	{
		cudaMalloc(args + i, size);
		curandGenerateUniform(gen, args[i], count);
	}

	float *dc, *dp;
	cudaMalloc(&dc, size);
	cudaMalloc(&dp, size);


	price<<<1, count>>>(args[0], args[1], args[2], args[3], args[4], dc, dp);

	float *hc, *hp;
	hc = (float*)malloc(size);
	hp = (float*)malloc(size);

	cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hp, dp, size, cudaMemcpyDeviceToHost);

	for(int i = 0 ; i < 512 ; i ++) {
		printf("Element %d has c = %f and p = %f\n", i, hc[i], hp[i]);
	}

	free(hc);
	free(hp);
	cudaFree(dc);
	cudaFree(dp);
	return 0;
}
