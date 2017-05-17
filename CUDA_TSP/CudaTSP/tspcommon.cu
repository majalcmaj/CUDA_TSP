#include<cuda_runtime.h>
__device__ int binlog(int a)
{
	// Opt p. Draszawki
	int log = 0;
	while(a >>= 1) log++;
	return log;
}


