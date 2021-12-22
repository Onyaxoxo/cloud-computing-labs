#include<stdio.h> 
__global__ void kernel()
{
	printf("Hello World! \n");
}
int main()
{
	kernel << <1, 2 >> > ();
	kernel << <3, 1 >> > ();
	printf("Hello from CPU! \n");
	cudaDeviceSynchronize();
	return 0;
}
