#include <iostream>
#include <ctime>
#include <stdio.h>

using namespace std;

#define N 1000000000

int main(void) {
	int Host1[N], Host2[N];
	for (int i = 0; i < N; i++) {
		Host1[i] = i * i;
	}
	int* Device1, * Device2;
	cudaMalloc((void**)&Device1, N * sizeof(int));
	cudaMalloc((void**)&Device2, N * sizeof(int));

	std::time_t time = std::time(NULL);
	cudaMemcpy(Host2, Host1, N * sizeof(int), cudaMemcpyHostToHost);	   // HostToHost
	cudaThreadSynchronize();
	std::time_t time_HtH = std::time(NULL) - time;

	time = std::time(NULL);
	cudaMemcpy(Device1, Host1, N * sizeof(int), cudaMemcpyHostToDevice);  // HostToDevice 
	cudaThreadSynchronize();
	std::time_t time_HtD1 = std::time(NULL) - time;

	cudaMallocHost((void**)&Host1, N * sizeof(int));	                       // HostToDevice pagelocking
	time = std::time(NULL);
	cudaMemcpy(Device1, Host1, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	std::time_t time_HtD2 = std::time(NULL) - time;

	time = std::time(NULL);
	cudaMemcpy(Host2, Device2, N * sizeof(int), cudaMemcpyDeviceToHost); // DeviceToHost 
	cudaThreadSynchronize();
	std::time_t time_DtH1 = std::time(NULL) - time;

	cudaMallocHost((void**)&Host2, N * sizeof(int));	                       // DeviceToHost pagelocking
	time = std::time(NULL);
	cudaMemcpy(Host2, Device2, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	std::time_t time_DtH2 = std::time(NULL) - time;

	time = std::time(NULL);
	cudaMemcpy(Device2, Device1, N * sizeof(int), cudaMemcpyDeviceToDevice);           // DeviceToDevice
	cudaThreadSynchronize();
	std::time_t time_DtD = std::time(NULL) - time;

	cout << "Host1 - " << (N * sizeof(int) / (1024 ^ 3)) << " GB" << endl;

	printf("Bandwidth:\n");
	printf("HostToHost: %d\n", time_HtH);
	printf("HostToDevice normal: %d\n", time_HtD1);
	printf("HostToDevice pagelocking: %d\n", time_HtD2);
	printf("DeviceToHost normal: %d\n", time_DtH1);
	printf("DeviceToHost pagelocking: %d\n", time_DtH2);
	printf("DeviceToDevice: %d\n", time_DtD);

	cudaFree(Device1);
	cudaFree(Device2);
	return 0;
}
