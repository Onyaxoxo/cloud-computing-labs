#include<iostream> 
using namespace std;
int main()
{
	int device_count;
	cudaGetDeviceCount(&device_count);
	cudaDeviceProp dp;
	cout << "CUDA device count: " << device_count << "\n";
	for (int i = 0; i < device_count; i++)
	{
		cudaGetDeviceProperties(&dp, i);
		cout << i << ": " << dp.name << " with CUDA compute compatibility " << dp.major << "." << dp.minor << "\n";
	}
	return 0;
}
