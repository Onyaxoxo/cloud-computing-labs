#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>

using namespace std;

int main(void)
{
    cudaDeviceProp dp;
    int dev_count;

    cudaGetDeviceCount(&dev_count);

    cout << dev_count << " CUDA devices" << "\n";

    //int i = 0;
    for (int i = 0; i < dev_count; ++i)
    {
        cudaGetDeviceProperties(&dp, i);

        cout << i << "\n";
        cout << dp.name << "\n";
        cout << "Clock frequency " << dp.clockRate << "KHz" << "\n";

        cout << "Global Memory " << dp.totalGlobalMem << "b" << "\n";
        cout << "Global Constant " << dp.totalConstMem << "b" << "\n";
        cout << "Maximum pitch " << dp.memPitch << "b" << "\n";

        cout << "Number of multiprocessors " << dp.multiProcessorCount << "\n";
        cout << "Shared memory available per block " << dp.sharedMemPerBlock << "\n";
        cout << "Registers available per block  " << dp.regsPerBlock << "\n";
        cout << "Warp size in threads " << dp.warpSize << "\n";
        cout << "Maximum number of threads per block " << dp.maxThreadsPerBlock << "\n";
        cout << "Maximum size of each dimension of a block  " << dp.maxThreadsDim[0] << " " << dp.maxThreadsDim[1] << " " << dp.maxThreadsDim[2] << "\n";
        cout << "Maximum size of each dimension of a grid " << dp.maxGridSize[0] << " " << dp.maxGridSize[1] << " " << dp.maxGridSize[2] << endl;

    }
    return 0;
}
