#include <iostream>
#include <stdio.h>
#include <ctime>
using namespace std;

__global__ void func(unsigned long long int *a)
{ 

}

int main(void){
    int device_count; 
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&device_count);

    double seconds, t1, t2;
  
    unsigned long long int N = 131072000;
    unsigned long long int *dev_a, *hst_a;
    unsigned long long int size = sizeof( unsigned long long int );

    double sizeMByte = (double)(N * size)/pow(1024,3);
    cout << "block size =  " << (N * sizeof(hst_a))/pow(1024,2) << " MB\n" << endl;
    cout << device_count << " CUDA device(s) found \n" << endl;

    for (int device = 0; device < device_count; device++) {
        cudaGetDeviceProperties(&deviceProp, device);
        cout << "GPU " << device << " " << deviceProp.name << endl << endl;

        ///************ Copying Host -> Device *****************
        t1 = clock();
        cout << "Copying Host -> Device " << endl;  
        cudaMallocHost((void**)&hst_a, N * size);   
        cudaMalloc((void**)&dev_a, N * size); 

        cudaMemcpy(dev_a, hst_a, N * size, cudaMemcpyHostToDevice);
        func<<< 1, 1 >>>(dev_a);

        cudaDeviceSynchronize(); 
        t2 = clock();
        seconds = (double)(t2-t1)/CLOCKS_PER_SEC + 0.00000000001;
        printf("Average bandwidth: %.6f GB/s \n\n", (double)sizeMByte/seconds);
        cudaFree( dev_a ); cudaFree( hst_a );

        ///************ Copying Device -> Host *****************
        t1 = clock();
        cout << "Copying Device -> Host " << endl;    
        cudaMallocHost((void**)&hst_a, N * size);   
        cudaMalloc((void**)&dev_a, N * size); 

        cudaMemcpy(hst_a, dev_a, N * size, cudaMemcpyDeviceToHost);
        func<<< 1, 1 >>>(dev_a);

        cudaDeviceSynchronize(); 
        t2 = clock();
        seconds = (double)(t2-t1)/CLOCKS_PER_SEC + 0.00000000001;
        printf("Average bandwidth: %.6f GB/s \n\n", (double)sizeMByte/seconds);
        cudaFree( dev_a ); cudaFree( hst_a );


        ///************ Copying Host -> Host *****************
        t1 = clock();
        cout << "Copying Host -> Host " << endl;      
        cudaMallocHost((void**)&hst_a, N * size);  
        cudaMallocHost((void**)&dev_a, N * size);  

        cudaMemcpy(hst_a, dev_a, N * size, cudaMemcpyHostToHost);
        func<<< 1, 1 >>>(dev_a);

        cudaDeviceSynchronize(); 
        t2 = clock();
        seconds = (double)(t2-t1)/CLOCKS_PER_SEC + 0.00000000001;
        printf("Average bandwidth: %.6f GB/s \n\n", (double)sizeMByte/seconds);
        cudaFree( dev_a ); cudaFree( hst_a );
        

        ///************ Copying Device -> Device *****************  
        t1 = clock();
        cout << "Copying Device -> Device " << endl;  
        cudaMalloc((void**)&dev_a, N * size);
        cudaMalloc((void**)&hst_a, N * size);        

        cudaMemcpy(hst_a, dev_a, N * size, cudaMemcpyDeviceToDevice);
        func<<< 1, 1 >>>(dev_a);
        func<<< 1, 1 >>>(hst_a);

        cudaDeviceSynchronize(); 
        t2 = clock();
        seconds = (double)(t2-t1)/CLOCKS_PER_SEC + 0.00000000001;
        printf("Average bandwidth: %.6f GB/s \n\n", (double)sizeMByte/seconds);
        cudaFree( dev_a ); cudaFree( hst_a );

        

        // PAGELOCKED
        ///************ Copying Host -> Device *****************

        t1 = clock();
        cout << "Copying Host -> Device (usnig pagelocked)" << endl;  
        hst_a = (unsigned long long int*)malloc(N * size);   
        cudaMalloc((void**)&dev_a, N * size); 

        cudaMemcpy(dev_a, hst_a, N * size, cudaMemcpyHostToDevice);
        func<<< 1, 1 >>>(dev_a);

        cudaDeviceSynchronize(); 
        t2 = clock();
        seconds = (double)(t2-t1)/CLOCKS_PER_SEC + 0.00000000001;
        printf("Average bandwidth: %.6f GB/s \n\n", (double)sizeMByte/seconds);
        cudaFree( dev_a ); free( hst_a );



        ///************ Copying Device -> Host *****************
        t1 = clock();
        cout << "Copying Device -> Host (usnig pagelocked)" << endl; 
        hst_a = (unsigned long long int*)malloc(N * size); 
        cudaMalloc((void**)&dev_a, N * size); 

        cudaMemcpy(hst_a, dev_a, N * size, cudaMemcpyDeviceToHost);
        func<<< 1, 1 >>>(dev_a);

        cudaDeviceSynchronize(); 
        t2 = clock();
        seconds = (double)(t2-t1)/CLOCKS_PER_SEC + 0.00000000001;
        printf("Average bandwidth: %.6f GB/s \n\n", (double)sizeMByte/seconds);
        cudaFree( dev_a ); free( hst_a );
        
    }
    return 0;
}