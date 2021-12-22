#include <iostream>
#include <thrust/random.h>
using namespace std;
//макрос для отлова ошибок
static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in file '" << file << "' at line " << line << endl;
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
// поиск максимального значения элементов векторов.
#define N 10000
__device__ int max_of(int a, int b, int c) {
	int max;
	int mas[3] = { a,b,c };
	max = mas[0];
	for (int i = 0; i < 3; i++) {
		if (max < mas[i]) max = mas[i];
	}
	return max;
}
__global__ void add(int* a, int* b, int* c, int* d) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	thrust::default_random_engine rand(123 * tid);
	thrust::uniform_int_distribution<int> dist(0, 9);

	a[tid] = dist(rand);
	b[tid] = dist(rand);
	c[tid] = dist(rand);

	if (tid < N) {
		d[tid] = max_of(a[tid], b[tid], c[tid]);
	}
}

int main(void) {
	int a[N], b[N], c[N], d[N];
	int* dev_a, * dev_b, * dev_c, * dev_d;
	int numThreadsPerBlock = 1023;
	int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_d, N * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice));

	add << <numBlocks, numThreadsPerBlock >> > (dev_a, dev_b, dev_c, dev_d);

	HANDLE_ERROR(cudaMemcpy(d, dev_d, N * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(b, dev_b, N * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for(int i = 0; i < 5; i++) {
		cout << "a[" << a[i] << "		" << "b["<< b[i] << "		" << "c[" << c[i] << "		" << "d[" << d[i] << endl;
	}

	cout << " " << endl;

	for(int i = N - 5; i < N; i++) {
		cout << "a[" << a[i] << "		" << "b[" << b[i] << "		" << "c[" << c[i] << "		" << "d[" << d[i] << endl;
	}

	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	return 0;
}