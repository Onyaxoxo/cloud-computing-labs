#include <iostream>
#include <thrust/random.h>
using namespace std;

#define N 10000
__device__ int Max_Of(int a, int b, int c) {
	int max;
	int mas[3] = { a,b,c };
	max = mas[0];
	for (int i = 0; i < 3; i++) {
		if (max < mas[i]) {
			max = mas[i]; // опред максимального
		}
	}
	return max;
}
__global__ void add(int *a, int *b, int *c, int *d) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	thrust::default_random_engine rand(123 * tid);
	thrust::uniform_int_distribution<int> dist(0, 9);
	a[tid] = dist(rand);
	b[tid] = dist(rand);
	c[tid] = dist(rand);

	if (tid < N) {
		d[tid] = Max_Of(a[tid], b[tid], c[tid]);
	}
}

int main(void) {
	int a[N], b[N], c[N], d[N];
	int *dev_a,  *dev_b,  *dev_c,  *dev_d;
	int numThreadsPerBlock = 1023; // нити (1023 максимум
	int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock; // блоки

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
	cudaMalloc((void**)&dev_d, N * sizeof(int));

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice); // копируем массив с цпу на гпу
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);

	add << <numBlocks, numThreadsPerBlock >> > (dev_a, dev_b, dev_c, dev_d); // нити блоки

	cudaMemcpy(d, dev_d, N * sizeof(int), cudaMemcpyDeviceToHost); // копируем с гпу на цпу
	cudaMemcpy(a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, dev_b, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 5; i++) {
		cout << "a|" << a[i] << "		" << "b|" << b[i] << "		" << "c|" << c[i] << "		" << "d|" << d[i] << endl;
	}

	cout << " " << endl;

	for (int i = N - 5; i < N; i++) {
		cout << "a|" << a[i] << "		" << "b|" << b[i] << "		" << "c|" << c[i] << "		" << "d|" << d[i] << endl;
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
