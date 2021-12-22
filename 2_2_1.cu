//сложение двух векторов на GPU 
#include <iostream> 
using namespace std;
#define N 10 //используется как количество элементов в массивах, и как количество задач для GPU
__global__ void add(int *a, int *b, int *c)
{

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}
int main(void)
{
	int a[N], b[N], c[N];
	int *dev_a,  *dev_b,  *dev_c;
	int numThreadsPerBlock = 10;
	int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

	//выделяем память на GPU под массивы a,b,c//cudaMalloc выделяет память на аппаратном уровне, т.е. на видеокарте.
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
	//заполняем массивы 'a' и 'b' на CPU
	for (int i = 0; i < N; i++)
	{
		a[i] = i + 1;
		b[i] = i + 1;
	}
	//копируем массивы 'a' и 'b' на GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	//запускаем ядро на N блоках
	add << <numBlocks, numThreadsPerBlock >> > (dev_a, dev_b, dev_c);
	// копируем массив 'c' с GPU на CPU
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	//выводим результат
	for (int i = 0; i < N; i++){
		cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
	}
	//освобождаем память, выделенную на GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
