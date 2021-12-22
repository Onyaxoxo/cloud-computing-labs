//сложение двух векторов на GPU 
#include <iostream> 
using namespace std;
#define N 10 //используется как количество элементов в массивах, и как количество задач для GPU
__global__ void add(int* a, int* b, int* c)
{

	int tid = blockIdx.x; //обработать данные, находящиеся по этому индексу блока

	//threadIdx - координаты нити в блоке нитей (threadIdx.x, threadIdx.y, threadIdx.z), значение не должно
	//превышать 1023
	//blockIdx - координаты блока нитей в сетке (blockIdx.x, blockIdx.y, blockIdx.z), значение не должно
	// превышать 65535 по одному из измерений
	//blockDim - размеры блока нитей (blockDim.x, blockDim.y, blockDim.z)
	//gridDim - размеры сетки блоков нитей (gridDim.x, gridDim.y, gridDim.z)

	if (tid < N)
		c[tid] = a[tid] + b[tid];
}
int main(void)
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;
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
	add << <N, 1 >> > (dev_a, dev_b, dev_c);
	// копируем массив 'c' с GPU на CPU
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	//выводим результат
	for (int i = 0; i < N; i++)
	{
		cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
	}
	//освобождаем память, выделенную на GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
