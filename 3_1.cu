#include <iostream>
using namespace std;

//������ ��� ������ ������
static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in file '" << file << "' at line " << line << endl;
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define N 10
__global__ void add(int* a, int* b, int* c)
{
	//���������� ������, ����������� �� ����� �������
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}
int main(void) {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;
	//�������� ������ �� GPU ��� ������� a,b,c
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));
	//��������� ������� 'a' � 'b' �� CPU
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	//�������� ������� 'a' � 'b' �� GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
	//��������� ���� �� N ������
	add << <N, 1 >> > (dev_a, dev_b, dev_c);
	//�������� ������ 'c' � GPU �� CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
	//������� ���������
	for (int i = 0; i < N; i++)
	{
		cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
	}
	//����������� ������, ���������� �� GPU
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));
	return 0;
}