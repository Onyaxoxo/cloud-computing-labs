//�������� ���� �������� �� GPU 
#include <iostream> 
using namespace std;
#define N 10 //������������ ��� ���������� ��������� � ��������, � ��� ���������� ����� ��� GPU
__global__ void add(int* a, int* b, int* c)
{

	int tid = blockIdx.x; //���������� ������, ����������� �� ����� ������� �����

	//threadIdx - ���������� ���� � ����� ����� (threadIdx.x, threadIdx.y, threadIdx.z), �������� �� ������
	//��������� 1023
	//blockIdx - ���������� ����� ����� � ����� (blockIdx.x, blockIdx.y, blockIdx.z), �������� �� ������
	// ��������� 65535 �� ������ �� ���������
	//blockDim - ������� ����� ����� (blockDim.x, blockDim.y, blockDim.z)
	//gridDim - ������� ����� ������ ����� (gridDim.x, gridDim.y, gridDim.z)

	if (tid < N)
		c[tid] = a[tid] + b[tid];
}
int main(void)
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;
	//�������� ������ �� GPU ��� ������� a,b,c//cudaMalloc �������� ������ �� ���������� ������, �.�. �� ����������.
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));
	//��������� ������� 'a' � 'b' �� CPU
	for (int i = 0; i < N; i++)
	{
		a[i] = i + 1;
		b[i] = i + 1;
	}
	//�������� ������� 'a' � 'b' �� GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	//��������� ���� �� N ������
	add << <N, 1 >> > (dev_a, dev_b, dev_c);
	// �������� ������ 'c' � GPU �� CPU
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	//������� ���������
	for (int i = 0; i < N; i++)
	{
		cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
	}
	//����������� ������, ���������� �� GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
