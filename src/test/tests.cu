#include "tests.h"

#include <cuda_runtime.h>
#include <cuda/cmath>

#include <random>
#include <iostream>

#include "simulation/utils.h"

__global__ static void testPrefixSumKernel(int* values, int len)
{
	prefixSum<int>(values, len);
}

bool testPrefixSum()
{
	int len = 5000;
	int* h_testValues;
	cudaMallocHost(&h_testValues, len * sizeof(int));

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> dist(0, 2);
	for (int i = 0; i < len; i++)
	{
		h_testValues[i] = static_cast<int>(dist(rng) > 1);
	}

	int* d_values;
	cudaMalloc(&d_values, len * sizeof(int));
	cudaMemcpy(d_values, h_testValues, len * sizeof(int), cudaMemcpyDefault);

	void* kernelArgs[] = { &d_values, &len };
	int threadsPerBlock = 256;
	int blocks = cuda::ceil_div(len, threadsPerBlock);
	cudaLaunchCooperativeKernel(testPrefixSumKernel, dim3(blocks), dim3(threadsPerBlock), kernelArgs);

	for (int i = len - 1; i >= 0; i--)
	{
		int sum = 0;
		for (int j = 0; j <= i; j++)
		{
			sum += h_testValues[j];
		}
		h_testValues[i] = sum;
	}

	int* h_compareValues;
	cudaMallocHost(&h_compareValues, len * sizeof(int));

	cudaDeviceSynchronize();

	cudaMemcpy(h_compareValues, d_values, len * sizeof(int), cudaMemcpyDefault);

	for (int i = 0; i < len; i++)
	{
		if (h_testValues[i] != h_compareValues[i])
		{
			return false;
		}
	}

	return true;
}

bool testSplitAndSort()
{
	int len = 5000;
	int* h_testValues;
	int* h_testKeys;
	cudaMallocHost(&h_testValues, len * sizeof(int));
	cudaMallocHost(&h_testKeys, len * sizeof(int));

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> dist(0, 500);
	for (int i = 0; i < len; i++)
	{
		h_testValues[i] = static_cast<int>(dist(rng));
		h_testKeys[i] = (h_testValues[i] % 2 == 1);
	}

	int* d_values;
	int* d_keys;
	cudaMalloc(&d_values, len * sizeof(int));
	cudaMalloc(&d_keys, len * sizeof(int));
	cudaMemcpy(d_values, h_testValues, len * sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(d_keys, h_testKeys, len * sizeof(int), cudaMemcpyDefault);

	void* kernelArgs[] = { &d_values, &d_keys, &len };
	int threadsPerBlock = 256;
	int blocks = cuda::ceil_div(len, threadsPerBlock);
	cudaLaunchCooperativeKernel(k_splitAndSort<int, int>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);

	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len - i - 1; j++)
		{
			if (h_testKeys[j] > h_testKeys[j + 1])
			{
				int temp1 = h_testValues[j];
				h_testValues[j] = h_testValues[j + 1];
				h_testValues[j + 1] = temp1;

				int temp2 = h_testKeys[j];
				h_testKeys[j] = h_testKeys[j + 1];
				h_testKeys[j + 1] = temp2;
			}
		}
	}

	int* h_compareValues;
	int* h_compareKeys;
	cudaMallocHost(&h_compareValues, len * sizeof(int));
	cudaMallocHost(&h_compareKeys, len * sizeof(int));

	cudaDeviceSynchronize();

	cudaMemcpy(h_compareValues, d_values, len * sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(h_compareKeys, d_keys, len * sizeof(int), cudaMemcpyDefault);

	for (int i = 0; i < len; i++)
	{
		if (h_testValues[i] != h_compareValues[i] || h_testKeys[i] != h_compareKeys[i])
		{
			return false;
		}
	}

	return true;
}

bool testCompact()
{
	int len = 5000;
	int* h_testValues;
	int* h_testKeys;
	cudaMallocHost(&h_testValues, len * sizeof(int));
	cudaMallocHost(&h_testKeys, len * sizeof(int));

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> dist(0, 500);
	int newCount = 0;
	for (int i = 0; i < len; i++)
	{
		h_testValues[i] = static_cast<int>(dist(rng));
		h_testKeys[i] = (h_testValues[i] % 2 == 1);
		if (h_testKeys[i] == 1)
		{
			newCount++;
		}
	}

	int* d_values;
	int* d_keys;
	int* d_newLen;
	cudaMalloc(&d_values, len * sizeof(int));
	cudaMalloc(&d_keys, len * sizeof(int));
	cudaMalloc(&d_newLen, sizeof(int));
	cudaMemcpy(d_values, h_testValues, len * sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(d_keys, h_testKeys, len * sizeof(int), cudaMemcpyDefault);

	void* kernelArgs[] = { &d_values, &d_keys, &len, &d_newLen };
	int threadsPerBlock = 256;
	int blocks = cuda::ceil_div(len, threadsPerBlock);
	cudaLaunchCooperativeKernel(k_compact<int, int>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);

	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len - i - 1; j++)
		{
			if (h_testKeys[j] < h_testKeys[j + 1])
			{
				int temp1 = h_testValues[j];
				h_testValues[j] = h_testValues[j + 1];
				h_testValues[j + 1] = temp1;

				int temp2 = h_testKeys[j];
				h_testKeys[j] = h_testKeys[j + 1];
				h_testKeys[j + 1] = temp2;
			}
		}
	}

	int* h_compareValues;
	int* h_compareKeys;
	cudaMallocHost(&h_compareValues, len * sizeof(int));
	cudaMallocHost(&h_compareKeys, len * sizeof(int));

	cudaDeviceSynchronize();

	cudaMemcpy(h_compareValues, d_values, len * sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(h_compareKeys, d_keys, len * sizeof(int), cudaMemcpyDefault);

	for (int i = 0; i < newCount; i++)
	{
		if (h_testValues[i] != h_compareValues[i] || h_testKeys[i] != h_compareKeys[i])
		{
			return false;
		}
	}

	int* h_newLen;
	cudaMallocHost(&h_newLen, sizeof(int));
	cudaMemcpy(h_newLen, d_newLen, sizeof(int), cudaMemcpyDefault);

	return newCount == *h_newLen;
}

bool testRadixSort()
{
	int len = 5000;
	int* h_testValues;
	cudaMallocHost(&h_testValues, len * sizeof(int));

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> dist(0, 100000);
	for (int i = 0; i < len; i++)
	{
		h_testValues[i] = static_cast<int>(dist(rng));
	}

	int* d_values;
	int* d_keys;
	cudaMalloc(&d_values, len * sizeof(int));
	cudaMalloc(&d_keys, len * sizeof(int));
	cudaMemcpy(d_values, h_testValues, len * sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(d_keys, h_testValues, len * sizeof(int), cudaMemcpyDefault);

	void* kernelArgs[] = { &d_values, &d_keys, &len };
	int threadsPerBlock = 256;
	int blocks = cuda::ceil_div(len, threadsPerBlock);
	cudaLaunchCooperativeKernel(k_radixSortByKey<int, int>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);

	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len - i - 1; j++)
		{
			if (h_testValues[j] > h_testValues[j + 1])
			{
				int temp = h_testValues[j];
				h_testValues[j] = h_testValues[j + 1];
				h_testValues[j + 1] = temp;
			}
		}
	}

	int* h_compareValues;
	int* h_compareKeys;
	cudaMallocHost(&h_compareValues, len * sizeof(int));
	cudaMallocHost(&h_compareKeys, len * sizeof(int));

	cudaDeviceSynchronize();

	cudaMemcpy(h_compareValues, d_values, len * sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(h_compareKeys, d_keys, len * sizeof(int), cudaMemcpyDefault);

	for (int i = 0; i < len; i++)
	{
		if (h_testValues[i] != h_compareValues[i] || h_testValues[i] != h_compareKeys[i])
		{
			return false;
		}
	}

	return true;
}
