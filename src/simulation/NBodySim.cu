#include "NBodySim.h"

#include <cuda_runtime.h>
#include <cuda/cmath>
#include <iostream>

#include <random>

#include "NBodySimKernels.h"
#include "utils.h"

const int domainMin = 0;
const int domainMax = 1000;
const int Nleaf = 16;

NBodySim::NBodySim(int bodyCount)
{
	_bodyCount = bodyCount;

	cudaMallocHost(&_h_particleInfos, _bodyCount * sizeof(float4));
	cudaMalloc(&_d_particleInfos, _bodyCount * sizeof(float4));
	cudaMalloc(&_keys, _bodyCount * sizeof(uint64_t));
	cudaMalloc(&_flagged, _bodyCount * sizeof(bool));
	cudaMalloc(&_activeList, _bodyCount * sizeof(int));
	cudaMalloc(&_maskedKeys, _bodyCount * sizeof(uint64_t));
	cudaMalloc(&_headFlags, _bodyCount * sizeof(int));
	cudaMalloc(&_groupStarts, _bodyCount * sizeof(int));

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> posDist(domainMin, domainMax);
	std::uniform_real_distribution<float> massDist(100, 200);
	for (int i = 0; i < bodyCount; i++)
	{
		_h_particleInfos[i] = make_float4(posDist(rng), posDist(rng), 0, massDist(rng));
	}

	cudaMemcpy(_d_particleInfos, _h_particleInfos, bodyCount * sizeof(float4), cudaMemcpyDefault);
	cudaFreeHost(_h_particleInfos);
}

NBodySim::~NBodySim()
{
	cudaFree(_d_particleInfos);
	cudaFree(_keys);
	cudaFree(_flagged);
	cudaFree(_activeList);
	cudaFree(_maskedKeys);
	cudaFree(_headFlags);
	cudaFree(_groupStarts);
}

void NBodySim::Simulate()
{
	int threadsPerBlock = 256;
	int blocks = cuda::ceil_div(_bodyCount, threadsPerBlock);
	computeMortonKeys<<<blocks, threadsPerBlock>>>(_d_particleInfos, _bodyCount, domainMin, domainMax, _keys);

	cudaDeviceSynchronize();

	void* kernelArgs[] = { &_d_particleInfos, &_keys, &_bodyCount };
	cudaLaunchCooperativeKernel(radixSortByKey<float4, uint64_t>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);

	cudaDeviceSynchronize();

	bool* _flagged;
	initActiveList<<<blocks, threadsPerBlock>>>(_activeList, _bodyCount);
	int level = 0;
	int len = _bodyCount;
	while (len > 0 && level < 20)
	{
		getMaskedValues<<<blocks, threadsPerBlock>>>(_keys, _activeList, len, level, _maskedKeys);
		cudaDeviceSynchronize();

		void* kernelArgs[] = { &_activeList, &_maskedKeys, &len };
		cudaLaunchCooperativeKernel(radixSortByKey<int, uint64_t>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);
		cudaDeviceSynchronize();

		setHeadFlags<<<blocks, threadsPerBlock>>>(_maskedKeys, len, _headFlags);
		cudaDeviceSynchronize();

		int dummyKey1 = 0;
		void* kernelArgs[] = { &_headFlags, &len, &dummyKey1, &_groupStarts };
		cudaLaunchCooperativeKernel(compactIndices<int, int>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);
		cudaDeviceSynchronize();

		// TODO: this is wrong, _headFlags is on the GPU
		int numGroups = 0;
		for (int i = 0; i < len; i++)
		{
			if (_headFlags[i])
			{
				numGroups++;
			}
		}
		int* groupSizes = _headFlags;
		getGroupSizes<<<blocks, threadsPerBlock>>>(_groupStarts, numGroups, len, groupSizes);
		cudaDeviceSynchronize();

		classifyGroups<<<blocks, threadsPerBlock>>>(_activeList, _groupStarts, groupSizes, numGroups);
		cudaDeviceSynchronize();

		void* kernelArgs[] = { &_activeList, &len, &_flagged };
		cudaLaunchCooperativeKernel(setFlagged, dim3(blocks), dim3(threadsPerBlock), kernelArgs);
		cudaDeviceSynchronize();

		// TODO: this is wrong, _flagged is on the GPU
		int newLen = 0;
		for (int i = 0; i < len; i++)
		{
			if (_flagged[i])
			{
				newLen++;
			}
		}

		bool dummyKey2 = 0;
		void* kernelArgs[] = { &_activeList, &_flagged, &len, &dummyKey2 };
		cudaLaunchCooperativeKernel(compact<int, bool>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);
		cudaDeviceSynchronize();

		len = newLen;

		level++;
	}
}

void NBodySim::Render(uchar4* pbo)
{
}
