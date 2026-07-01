#include "NBodySim.h"

#include <cuda_runtime.h>
#include <cuda/cmath>
#include <iostream>

#include <random>

#include "NBodySimKernels.h"
#include "utils.h"

const int domainMin = 0;
const int domainMax = 1000;
const int NLeaf = 16;

NBodySim::NBodySim(int bodyCount)
{
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	_bodyCount = bodyCount;

	cudaMallocHost(&_h_particleInfos, _bodyCount * sizeof(float4));
	cudaMalloc(&_d_particleInfos, _bodyCount * sizeof(float4));
	cudaMalloc(&_keys, _bodyCount * sizeof(uint64_t));
	cudaMalloc(&_flagged, _bodyCount * sizeof(bool));
	cudaMalloc(&_activeList, _bodyCount * sizeof(int));
	cudaMalloc(&_maskedKeys, _bodyCount * sizeof(uint64_t));
	cudaMalloc(&_headFlags, _bodyCount * sizeof(int));
	cudaMalloc(&_groupStarts, _bodyCount * sizeof(int));
	cudaMalloc(&_numGroups, sizeof(int));
	cudaMemset(_numGroups, 0, sizeof(int));
	cudaMalloc(&_newLen, sizeof(int));
	cudaMemset(_newLen, 0, sizeof(int));
	cudaMalloc(&_cells, _bodyCount * sizeof(Cell));
	cudaMalloc(&_cellCount, sizeof(int));
	cudaMemset(_cellCount, 0, sizeof(int));
	cudaMalloc(&_leafParticles, _bodyCount * sizeof(int));
	cudaMalloc(&_leafParticleCount, sizeof(int));
	cudaMemset(_leafParticleCount, 0, sizeof(int));
	cudaMalloc(&_flaggedTemp, _bodyCount * sizeof(int));

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> posDist(domainMin, domainMax);
	std::uniform_real_distribution<float> massDist(100, 10000);
	for (int i = 0; i < bodyCount; i++)
	{
		_h_particleInfos[i] = make_float4(posDist(rng), 0, posDist(rng), massDist(rng));
	}

	cudaMemcpy(_d_particleInfos, _h_particleInfos, bodyCount * sizeof(float4), cudaMemcpyDefault);
	cudaFreeHost(_h_particleInfos);

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float time;
	cudaEventElapsedTime(&time, start, end);
	std::cout << "\n" << "Sim init time: " << time << " ms\n" << std::flush;
	for (int i = 0; i < 3; i++)
	{
		std::cout << "Dummy\n" << std::flush;
	}
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
	cudaFree(_numGroups);
	cudaFree(_newLen);
	cudaFree(_cells);
	cudaFree(_cellCount);
	cudaFree(_leafParticles);
	cudaFree(_leafParticleCount);
	cudaFree(_flaggedTemp);
}

void NBodySim::Simulate()
{
	cudaEvent_t startMorton, endMorton;
	cudaEventCreate(&startMorton);
	cudaEventCreate(&endMorton);

	cudaEventRecord(startMorton);

	int threadsPerBlock = 256;
	int blocks = cuda::ceil_div(_bodyCount, threadsPerBlock);
	computeMortonKeys<<<blocks, threadsPerBlock>>>(_d_particleInfos, _bodyCount, domainMin, domainMax, _keys);
	cudaDeviceSynchronize();

	void* kernelArgs[] = { &_d_particleInfos, &_keys, &_bodyCount };
	cudaLaunchCooperativeKernel(k_radixSortByKey<float4, uint64_t>, dim3(blocks), dim3(threadsPerBlock), kernelArgs);
	cudaDeviceSynchronize();

	cudaEventRecord(endMorton);

	cudaEvent_t startPartition, endPartition;
	cudaEventCreate(&startPartition);
	cudaEventCreate(&endPartition);

	cudaEventRecord(startPartition);

	initActiveList<<<blocks, threadsPerBlock>>>(_activeList, _bodyCount);
	cudaMemset(_flagged, 0, _bodyCount * sizeof(bool));
	cudaMemset(_cellCount, 0, sizeof(int));
	cudaMemset(_leafParticleCount, 0, sizeof(int));
	cudaMemset(_numGroups, 0, sizeof(int));
	cudaMemset(_newLen, 0, sizeof(int));
	int level = 0;
	int len = _bodyCount;
	while (len > 0 && level < 20)
	{
		blocks = cuda::ceil_div(len, threadsPerBlock);

		getMaskedValues<<<blocks, threadsPerBlock>>>(_keys, _activeList, len, level, _maskedKeys);
		cudaDeviceSynchronize();

		void* kernelArgs1[] = { &_activeList, &_maskedKeys, &len };
		cudaLaunchCooperativeKernel(k_radixSortByKey<int, uint64_t>, dim3(blocks), dim3(threadsPerBlock), kernelArgs1);
		cudaDeviceSynchronize();

		setHeadFlags<<<blocks, threadsPerBlock>>>(_maskedKeys, len, _headFlags);
		cudaDeviceSynchronize();

		void* kernelArgs2[] = { &_headFlags, &len, &_groupStarts, &_numGroups };
		cudaLaunchCooperativeKernel(k_compactIndices<int, int>, dim3(blocks), dim3(threadsPerBlock), kernelArgs2);
		cudaDeviceSynchronize();

		int* groupSizes = _headFlags;
		getGroupSizes<<<blocks, threadsPerBlock>>>(_groupStarts, _numGroups, len, groupSizes);
		cudaDeviceSynchronize();

		classifyGroups<<<blocks, threadsPerBlock>>>(_activeList, _groupStarts, groupSizes, _numGroups, _maskedKeys, level, NLeaf, _flagged, _cells, _cellCount, _leafParticles, _leafParticleCount);
		cudaDeviceSynchronize();
		
		setCompactFlags<<<blocks, threadsPerBlock>>>(_activeList, len, _flagged, _flaggedTemp);
		cudaDeviceSynchronize();

		void* kernelArgs4[] = { &_activeList, &_flaggedTemp, &len, &_newLen };
		cudaLaunchCooperativeKernel(k_compact<int, int>, dim3(blocks), dim3(threadsPerBlock), kernelArgs4);
		cudaDeviceSynchronize();
		
		cudaMemcpy(&len, _newLen, sizeof(int), cudaMemcpyDeviceToHost);

		level++;
	}

	int cellCount;
	cudaMemcpy(&cellCount, _cellCount, sizeof(int), cudaMemcpyDeviceToHost);
	blocks = cuda::ceil_div(cellCount, threadsPerBlock);
	extractCellKeys<<<blocks, threadsPerBlock>>>(_cells, cellCount, _maskedKeys);
	cudaDeviceSynchronize();

	void* kernelArgs1[] = { &_cells, &_maskedKeys, &cellCount };
	cudaLaunchCooperativeKernel(k_radixSortByKey<Cell, uint64_t>, dim3(blocks), dim3(threadsPerBlock), kernelArgs1);
	cudaDeviceSynchronize();

	linkCellsToParents<<<blocks, threadsPerBlock>>>(_cells, cellCount);
	cudaDeviceSynchronize();

	cudaEventRecord(endPartition);

	cudaEventSynchronize(endMorton);
	cudaEventSynchronize(endPartition);

	float timeMorton;
	cudaEventElapsedTime(&timeMorton, startMorton, endMorton);
	std::cout << "\x1b[3A" << std::flush;
	std::cout << "Morton keys and sort time: " << timeMorton << " ms\x1b[K\n" << std::flush;

	float timePartition;
	cudaEventElapsedTime(&timePartition, startPartition, endPartition);
	std::cout << "Partition time: " << timePartition << " ms\x1b[K\n" << std::flush;
}
