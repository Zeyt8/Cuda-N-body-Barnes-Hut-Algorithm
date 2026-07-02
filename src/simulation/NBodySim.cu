#include "NBodySim.h"

#include <cuda_runtime.h>
#include <cuda/cmath>
#include <iostream>

#include <random>

#include "NBodySimKernels.h"
#include "utils.h"

const int domainMin = 0;
const int domainMax = 5000;
const int NLeaf = 16;

NBodySim::NBodySim(int bodyCount)
{
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	_bodyCount = bodyCount;

	float4* h_particleInfos;
	float3* h_initVelocities;
	cudaMallocHost(&h_particleInfos, _bodyCount * sizeof(float4));
	cudaMallocHost(&h_initVelocities, _bodyCount * sizeof(float3));
	cudaMalloc(&_d_particleInfos, _bodyCount * sizeof(float4));
	cudaMalloc(&_keys, _bodyCount * sizeof(uint64_t));
	cudaMalloc(&_keysVel, _bodyCount * sizeof(uint64_t));
	cudaMalloc(&_keysAcc, _bodyCount * sizeof(uint64_t));
	cudaMalloc(&_keysAccOld, _bodyCount * sizeof(uint64_t));
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
	cudaMalloc(&_leafIndices, _bodyCount * sizeof(int));
	cudaMalloc(&_leafCount, sizeof(int));
	cudaMemset(_leafCount, 0, sizeof(int));
	cudaMalloc(&_velocities, bodyCount * sizeof(float3));
	cudaMalloc(&_accelerations, bodyCount * sizeof(float3));
	cudaMemset(_accelerations, 0, bodyCount * sizeof(float3));
	cudaMalloc(&_accelerationsPrev, bodyCount * sizeof(float3));
	cudaMemset(_accelerationsPrev, 0, bodyCount * sizeof(float3));

	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> posDist(domainMin + (domainMax - domainMin) / 4, domainMax - (domainMax - domainMin) / 4);
	std::uniform_real_distribution<float> massDist(0, 1);
	int bigMass = 101;
	h_particleInfos[0] = make_float4(domainMax / 2 + 10, 1, domainMax / 2 + 10, bigMass);
	for (int i = 1; i < bodyCount; i++)
	{
		double r = massDist(rng);
		double biased = std::pow(r, 10);
		double scaled = 0.1f + biased * (1 - 0.1f);
		h_particleInfos[i] = make_float4(posDist(rng), 1, posDist(rng), scaled);
	}

	h_initVelocities[0] = make_float3(0, 0, 0);
	for (int i = 1; i < bodyCount; i++)
	{
		float dx = h_particleInfos[i].x - (domainMax + domainMin) / 2.0f;
		float dz = h_particleInfos[i].z - (domainMax + domainMin) / 2.0f;
		float r = sqrtf(dx * dx + dz * dz);
		float v = sqrtf(bigMass / r);
		float3 vel = make_float3(dz / r, 0.0f, -dx / r) * v;
		h_initVelocities[i] = vel;
	}
	cudaMemcpy(_velocities, h_initVelocities, bodyCount * sizeof(float3), cudaMemcpyDefault);

	cudaMemcpy(_d_particleInfos, h_particleInfos, bodyCount * sizeof(float4), cudaMemcpyDefault);
	cudaFreeHost(h_initVelocities);
	cudaFreeHost(h_particleInfos);

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
	cudaFree(_keysVel);
	cudaFree(_keysAcc);
	cudaFree(_keysAccOld);
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
	cudaFree(_leafIndices);
	cudaFree(_leafCount);
	cudaFree(_accelerations);
	cudaFree(_velocities);
	cudaFree(_accelerationsPrev);
}

void NBodySim::Simulate(float delta)
{
	int threadsPerBlock = 256;
	int bodyBlocks = cuda::ceil_div(_bodyCount, threadsPerBlock);

	correctVelocities<<<bodyBlocks, threadsPerBlock>>>(_velocities, _bodyCount, _accelerations, _accelerationsPrev, delta);
	cudaDeviceSynchronize();

	movePos<<<bodyBlocks, threadsPerBlock>>>(_d_particleInfos, _bodyCount, _velocities, _accelerations, delta);
	cudaDeviceSynchronize();

	cudaEvent_t startMorton, endMorton;
	cudaEventCreate(&startMorton);
	cudaEventCreate(&endMorton);

	cudaEventRecord(startMorton);

	computeMortonKeys<<<bodyBlocks, threadsPerBlock>>>(_d_particleInfos, _bodyCount, domainMin, domainMax, _keys);
	cudaDeviceSynchronize();

	cudaMemcpy(_keysVel, _keys, _bodyCount * sizeof(uint64_t), cudaMemcpyDefault);
	cudaMemcpy(_keysAcc, _keys, _bodyCount * sizeof(uint64_t), cudaMemcpyDefault);
	cudaMemcpy(_keysAccOld, _keys, _bodyCount * sizeof(uint64_t), cudaMemcpyDefault);

	void* kernelArgs[] = { &_d_particleInfos, &_keys, &_bodyCount };
	cudaLaunchCooperativeKernel(k_radixSortByKey<float4, uint64_t>, dim3(bodyBlocks), dim3(threadsPerBlock), kernelArgs);
	cudaDeviceSynchronize();

	void* argsSort1[] = { &_velocities, &_keysVel, &_bodyCount };
	cudaLaunchCooperativeKernel(k_radixSortByKey<float3, uint64_t>, dim3(bodyBlocks), dim3(threadsPerBlock), argsSort1);
	cudaDeviceSynchronize();

	void* argsSort2[] = { &_accelerations, &_keysAcc, &_bodyCount };
	cudaLaunchCooperativeKernel(k_radixSortByKey<float3, uint64_t>, dim3(bodyBlocks), dim3(threadsPerBlock), argsSort2);
	cudaDeviceSynchronize();

	void* argsSort3[] = { &_accelerationsPrev, &_keysAccOld, &_bodyCount };
	cudaLaunchCooperativeKernel(k_radixSortByKey<float3, uint64_t>, dim3(bodyBlocks), dim3(threadsPerBlock), argsSort3);
	cudaDeviceSynchronize();

	cudaEventRecord(endMorton);

	cudaEvent_t startPartition, endPartition;
	cudaEventCreate(&startPartition);
	cudaEventCreate(&endPartition);

	cudaEventRecord(startPartition);

	initActiveList<<<bodyBlocks, threadsPerBlock>>>(_activeList, _bodyCount);
	cudaMemset(_flagged, 0, _bodyCount * sizeof(bool));
	cudaMemset(_cellCount, 0, sizeof(int));
	cudaMemset(_leafParticleCount, 0, sizeof(int));
	cudaMemset(_numGroups, 0, sizeof(int));
	cudaMemset(_newLen, 0, sizeof(int));
	int level = 0;
	int maxLevel = 0;
	int len = _bodyCount;
	while (len > 0 && level < 20)
	{
		int activeBlocks = cuda::ceil_div(len, threadsPerBlock);

		getMaskedValues<<<activeBlocks, threadsPerBlock>>>(_keys, _activeList, len, level, _maskedKeys);
		cudaDeviceSynchronize();

		void* kernelArgs1[] = { &_activeList, &_maskedKeys, &len };
		cudaLaunchCooperativeKernel(k_radixSortByKey<int, uint64_t>, dim3(activeBlocks), dim3(threadsPerBlock), kernelArgs1);
		cudaDeviceSynchronize();

		setHeadFlags<<<activeBlocks, threadsPerBlock>>>(_maskedKeys, len, _headFlags);
		cudaDeviceSynchronize();

		void* kernelArgs2[] = { &_headFlags, &len, &_groupStarts, &_numGroups };
		cudaLaunchCooperativeKernel(k_compactIndices<int, int>, dim3(activeBlocks), dim3(threadsPerBlock), kernelArgs2);
		cudaDeviceSynchronize();

		int* groupSizes = _headFlags;
		getGroupSizes<<<activeBlocks, threadsPerBlock>>>(_groupStarts, _numGroups, len, groupSizes);
		cudaDeviceSynchronize();

		classifyGroups<<<activeBlocks, threadsPerBlock>>>(_activeList, _groupStarts, groupSizes, _numGroups, _maskedKeys, level, NLeaf, _flagged, _cells, _cellCount, _leafParticles, _leafParticleCount);
		cudaDeviceSynchronize();
		
		setCompactFlags<<<activeBlocks, threadsPerBlock>>>(_activeList, len, _flagged, _flaggedTemp);
		cudaDeviceSynchronize();

		void* kernelArgs4[] = { &_activeList, &_flaggedTemp, &len, &_newLen };
		cudaLaunchCooperativeKernel(k_compact<int, int>, dim3(activeBlocks), dim3(threadsPerBlock), kernelArgs4);
		cudaDeviceSynchronize();
		
		cudaMemcpy(&len, _newLen, sizeof(int), cudaMemcpyDeviceToHost);

		level++;
		maxLevel = level;
	}

	int cellCount;
	cudaMemcpy(&cellCount, _cellCount, sizeof(int), cudaMemcpyDeviceToHost);
	int cellBlocks = cuda::ceil_div(cellCount, threadsPerBlock);
	extractCellKeys<<<cellBlocks, threadsPerBlock>>>(_cells, cellCount, _maskedKeys);
	cudaDeviceSynchronize();

	void* kernelArgs1[] = { &_cells, &_maskedKeys, &cellCount };
	cudaLaunchCooperativeKernel(k_radixSortByKey<Cell, uint64_t>, dim3(cellBlocks), dim3(threadsPerBlock), kernelArgs1);
	cudaDeviceSynchronize();

	linkCellsToParents<<<cellBlocks, threadsPerBlock>>>(_cells, cellCount);
	cudaDeviceSynchronize();

	cudaEventRecord(endPartition);

	cudaEvent_t movementStart, movementEnd;
	cudaEventCreate(&movementStart);
	cudaEventCreate(&movementEnd);

	cudaEventRecord(movementStart);

	int hLeafCount = 0;
	cudaMemset(_leafCount, 0, sizeof(int));

	extractLeafIndices<<<cellBlocks, threadsPerBlock>>>(_cells, cellCount, _leafIndices, _leafCount);
	cudaDeviceSynchronize();

	cudaMemcpy(&hLeafCount, _leafCount, sizeof(int), cudaMemcpyDeviceToHost);

	setLeafMoments<<<cellBlocks, threadsPerBlock>>>(_cells, cellCount, _d_particleInfos, _leafParticles);
	cudaDeviceSynchronize();

	for (int level = maxLevel - 1; level >= 0; level--)
	{
		setNodeMoments<<<cellBlocks, threadsPerBlock>>>(_cells, cellCount, level);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(_accelerationsPrev, _accelerations, _bodyCount * sizeof(float3), cudaMemcpyDefault);

	const float theta = 0.1f;
	cudaMemset(_accelerations, 0, _bodyCount * sizeof(float3));
	computeVelocities<<<hLeafCount, threadsPerBlock>>>(_cells, cellCount, _leafIndices, _d_particleInfos, _leafParticles, _accelerations, theta, domainMin, domainMax);
	cudaDeviceSynchronize();

	cudaEventRecord(movementEnd);

	cudaEventSynchronize(endMorton);
	cudaEventSynchronize(endPartition);

	float timeMorton;
	cudaEventElapsedTime(&timeMorton, startMorton, endMorton);
	std::cout << "\x1b[4A" << std::flush;
	std::cout << "Morton keys and sort time: " << timeMorton << " ms\x1b[K\n" << std::flush;

	float timePartition;
	cudaEventElapsedTime(&timePartition, startPartition, endPartition);
	std::cout << "Partition time: " << timePartition << " ms\x1b[K\n" << std::flush;

	float timeMovement;
	cudaEventElapsedTime(&timeMovement, movementStart, movementEnd);
	std::cout << "Movement time: " << timeMovement << " ms\x1b[K\n" << std::flush;
}
