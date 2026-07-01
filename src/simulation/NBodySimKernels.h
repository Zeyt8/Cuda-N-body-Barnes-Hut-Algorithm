#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <climits>

#include "Cell.h"

__device__ uint32_t dilate10(const uint32_t value) {
	uint32_t x;
	x = value & 0x03FF;
	x = ((x << 16) + x) & 0xFF0000FF;
	x = ((x << 8) + x) & 0x0F00F00F;
	x = ((x << 4) + x) & 0xC30C30C3;
	x = ((x << 2) + x) & 0x49249249;
	return x;
}

__device__ uint64_t dilate20(const uint32_t value) {
	uint32_t lo = value & 0x3FF;
	uint32_t hi = (value >> 10) & 0x3FF;

	uint64_t dlo = dilate10(lo);
	uint64_t dhi = dilate10(hi);

	return dlo | (dhi << 30);
}

__global__ void computeMortonKeys(const float4* __restrict__ values, const int len, const float domainMin, const float domainMax, uint64_t* __restrict__ keys)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	float3 v = make_float3(values[idx].x, values[idx].y, values[idx].z);
	v.x -= domainMin;
	v.y -= domainMin;
	v.z -= domainMin;

	float scale = (float)(1 << 20) / (domainMax - domainMin);

	uint64_t dx = dilate20((uint32_t)(v.x * scale));
	uint64_t dy = dilate20((uint32_t)(v.y * scale));
	uint64_t dz = dilate20((uint32_t)(v.z * scale));

	uint64_t key = dx | (dy << 1) | (dz << 2);

	keys[idx] = key;
}

__global__ void initActiveList(int* list, const int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	list[idx] = idx;
}

__global__ void getMaskedValues(const uint64_t* __restrict__ keys, const int* __restrict__ activeList, const int len, const int level, uint64_t* __restrict__ maskedKeys)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	int bits = 3 * (level + 1);
	uint64_t bitMask = (uint64_t{1} << bits) - 1;
	int shift = 60 - bits;
	maskedKeys[idx] = keys[activeList[idx]] & (bitMask << shift);
}

__global__ void setHeadFlags(const uint64_t* __restrict__ maskedKeys, const int len, int* __restrict__ headFlags)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	headFlags[idx] = (idx == 0) || (maskedKeys[idx - 1] != maskedKeys[idx]);
}

__global__ void getGroupSizes(const int* __restrict__ groupStarts, const int* numGroups, const int len, int* __restrict__ groupSizes)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= *numGroups) return;

	groupSizes[idx] = ((idx == *numGroups - 1) ? len : groupStarts[idx + 1]) - groupStarts[idx];
}

__global__ void classifyGroups(const int* __restrict__ activeList, const int* __restrict__ groupStarts, const int* __restrict__ groupSizes, const int* numGroups,
								const uint64_t* __restrict__ maskedKeys, int level, int NLeaf, bool* __restrict__ flagged,
								Cell* __restrict__ cells, int* __restrict__ cellCount, int* __restrict__ leafParticles, int* __restrict__ leafParticleCount)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= *numGroups) return;

	int start = groupStarts[idx];
	int count = groupSizes[idx];

	uint64_t key = maskedKeys[start];
	int cellSlot = atomicAdd(cellCount, 1);
	Cell cell;
	cell.key = key;
	cell.level = level;
	cell.count = 0;
	cell.start = INT_MAX;

	if (count <= NLeaf)
	{
		cell.type = LEAF;
		int leafStart = atomicAdd(leafParticleCount, count);
		cell.start = leafStart;
		cell.count = count;

		for (int i = 0; i < count; i++)
		{
			int particle = activeList[start + i];
			leafParticles[leafStart + i] = particle;
			flagged[particle] = true;
		}
	}
	else
	{
		cell.type = NODE;
	}

	cells[cellSlot] = cell;
}

__global__ void setCompactFlags(const int* __restrict__ activeList, const int len, const bool* __restrict__ flagged, int* __restrict__ out)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	out[idx] = !flagged[activeList[idx]];
}

__global__ void extractCellKeys(const Cell* __restrict__ cells, const int len, uint64_t* __restrict__ keys)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= len) return;

	keys[idx] = cells[idx].key;
}

__device__ int binarySearchCells(const Cell* __restrict__ cells, const int totalCells, const uint64_t targetKey, const int targetLevel)
{
	int lo = 0;
	int hi = totalCells - 1;

	while (lo <= hi)
	{
		int mid = (lo + hi) / 2;

		if (cells[mid].key == targetKey && cells[mid].level == targetLevel)
		{
			return mid;
		}

		if (cells[mid].key < targetKey)
		{
			lo = mid + 1;
		}
		else
		{
			hi = mid - 1;
		}
	}

	return -1;
}

__global__ void linkCellsToParents(Cell* __restrict__ cells, int cellCount)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= cellCount) return;

	Cell cell = cells[idx];

	if (cell.level == 0)
	{
		return;
	}

	int parentLevel = cell.level - 1;
	int parentBits = 3 * (parentLevel + 1);
	uint64_t parentMask = ((1ULL << parentBits) - 1ULL) << (60 - parentBits);
	uint64_t parentKey = cell.key & parentMask;

	int parentIdx = binarySearchCells(cells, cellCount, parentKey, parentLevel);

	atomicAdd(&cells[parentIdx].count, 1);
	atomicMin(&cells[parentIdx].start, idx);
}
