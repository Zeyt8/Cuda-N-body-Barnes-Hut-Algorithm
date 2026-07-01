#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <climits>

#include "Cell.h"
#include "simulation/utils.h"
#include "utils/float3_helpers.cuh"

__constant__ float epsilon = 1.0f;

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

__global__ void setLeafMoments(Cell* __restrict__ cells, const int cellCount, const float4* __restrict__ particleInfos, const int* __restrict__ leafParticles)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= cellCount) return;

	Cell cell = cells[idx];
	if (cell.type != LEAF) return;

	double M = 0;
	double3 com = make_double3(0, 0, 0);
	for (int i = 0; i < cell.count; i++)
	{
		int particle = leafParticles[cell.start + i];
		float4 info = particleInfos[particle];
		double mass = info.w;
		M += mass;
		com.x += mass * info.x;
		com.y += mass * info.y;
		com.z += mass * info.z;
	}
	com.x /= M;
	com.y /= M;
	com.z /= M;

	cell.mass = M;
	cell.com = com;

	double Qxx = 0, Qyy = 0, Qxy = 0, Qxz = 0, Qyz = 0;
	for (int i = 0; i < cell.count; i++)
	{
		int particle = leafParticles[cell.start + i];
		float4 info = particleInfos[particle];
		double mass = info.w;
		double dx = info.x - com.x;
		double dy = info.y - com.y;
		double dz = info.z - com.z;
		double r2 = dx * dx + dy * dy + dz * dz;

		Qxx += mass * (3 * dx * dx - r2);
		Qyy += mass * (3 * dy * dy - r2);
		Qxy += mass * (3 * dx * dy);
		Qxz += mass * (3 * dx * dz);
		Qyz += mass * (3 * dy * dz);
	}

	cell.Qxx = Qxx;
	cell.Qyy = Qyy;
	cell.Qxy = Qxy;
	cell.Qxz = Qxz;
	cell.Qyz = Qyz;

	cells[idx] = cell;
}

__global__ void setNodeMoments(Cell* __restrict__ cells, const int cellCount, const int level)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= cellCount) return;

	Cell cell = cells[idx];

	if (cell.type != NODE) return;
	if (cell.level != level) return;

	double M = 0;
	double3 com = make_double3(0, 0, 0);
	for (int i = 0; i < cell.count; i++)
	{
		Cell child = cells[cell.start + i];
		M += child.mass;
		com.x += child.mass * child.com.x;
		com.y += child.mass * child.com.y;
		com.z += child.mass * child.com.z;
	}
	com.x /= M;
	com.y /= M;
	com.z /= M;

	cell.mass = M;
	cell.com = com;

	double Qxx = 0, Qyy = 0, Qxy = 0, Qxz = 0, Qyz = 0;
	for (int i = 0; i < cell.count; i++)
	{
		Cell child = cells[cell.start + i];
		double dx = child.com.x - com.x;
		double dy = child.com.y - com.y;
		double dz = child.com.z - com.z;
		double r2 = dx * dx + dy * dy + dz * dz;

		Qxx += child.Qxx + child.mass * (3 * dx * dx - r2);
		Qyy += child.Qyy + child.mass * (3 * dy * dy - r2);
		Qxy += child.Qxy + child.mass * (3 * dx * dy);
		Qxz += child.Qxz + child.mass * (3 * dx * dz);
		Qyz += child.Qyz + child.mass * (3 * dy * dz);
	}

	cell.Qxx = Qxx;
	cell.Qyy = Qyy;
	cell.Qxy = Qxy;
	cell.Qxz = Qxz;
	cell.Qyz = Qyz;

	cells[idx] = cell;
}

__global__ void extractLeafIndices(const Cell* __restrict__ cells, const int cellCount, int* __restrict__ leafIndices, int* __restrict__ leafCount)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= cellCount) return;
	if (cells[idx].type != LEAF) return;

	int slot = atomicAdd(leafCount, 1);
	leafIndices[slot] = idx;
}

__global__ void findRoot(const Cell* cells, int cellCount, int* rootIndex)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= cellCount) return;
	if (cells[idx].level == 0)
	{
		*rootIndex= idx;
	}
}

__device__ float3 monopoleForce(float3 particlePos, float3 cellCom, float cellMass)
{
	float3 r = cellCom - particlePos;
	float r2 = lengthSquared(r) + epsilon * epsilon;
	float r1 = sqrtf(r2);
	float r3 = r2 * r1;
	return make_float3(cellMass * r.x / r3,
					   cellMass * r.y / r3,
					   cellMass * r.z / r3);
}

__device__ void flushCellList(const Cell* __restrict__ cells, const float4* __restrict__ groupParticles, const int* __restrict__ cellList,
	int& cellListSize, float3* __restrict__ myAcc, const int groupCount)
{
	__shared__ float4 cellData[256];

	if (threadIdx.x < cellListSize)
	{
		Cell c = cells[cellList[threadIdx.x]];
		cellData[threadIdx.x] = make_float4(c.com.x, c.com.y, c.com.z, c.mass);
	}
	__syncthreads();

	if (threadIdx.x < groupCount)
	{
		float3 pos = make_float3(groupParticles[threadIdx.x].x,
								 groupParticles[threadIdx.x].y,
								 groupParticles[threadIdx.x].z);
		int count = min(cellListSize, blockDim.x);
		for (int i = 0; i < count; i++)
		{
			myAcc->x += monopoleForce(pos, make_float3(cellData[i].x, cellData[i].y, cellData[i].z), cellData[i].w).x;
			myAcc->y += monopoleForce(pos, make_float3(cellData[i].x, cellData[i].y, cellData[i].z), cellData[i].w).y;
			myAcc->z += monopoleForce(pos, make_float3(cellData[i].x, cellData[i].y, cellData[i].z), cellData[i].w).z;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
		cellListSize = 0;
	}
	__syncthreads();
}

__device__ void flushParticleList(const float4* __restrict__ particleInfos, const int* __restrict__ leafParticles, const float4* __restrict__ groupParticles,
	const int* __restrict__ particleList, int& particleListSize, float3* __restrict__ myAcc, const int groupCount)
{
	__shared__ float4 particleData[256];

	if (threadIdx.x < particleListSize)
	{
		int p = leafParticles[particleList[threadIdx.x]];
		particleData[threadIdx.x] = particleInfos[p];
	}
	__syncthreads();

	if (threadIdx.x < groupCount)
	{
		float3 pos = make_float3(groupParticles[threadIdx.x].x,
								 groupParticles[threadIdx.x].y,
								 groupParticles[threadIdx.x].z);
		int count = min(particleListSize, blockDim.x);
		for (int i = 0; i < count; i++)
		{
			float3 otherPos = make_float3(particleData[i].x, particleData[i].y, particleData[i].z);
			float3 r = make_float3(otherPos.x - pos.x, otherPos.y - pos.y, otherPos.z - pos.z);
			float r2 = r.x * r.x + r.y * r.y + r.z * r.z;
			if (r2 < epsilon * epsilon) continue;
			r2 += epsilon * epsilon;
			float r1 = sqrtf(r2);
			float r3 = r2 * r1;
			float mass = particleData[i].w;
			myAcc->x += mass * r.x / r3;
			myAcc->y += mass * r.y / r3;
			myAcc->z += mass * r.z / r3;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
		particleListSize = 0;
	}
	__syncthreads();
}

__global__ void computeVelocities(const Cell* __restrict__ cells, const int cellCount, const int* __restrict__ leafCellIndices,
	const float4* __restrict__ particleInfos, const int* __restrict__ leafParticles, float3* __restrict__ accelerations, const float theta, const float domainMin, const float domainMax)
{
	int leafIdx = leafCellIndices[blockIdx.x];
	Cell group = cells[leafIdx];

	__shared__ float4 groupParticles[64];
	__shared__ int currentStack[256];
	__shared__ int nextStack[256];
	__shared__ int currentStackSize;
	__shared__ int nextStackSize;
	__shared__ int cellList[256];
	__shared__ int particleList[256];
	__shared__ int cellListSize;
	__shared__ int particleListSize;
	__shared__ float3 groupBBoxMin;
	__shared__ float3 groupBBoxMax;

	float3 myAcc = make_float3(0, 0, 0);

	if (threadIdx.x == 0)
	{
		currentStackSize = 0;
		nextStackSize = 0;
		cellListSize = 0;
		particleListSize = 0;

		for (int i = 0; i < cellCount; i++)
		{
			if (cells[i].level == 0)
			{
				currentStack[currentStackSize++] = i;
			}
		}

		float cellSize = (domainMax - domainMin) / (float)(1 << (group.level + 1));
		uint32_t ix, iy, iz;
		decodeMortonKey(group.key, group.level, ix, iy, iz);
		groupBBoxMin = make_float3(domainMin + ix * cellSize,
								   domainMin + iy * cellSize,
								   domainMin + iz * cellSize);
		groupBBoxMax = make_float3(groupBBoxMin.x + cellSize,
								   groupBBoxMin.y + cellSize,
								   groupBBoxMin.z + cellSize);
	}
	__syncthreads();

	if (threadIdx.x < group.count)
	{
		int particle = leafParticles[group.start + threadIdx.x];
		groupParticles[threadIdx.x] = particleInfos[particle];
	}
	__syncthreads();

	while (currentStackSize > 0)
	{
		for (int base = 0; base < currentStackSize; base += blockDim.x)
		{
			int stackIdx = base + threadIdx.x;
			if (stackIdx < currentStackSize)
			{
				int cellIdx = currentStack[stackIdx];
				Cell cell = cells[cellIdx];

				float cellSize = (domainMax - domainMin) / (float)(1 << (cell.level + 1));
				uint32_t ix, iy, iz;
				decodeMortonKey(cell.key, cell.level, ix, iy, iz);
				float3 geomCenter = make_float3(domainMin + ix * cellSize + cellSize * 0.5f,
												domainMin + iy * cellSize + cellSize * 0.5f,
												domainMin + iz * cellSize + cellSize * 0.5f);

				float3 com = make_float3(cell.com.x, cell.com.y, cell.com.z);
				float delta = length(geomCenter - com);

				float3 closest = make_float3(fmaxf(groupBBoxMin.x, fminf(com.x, groupBBoxMax.x)),
											 fmaxf(groupBBoxMin.y, fminf(com.y, groupBBoxMax.y)),
											 fmaxf(groupBBoxMin.z, fminf(com.z, groupBBoxMax.z)));
				float d = length(closest - com);

				bool accept = d > (cellSize / theta + delta);

				if (accept)
				{
					int slot = atomicAdd(&cellListSize, 1);
					if (slot < blockDim.x)
					{
						cellList[slot] = cellIdx;
					}
				}
				else if (cell.type == LEAF)
				{
					for (int i = 0; i < cell.count; i++)
					{
						int slot = atomicAdd(&particleListSize, 1);
						if (slot < blockDim.x)
						{
							particleList[slot] = cell.start + i;
						}
					}
				}
				else
				{
					for (int i = 0; i < cell.count; i++)
					{
						int slot = atomicAdd(&nextStackSize, 1);
						if (slot < 256)
						{
							nextStack[slot] = cell.start + i;
						}
					}
				}
			}
			__syncthreads();

			if (cellListSize >= blockDim.x)
			{
				flushCellList(cells, groupParticles, cellList, cellListSize, &myAcc, group.count);
			}

			if (particleListSize >= blockDim.x)
			{
				flushParticleList(particleInfos, leafParticles, groupParticles, particleList, particleListSize, &myAcc, group.count);
			}
		}

		if (threadIdx.x == 0)
		{
			currentStackSize = nextStackSize;
			nextStackSize = 0;
			for (int i = 0; i < currentStackSize; i++)
				currentStack[i] = nextStack[i];
		}
		__syncthreads();
	}

	if (cellListSize > 0)
	{
		flushCellList(cells, groupParticles, cellList, cellListSize, &myAcc, group.count);
	}
	if (particleListSize > 0)
	{
		flushParticleList(particleInfos, leafParticles, groupParticles, particleList, particleListSize, &myAcc, group.count);
	}

	if (threadIdx.x < group.count)
	{
		int particle = leafParticles[group.start + threadIdx.x];
		accelerations[particle] = myAcc;
	}
}

__global__ void movePos(float4* __restrict__ particles, const int particleCount, const float3* __restrict__ velocities, const float3* __restrict__ accelerations, const float deltaTime)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= particleCount) return;

	float3 posChange = velocities[idx] * deltaTime + accelerations[idx] * 0.5f * deltaTime * deltaTime;
	float3 velChange = accelerations[idx] * deltaTime;

	particles[idx].x += posChange.x;
	particles[idx].y += posChange.y;
	particles[idx].z += posChange.z;
}

__global__ void correctVelocities(float3* __restrict__ velocities, const int count, const float3* __restrict__ accsNew, const float3* __restrict__ accsOld, const float deltaTime)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= count) return;

	velocities[idx] = velocities[idx] + (accsNew[idx] + accsOld[idx]) * 0.5f * deltaTime;
}
