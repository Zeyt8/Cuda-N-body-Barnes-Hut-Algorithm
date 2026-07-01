#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ static uint32_t dilate10(const uint32_t value)
{
	uint32_t x;
	x = value & 0x03FF;
	x = ((x << 16) + x) & 0xFF0000FF;
	x = ((x << 8) + x) & 0x0F00F00F;
	x = ((x << 4) + x) & 0xC30C30C3;
	x = ((x << 2) + x) & 0x49249249;
	return x;
}

__device__ static uint64_t dilate20(const uint32_t value)
{
	uint32_t lo = value & 0x3FF;
	uint32_t hi = (value >> 10) & 0x3FF;

	uint64_t dlo = dilate10(lo);
	uint64_t dhi = dilate10(hi);

	return dlo | (dhi << 30);
}

__host__ __device__ static uint32_t undilate10(uint32_t x)
{
	x &= 0x49249249;
	x = (x | (x >> 2)) & 0xC30C30C3;
	x = (x | (x >> 4)) & 0x0F00F00F;
	x = (x | (x >> 8)) & 0xFF0000FF;
	x = (x | (x >> 16)) & 0x000003FF;
	return x;
}

__host__ __device__ static void decodeMortonKey(uint64_t key, int level, uint32_t& ix, uint32_t& iy, uint32_t& iz)
{
	int bits = 3 * (level + 1);
	uint64_t n = key >> (60 - bits);

	uint32_t xlo = undilate10((uint32_t)(n & 0x3FFFFFFF));
	uint32_t xhi = undilate10((uint32_t)((n >> 30) & 0x3FFFFFFF));
	ix = xlo | (xhi << 10);

	uint32_t ylo = undilate10((uint32_t)((n >> 1) & 0x3FFFFFFF));
	uint32_t yhi = undilate10((uint32_t)((n >> 31) & 0x3FFFFFFF));
	iy = ylo | (yhi << 10);

	uint32_t zlo = undilate10((uint32_t)((n >> 2) & 0x3FFFFFFF));
	uint32_t zhi = undilate10((uint32_t)((n >> 32) & 0x3FFFFFFF));
	iz = zlo | (zhi << 10);
}

template<typename Key>
__device__ void prefixSum(Key* values, const int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	cg::grid_group g = cg::this_grid();

	for (int i = 1; i < len; i *= 2)
	{
		Key b = 0;
		if (idx < len && idx >= i)
		{
			b = values[idx - i];
		}
		g.sync();
		if (idx < len && idx >= i)
		{
			values[idx] += b;
		}
		g.sync();
	}
}

template<typename T, typename Key>
__device__ void splitAndSort(T* __restrict__ values, Key* __restrict__ keys, const int len, Key keyBefore)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	bool active = idx < len;
	cg::grid_group g = cg::this_grid();

	int b_i;
	T valueBefore;
	if (active)
	{
		b_i = keys[idx];
		valueBefore = values[idx];
	}
	g.sync();

	prefixSum(keys, len);

	int zeroTotal;
	int oneBefore;
	if (active)
	{
		zeroTotal = len - keys[len - 1];
		oneBefore = keys[idx];
	}
	g.sync();

	if (active)
	{
		if (b_i)
		{
			values[zeroTotal + oneBefore - 1] = valueBefore;
			keys[zeroTotal + oneBefore - 1] = keyBefore;
		}
		else
		{
			values[idx - oneBefore] = valueBefore;
			keys[idx - oneBefore] = keyBefore;
		}
	}
	g.sync();
}

template<typename T, typename Key>
__global__ void k_splitAndSort(T* __restrict__ values, Key* __restrict__ keys, const int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	cg::grid_group g = cg::this_grid();

	Key keyBefore = 0;
	if (idx < len)
	{
		keyBefore = keys[idx];
	}
	splitAndSort(values, keys, len, keyBefore);
}

template<typename T, typename Key>
__device__ void compact(T* __restrict__ values, Key* __restrict__ keys, const int len, Key keyBefore, int* newLen)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	bool active = idx < len;
	cg::grid_group g = cg::this_grid();

	int b_i;
	T valueBefore;
	if (active)
	{
		b_i = keys[idx];
		valueBefore = values[idx];
	}
	g.sync();

	prefixSum(keys, len);

	int oneBefore;
	int total;
	if (active)
	{
		oneBefore = keys[idx];
		total = keys[len - 1];
	}
	g.sync();

	if (active)
	{
		if (b_i)
		{
			values[oneBefore - 1] = valueBefore;
			keys[oneBefore - 1] = keyBefore;
		}
	}
	g.sync();

	if (idx == 0)
	{
		*newLen = total;
	}
}

template<typename T, typename Key>
__global__ void k_compact(T* __restrict__ values, Key* __restrict__ keys, const int len, int* newLen)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	cg::grid_group g = cg::this_grid();

	Key keyBefore = 0;
	if (idx < len)
	{
		keyBefore = keys[idx];
	}
	compact(values, keys, len, keyBefore, newLen);
}

template<typename T, typename Key>
__device__ void compactIndices(Key* __restrict__ keys, const int len, Key keyBefore, T* __restrict__ values, int* newLen)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	bool active = idx < len;
	cg::grid_group g = cg::this_grid();

	int b_i;
	if (active)
	{
		b_i = keys[idx];
	}
	g.sync();

	prefixSum(keys, len);

	int oneBefore;
	int total;
	if (active)
	{
		oneBefore = keys[idx];
		total = keys[len - 1];
	}
	g.sync();

	if (active)
	{
		if (b_i)
		{
			values[oneBefore - 1] = idx;
			keys[oneBefore - 1] = keyBefore;
		}
	}
	g.sync();

	if (idx == 0)
	{
		*newLen = total;
	}
}

template<typename T, typename Key>
__global__ void k_compactIndices(Key* __restrict__ keys, const int len, T* __restrict__ values, int* newLen)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	cg::grid_group g = cg::this_grid();

	Key keyBefore = 0;
	if (idx < len)
	{
		keyBefore = keys[idx];
	}
	compactIndices(keys, len, keyBefore, values, newLen);
}

template<typename T, typename Key>
__global__ void k_radixSortByKey(T* __restrict__ values, Key* __restrict__ keys, const int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	bool active = idx < len;
	cg::grid_group g = cg::this_grid();

	for (int bit = 0; bit < sizeof(keys[0]) * 8; bit++)
	{
		Key keyBefore = 0;
		if (active)
		{
			keyBefore = keys[idx];
			keys[idx] = (keyBefore >> bit) & 1;
		}
		g.sync();

		splitAndSort(values, keys, len, keyBefore);
		g.sync();
	}
}