#include "SimDraw.h"

#include <cuda/cmath>

#include <chrono>
#include <iostream>

#include "utils/float3_helpers.cuh"

#define PI 3.1415926535897932385f

__device__ static inline int idx(int x, int y, int width)
{
	return y * width + x;
}

__global__ static void setPixelCenterAndDir(int width, int height, float3 pixel100Loc, float3 pixelDeltaU, float3 pixelDeltaV, float3 cameraCenter, float3* rayDirs)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	float3 pixelCenter = pixel100Loc + (pixelDeltaU * x) + (pixelDeltaV * y);
	float3 rayDir = pixelCenter - cameraCenter;

	rayDirs[idx(x, y, width)] = rayDir;
}

SimDraw::SimDraw(int width, int height, Camera camera, float4* bodyInfos, int bodyCount, Cell* cells, int* cellCount)
{
	_width = width;
	_height = height;
	_camera = camera;
	_bodyInfos = bodyInfos;
	_bodyCount = bodyCount;
	_cells = cells;
	_cellCount = cellCount;

	float viewportHeight = 1;
	float viewportWidth = viewportHeight * ((float)width / height);
	float3 w = normalized(_camera.forward);
	float3 u = normalized(cross(_camera.up, w));
	float3 v = cross(w, u);
	float3 viewportU = u * viewportWidth;
	float3 viewportV = v * viewportHeight;

	_pixelDeltaU = viewportU / (float)width;
	_pixelDeltaV = viewportV / (float)height;

	float3 viewportUpperLeft = camera.pos + w * camera.focalLength - (viewportU + viewportV) / 2;
	float3 pixel00Loc = viewportUpperLeft + (_pixelDeltaU + _pixelDeltaV) / 2;

	cudaMalloc(&_rayDirs, width * height * sizeof(float3));

	dim3 block(32, 32);
	dim3 grid(cuda::ceil_div(width, block.x), cuda::ceil_div(height, block.y));
	setPixelCenterAndDir<<<grid, block>>>(width, height, pixel00Loc, _pixelDeltaU, _pixelDeltaV, camera.pos, _rayDirs);
	cudaDeviceSynchronize();
}

__host__ __device__ static inline uint32_t getRandom(uint32_t& state)
{
	state ^= state >> 16;
	state *= 0x21F0AAADu;
	state ^= state >> 15;
	state *= 0xD35A2D97u;
	state ^= state >> 15;
	return state;
}

__device__ static float hitBody(const float4& body, const float3& rayCenter, const float3& rayDir) {
	float3 center = make_float3(body.x, body.y, body.z);
	float radius = cbrtf(body.w * 3 / 4 / PI);

	float3 oc = center - rayCenter;
	float a = lengthSquared(rayDir);
	float h = dot(rayDir, oc);
	float c = lengthSquared(oc) - radius * radius;
	float discriminant = h * h - a * c;

	return (discriminant < 0) * -1 + (discriminant >= 0) * (h - sqrtf(discriminant)) / a;
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

__device__ static float hitWireBox(float3 boxMin, float3 boxMax, float3 rayOrigin, float3 rayDir, float edgeThickness)
{
	// box intersection
	float3 invDir = make_float3(1.0f / rayDir.x, 1.0f / rayDir.y, 1.0f / rayDir.z);

	float3 t0 = (boxMin - rayOrigin) * invDir;
	float3 t1 = (boxMax - rayOrigin) * invDir;

	float3 tMin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
	float3 tMax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

	float tClose = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
	float tFar = fminf(fminf(tMax.x, tMax.y), tMax.z);

	if (tFar < tClose || tFar < 0.0f || tClose < 0.0f) return -1.0f;

	// only wire
	float3 size = boxMax - boxMin;

	float tArr[2] = { tClose, tFar };
	for (int i = 0; i < 2; i++)
	{
		float3 hit = rayOrigin + rayDir * tArr[i];
	
		int nearFace = 0;
		nearFace += (hit.x < boxMin.x + edgeThickness || hit.x > boxMax.x - edgeThickness) ? 1 : 0;
		nearFace += (hit.y < boxMin.y + edgeThickness || hit.y > boxMax.y - edgeThickness) ? 1 : 0;
		nearFace += (hit.z < boxMin.z + edgeThickness || hit.z > boxMax.z - edgeThickness) ? 1 : 0;
	
		if (nearFace >= 2) return tArr[i];
	}
	
	return -1.0f;
}

__global__ static void raytrace(const float3 rayOrigin, const float3* __restrict__ rayDirs, const int width, const int height, const float3 pixelDU, const float3 pixelDV,
	const float4* __restrict__ bodyInfos, const int bodyCount, const Cell* __restrict__ cells, const int* __restrict__  cellCount,
	const float domainMin, const float domainMax, const uint32_t initState, uchar4* __restrict__ buffer)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	uint32_t state = initState + x + y * width;
	float3 initialDir = rayDirs[idx(x, y, width)];

	float3 color = make_float3(0, 0, 0);
	float3 backgroundColor = make_float3(0, 0, 0);
	int sampleCount = 10;
	for (int sample = 0; sample < sampleCount; sample++)
	{
		float3 dir = initialDir;
		dir.x += ((float)getRandom(state) / UINT32_MAX - 0.5f) * pixelDU.x;
		dir.y += ((float)getRandom(state) / UINT32_MAX - 0.5f) * pixelDV.y;

		for (int i = 0; i < *cellCount; i++)
		{
			uint32_t ix, iy, iz;
			decodeMortonKey(cells[i].key, cells[i].level, ix, iy, iz);

			float cellSize = (domainMax - domainMin) / (float)(1 << (cells[i].level + 1));

			float3 boxMin = make_float3(domainMin + ix * cellSize,
				/*domainMin + iy * cellSize*/-1,
				domainMin + iz * cellSize);
			float3 boxMax = make_float3(boxMin.x + cellSize,
				/*boxMin.y + cellSize*/ 1,
				boxMin.z + cellSize);

			float p = hitWireBox(boxMin, boxMax, rayOrigin, dir, 2.0f);
			if (p >= 0)
			{
				float3 wireColor = (cells[i].type == LEAF)
					? make_float3(0, 1, 0)
					: make_float3(1, 1, 0);
				color = color + wireColor / sampleCount;
				break;
			}
		}
	
		for (int i = 0; i < bodyCount; i++)
		{
		    float p = hitBody(bodyInfos[i], rayOrigin, dir);
		    if (p >= 0)
		    {
		        color = color + make_float3(1, 1, 1) / sampleCount;
				break;
		    }
		}
	}

	buffer[idx(x, y, width)] = uchar4(color.x * 255, color.y * 255, color.z * 255, 255);
}

void SimDraw::Render(uchar4* pbo)
{
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	dim3 block(32, 32);
	dim3 grid(cuda::ceil_div(_width, block.x), cuda::ceil_div(_height, block.y));
	uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	raytrace<<<grid, block>>>(_camera.pos, _rayDirs, _width, _height, _pixelDeltaU, _pixelDeltaV, _bodyInfos, _bodyCount, _cells, _cellCount, 0, 1000, ms, pbo);
	cudaDeviceSynchronize();

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float time;
	cudaEventElapsedTime(&time, start, end);
	std::cout << "Rendering time: " << time << " ms\x1b[K\n" << std::flush;
}
