#include "SimDraw.h"

#include <cuda/cmath>

#include <chrono>

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

SimDraw::SimDraw(int width, int height, Camera camera, float4* bodyInfos, int bodyCount)
{
	_width = width;
	_height = height;
	_camera = camera;
	_bodyInfos = bodyInfos;
	_bodyCount = bodyCount;

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

__global__ static void raytrace(const float3 rayOrigin, const float3* __restrict__ rayDirs, const int width, const int height, const float3 pixelDU, const float3 pixelDV,
	const float4* __restrict__ bodyInfos, const int bodyCount, const uint32_t initState, uchar4* __restrict__ buffer)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	uint32_t state = initState + x + y * width;
	float3 initialDir = rayDirs[idx(x, y, width)];

	float3 color = make_float3(0, 0, 0);
	int sampleCount = 10;
	for (int sample = 0; sample < sampleCount; sample++)
	{
		float3 dir = initialDir;
		dir.x += ((float)getRandom(state) / UINT32_MAX - 0.5f) * pixelDU.x;
		dir.y += ((float)getRandom(state) / UINT32_MAX - 0.5f) * pixelDV.y;
	
		for (int i = 0; i < bodyCount; i++)
		{
		    float p = hitBody(bodyInfos[i], rayOrigin, dir);
		    if (p >= 0)
		    {
		        color = color + make_float3(1, 1, 1) / sampleCount;
		    }
		    else
		    {
		        color = color + make_float3(0, 0, 0) / sampleCount;
		    }
		}
	}

	buffer[idx(x, y, width)] = uchar4(color.x * 255, color.y * 255, color.z * 255, 255);
}

void SimDraw::Render(uchar4* pbo)
{
	dim3 block(32, 32);
	dim3 grid(cuda::ceil_div(_width, block.x), cuda::ceil_div(_height, block.y));
	uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	raytrace<<<grid, block>>>(_camera.pos, _rayDirs, _width, _height, _pixelDeltaU, _pixelDeltaV, _bodyInfos, _bodyCount, ms, pbo);
	cudaDeviceSynchronize();
}
