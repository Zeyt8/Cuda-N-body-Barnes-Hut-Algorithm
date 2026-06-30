#pragma once

#include <cuda_runtime.h>

struct Camera
{
	float3 pos;
	float3 forward;
	float3 up;
	float focalLength;
};