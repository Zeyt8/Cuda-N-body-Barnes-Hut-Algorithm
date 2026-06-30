#pragma once

#include <cuda_runtime.h>

#include "Camera.h"

class SimDraw
{
public:
	SimDraw(int width, int height, Camera cam, float4* bodyInfos, int bodyCount);
	void Render(uchar4* pbo);

private:
	int _width;
	int _height;
	Camera _camera;
	float4* _bodyInfos;
	int _bodyCount;
	float3 _pixelDeltaU;
	float3 _pixelDeltaV;
	float3* _rayDirs;
};