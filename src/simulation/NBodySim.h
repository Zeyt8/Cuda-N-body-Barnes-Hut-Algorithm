#pragma once

#include <cuda_runtime.h>

class NBodySim
{
public:
	NBodySim(int bodyCount);
	~NBodySim();
	void Simulate();
	float4* GetBodyInfos() { return _d_particleInfos; }

private:
	int _bodyCount = 0;
	float4* _h_particleInfos = nullptr;
	float4* _d_particleInfos = nullptr;
	uint64_t* _keys = nullptr;
	bool* _flagged = nullptr;
	int* _activeList = nullptr;
	uint64_t* _maskedKeys = nullptr;
	int* _headFlags = nullptr;
	int* _groupStarts = nullptr;
};