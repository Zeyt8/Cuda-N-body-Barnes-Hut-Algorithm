#pragma once

#include <cuda_runtime.h>

#include "Cell.h"

class NBodySim
{
public:
	NBodySim(int bodyCount);
	~NBodySim();
	void Simulate();
	float4* GetBodyInfos() { return _d_particleInfos; }
	Cell* GetCells() { return _cells; }
	int* GetCellCount() { return _cellCount; }

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
	int* _numGroups = nullptr;
	int* _newLen = nullptr;
	Cell* _cells = nullptr;
	int* _cellCount = nullptr;
	int* _leafParticles = nullptr;
	int* _leafParticleCount = nullptr;
	int* _flaggedTemp = nullptr;
};