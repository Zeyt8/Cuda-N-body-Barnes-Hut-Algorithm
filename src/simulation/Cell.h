#pragma once

#include <cstdint>

enum CellType
{
	NODE = 0,
	LEAF = 1
};

struct Cell
{
	uint64_t key;
	int level;
	int type;
	int start;
	int count;

	double mass;
	double3 com;

	double Qxx, Qyy;
	double Qxy, Qxz, Qyz;
};