#pragma once

#include <cuda_runtime.h>

__host__ __device__ inline float3 operator+(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
__host__ __device__ inline float3 operator-(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
__host__ __device__ inline float3 operator*(const float3& lhs, float scalar)
{
	return make_float3(lhs.x * scalar, lhs.y * scalar, lhs.z * scalar);
}
__host__ __device__ inline float3 operator*(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}
__host__ __device__ inline float3 operator/(const float3& lhs, float scalar)
{
	return make_float3(lhs.x / scalar, lhs.y / scalar, lhs.z / scalar);
}

__host__ __device__ inline float3 operator-(const float3& f)
{
	return make_float3(-f.x, -f.y, -f.z);
}

__host__ __device__ inline float dot(const float3& lhs, const float3& rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__host__ __device__ inline float length(const float3& f)
{
	return sqrtf(f.x * f.x + f.y * f.y + f.z * f.z);
}

__host__ __device__ inline float lengthSquared(const float3& f)
{
	return f.x * f.x + f.y * f.y + f.z * f.z;
}

__host__ __device__ inline float3 normalized(const float3& f)
{
	float len = length(f);
	return f / len;
}

__host__ __device__ inline bool aproxZero(const float3& f)
{
	return (f.x > -1e-8f && f.x < 1e-8f) && (f.y > -1e-8f && f.y < 1e-8f) && (f.z > -1e-8f && f.z < 1e-8f);
}

__host__ __device__ inline float3 cross(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.y * rhs.z - lhs.z * rhs.y,
						lhs.z * rhs.x - lhs.x * rhs.z,
						lhs.x * rhs.y - lhs.y * rhs.x);
}