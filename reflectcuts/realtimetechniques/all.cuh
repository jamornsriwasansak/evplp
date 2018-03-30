#pragma once
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optix_device.h>

#include <curand_kernel.h>

//#define DEBUG

#ifdef DEBUG
#define ASSERT(condition, message) \
	if (!(condition))\
	{ rtPrintf(message); rtPrintf("\n"); }
#else
#define ASSERT(condition, message)
#endif