#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optix_device.h>

#include <curand_kernel.h>

#include "all.cuh"

using namespace optix;

// light source stuffs

rtBuffer<float, 1> areaLightCdf;
rtBuffer<float3, 1> areaLightVertices;
rtBuffer<float3, 1> areaLightNormals;
rtBuffer<uint3, 1> areaLightIndices;
rtDeclareVariable(float, areaLightArea, , );
rtDeclareVariable(float4, areaLightIntensity, , );

//////////////////////////// light ////////////////////////////////////

__device__ float3 LightSample(float3 * position, float3 * normal, float * pdf, curandState * state)
{
	float randNum = curand_uniform(state);
	unsigned int indicesIndex = 0;

#if 0
	float previousCdf = 0.f;
	for (;indicesIndex < areaLightCdf.size();indicesIndex++)
	{
		if (previousCdf <= randNum && randNum <= areaLightCdf[indicesIndex]) { break; }
		previousCdf = areaLightCdf[indicesIndex];
	}
#else
	// binary search
	unsigned int count = areaLightCdf.size();
	unsigned int step = 0;
	unsigned int first = 0;
	while (count > 0)
	{
		unsigned int it = first;
		step = count / 2;
		it += step;
		if (areaLightCdf[it] < randNum)
		{
			first = ++it;
			count -= step + 1;
		}
		else
		{
			count = step;
		}
	}
	indicesIndex = first;
#endif

	uint3 triIndices = areaLightIndices[indicesIndex];

	float beta, gamma;
	SquareToBarycentric(&beta, &gamma, curand_uniform(state), curand_uniform(state));

	const float3 & pos1 = areaLightVertices[triIndices.x];
	const float3 & pos2 = areaLightVertices[triIndices.y];
	const float3 & pos3 = areaLightVertices[triIndices.z];

	*position = pos1 * beta + pos2 * gamma + pos3 * (1.0f - gamma - beta);
	*normal = normalize(cross(pos2 - pos1, pos3 - pos1));

	*pdf = 1.f / areaLightArea;

	const float & invPdf = areaLightArea;
	
	ASSERT(!isnan(position->x) && !isnan(position->y) && !isnan(position->z), "LightSample: position is nan");
	ASSERT(!isnan(normal->x) && !isnan(normal->y) && !isnan(normal->z), "LightSample: normal is nan");
	ASSERT(!isnan(*pdf) && *pdf >= 0.0f, "LightSample: pdf has bad value");

	return make_float3(areaLightIntensity) * invPdf;
}

__inline__ __device__ float LightPdfA()
{
	return 1.f / areaLightArea;
}