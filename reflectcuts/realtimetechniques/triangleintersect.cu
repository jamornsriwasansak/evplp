#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optix_device.h>

using namespace optix;

rtBuffer<float3> vertexBuffer;
rtBuffer<int3> indexBuffer;
rtBuffer<float2> texcoordBuffer;

rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometryNormal, attribute geometryNormal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void meshFineIntersect(int primIndex)
{
	int3 vertexIndex = indexBuffer[primIndex];

	float3 p0 = vertexBuffer[vertexIndex.x];
	float3 p1 = vertexBuffer[vertexIndex.y];
	float3 p2 = vertexBuffer[vertexIndex.z];

	float3 n;
	float t, beta, gamma;
	if (optix::intersect_triangle_branchless(ray, p0, p1, p2, n, t, beta, gamma))
	{
		if (rtPotentialIntersection(t))
		{
			geometryNormal = normalize(n);

			float2 t0 = texcoordBuffer[vertexIndex.x];
			float2 t1 = texcoordBuffer[vertexIndex.y];
			float2 t2 = texcoordBuffer[vertexIndex.z];
			texcoord = t1 * beta + t2 * gamma + t0 * (1.0f - beta - gamma);

			rtReportIntersection(0);
		}
	}
}

RT_PROGRAM void meshIntersect(int primIndex)
{
	int3 vertexIndex = indexBuffer[primIndex];

	float3 p0 = vertexBuffer[vertexIndex.x];
	float3 p1 = vertexBuffer[vertexIndex.y];
	float3 p2 = vertexBuffer[vertexIndex.z];

	float3 n;
	float t, beta, gamma;
	if (optix::intersect_triangle_earlyexit(ray, p0, p1, p2, n, t, beta, gamma))
	{
		if (rtPotentialIntersection(t))
		{
			rtReportIntersection(0); // 0 = shadow material on host side
		}
	}
}

RT_PROGRAM void meshBound(int primIdx, float result[6])
{
	const int3 v_idx = indexBuffer[primIdx];

	const float3 v0 = vertexBuffer[v_idx.x];
	const float3 v1 = vertexBuffer[v_idx.y];
	const float3 v2 = vertexBuffer[v_idx.z];
	const float area = length(cross(v1 - v0, v2 - v0));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if (area > 0.0f && !isinf(area))
	{
		aabb->m_min = fminf(fminf(v0, v1), v2);
		aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
	}
	else
	{
		aabb->invalidate();
	}
}