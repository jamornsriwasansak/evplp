#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optix_device.h>

#include <curand_kernel.h>

#include "all.cuh"

using namespace optix;

__inline__ __device__ void ComputeOrthonormalBasis(float3 * xBasis, float3 * yBasis, const float3 & zBasis)
{
	float sign = copysign(1.0, zBasis.z);
	const float a = -1.0f / (sign + zBasis.z);
	const float b = zBasis.x * zBasis.y * a;
	*xBasis = make_float3(1.0f + sign * zBasis.x * zBasis.x * a, sign * b, -sign * zBasis.x);
	*yBasis = make_float3(b, sign + zBasis.y * zBasis.y * a, -zBasis.y);

	ASSERT(!isnan(xBasis->x) && !isnan(xBasis->y) && !isnan(xBasis->z), "ComputeOrthonormalBasis: xBasis is nan");
	ASSERT(!isnan(yBasis->x) && !isnan(yBasis->y) && !isnan(yBasis->z), "ComputeOrthonormalBasis: yBasis is nan");
}

__inline__ __device__ float MaxColor(const float3 & color)
{
	return max(max(color.x, color.y), color.z);
}

__inline__ __device__ float GeometryTerm(const float3 & n1, const float3 & n2, const float3 v12)
{
	const float cos1Unnorm = max(dot(n1, v12), 0.f);
	const float cos2Unnorm = max(-dot(n2, v12), 0.f);
	const float d2 = dot(v12, v12);

	ASSERT(d2 > 0.0f, "GeometryTerm: distance(denominator) is zero");
	return cos1Unnorm * cos2Unnorm / (d2 * d2);
}

__inline__ __device__ float LambertPdfW(const float3 & n1, const float3 & v12)
{
	const float cos1Unnorm = max(dot(n1, normalize(v12)), 0.f);
	return cos1Unnorm;
}

__inline__ __device__ float LambertPdfA(const float3 & n1, const float3 & n2, const float3 & v12)
{
	const float cos1Unnorm = max(dot(n1, v12), 0.f);
	const float cos2Unnorm = max(-dot(n2, v12), 0.f);
	const float d2 = dot(v12, v12);

	ASSERT(d2 > 0.0f, "LambertPdfA: distance(denominator) is zero");
	return cos1Unnorm * cos2Unnorm / (d2 * d2) * M_Inv_PIf;
}

__inline__ __device__ float3 LambertSample(float3 * out, float * pdfW, const float3 & in, const float3 & normal, const float3 & lambertReflectance, curandState * rngState)
{
	cosine_sample_hemisphere(curand_uniform(rngState), curand_uniform(rngState), *out);
	Onb onb(normal);
	onb.inverse_transform(*out);

	*pdfW = max(dot(*out, normal), 0.f) * M_Inv_PIf;

	ASSERT(!isnan(lambertReflectance.x) && !isnan(lambertReflectance.y) && !isnan(lambertReflectance.z), "LambertSample: lambertReflectance is nan");
	return lambertReflectance;
}

__inline__ __device__ float3 LambertEval(const float3 & out, const float3 & in, const float3 & normal, const float3 & lambertReflectance)
{
	return lambertReflectance * M_Inv_PIf;
}

__inline__ __device__ float LambertEvalF(const float3 & out, const float3 & in, const float3 & normal)
{
	return M_Inv_PIf;
}

__inline__ __device__ float PhongPdfW(const float3 & n1, const float3 & v12, const float3 & in, const float3 & phongReflectance, const float phongExponent)
{
	float3 wi12 = normalize(v12);
	float3 reflectVec = normalize(reflect(-in, n1));
	float cosReflect = max(dot(wi12, reflectVec), 0.f);
	if (cosReflect <= 0.000001f || phongReflectance.x <= 0.000001f) { return 0.0f; }
	return (phongExponent + 1.0f) * 0.5f * M_Inv_PIf * powf(cosReflect, phongExponent);
}

__inline__ __device__ float PhongPdfA(const float3 & n1, const float3 & n2, const float3 & v12, const float3 & in, const float3 & phongReflectance, const float phongExponent)
{
	float3 wi12 = normalize(v12);
	float3 reflectVec = normalize(reflect(-in, n1));
	float cosReflect = max(dot(wi12, reflectVec), 0.f);
	if (cosReflect <= 0.000001f || phongReflectance.x <= 0.000001f) { return 0.0f; }

	float pdfW = (phongExponent + 1.0f) * 0.5f * M_Inv_PIf * powf(cosReflect, phongExponent);
	float cos2 = max(-dot(n2, wi12), 0.0f);
	float dist2 = dot(v12, v12);

	//rtPrintf("cosreflect : %f, phong exp: %f\n", cosReflect, phongExponent);

	ASSERT(dist2 > 0.0f, "PhongPdfA: distance(denominator) is zero");
	return pdfW * cos2 / dist2;
}

__inline__ __device__ float3 PhongEval(const float3 & out, const float3 & in, const float3 & normal, const float3 & phongReflectance, const float phongExponent)
{
	float3 reflectVec = reflect(-in, normal);
	float dotWrWo = max(dot(out, reflectVec), 0.0f);
	if (dotWrWo <= 0.000001f || phongReflectance.x <= 0.000001f) { return make_float3(0.0f); }
	return phongReflectance * (phongExponent + 2.0f) * powf(dotWrWo, phongExponent) * (M_Inv_PIf) * 0.5f;
}

__inline__ __device__ float PhongEvalF(const float3 & out, const float3 & in, const float3 & normal, const float phongExponent)
{
	float3 reflectVec = reflect(-in, normal);
	float dotWrWo = max(dot(out, reflectVec), 0.0f);
	if (dotWrWo <= 0.000001f) { return 0.0f; }
	return (phongExponent + 2.0f) * powf(dotWrWo, phongExponent) * (M_Inv_PIf) * 0.5f;
}

__inline__ __device__ float3 PhongSample(float3 * out, float * pdfW, const float3 & in, const float3 & normal, const float3 & phongReflectance, const float phongExponent, curandState * rngState)
{
	float3 reflectVec = reflect(-in, normal);

	float sampleX = curand_uniform(rngState);
	float sampleY = curand_uniform(rngState);

	float cosTheta = powf(sampleX, 1.f / (phongExponent + 1.f));
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

	float phi = 2.f * M_PIf * sampleY;
	float cosPhi = cosf(phi);
	float sinPhi = sinf(phi);

	*out = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);

	Onb onb(reflectVec);
	onb.inverse_transform(*out);

	float unsafeCosNormal = dot(*out, normal);
	float cosNormal = max(unsafeCosNormal, 0.f);
	float cosReflect = max(dot(*out, reflectVec), 0.f);

	if (unsafeCosNormal > 0.0f)
	{
		*pdfW = (phongExponent + 1.0f) * 0.5f * powf(cosReflect, phongExponent) * M_Inv_PIf;
	}
	else
	{
		*pdfW = 0.0f;
	}

	float3 result = (phongExponent + 2.0f) / (phongExponent + 1.0f) * cosNormal * phongReflectance;
	ASSERT(!isnan(result.x) && !isnan(result.y) && !isnan(result.z), "PhongSample: result is nan");
	return result;
}