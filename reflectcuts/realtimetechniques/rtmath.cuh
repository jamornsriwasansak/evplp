#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optix_device.h>

#include <curand_kernel.h>

using namespace optix;

#define M_Inv_PIf 0.318309886183790671537767526745028724068919291480912897495f

__device__ void SquareToCosineWeightedHemisphere(float3 * result, const float x, const float y)
{
	const float term1 = sqrtf(1.f - x);
	const float term2 = M_PIf * 2.f * y;
	const float cosphi = cosf(term2);
	const float sinphi = sinf(term2);
	*result = make_float3(cosphi * term1, sinphi * term1, sqrtf(x));
}

__device__ void SquareToBarycentric(float * beta, float * gamma, const float x, const float y)
{
	const float sqrtX = sqrtf(x);
	*beta = (sqrtX * (1.0f - y));
	*gamma = (sqrtX * y);
}