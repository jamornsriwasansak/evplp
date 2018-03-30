#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optix_device.h>

#include <curand_kernel.h>

#include "rtmath.cuh"
#include "rtlightsource.cuh"
#include "rtmaterial.cuh"
#include "rtcomphoton/rtphotonrecord.h"

///// shared info /////

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDimension, rtLaunchDim, );

rtDeclareVariable(uint, rngSeed, , );
rtDeclareVariable(rtObject, topObject, , );
rtDeclareVariable(uint, numLightPaths, , );
rtDeclareVariable(uint, numPhotonsPerLightPath, , );
rtDeclareVariable(uint, numVplLightPaths, , );

///// Light Trace info /////

rtBuffer<RtPhotonRecord, 1> photons;
rtBuffer<RtPhotonInfo, 1> photonInfo;

//rtDeclareVariable(uint, numMaxPhotonsPerLightPath, , );

///// VPL splat /////

rtBuffer<float4, 2> outputBuffer;

rtTextureSampler<float4, 2> deferredPositionTexture;
rtTextureSampler<float4, 2> deferredNormalTexture;
rtTextureSampler<float4, 2> deferredDiffuseTexture;
rtTextureSampler<float4, 2> deferredPhongReflectanceTexture;
rtTextureSampler<float4, 2> deferredPhongExpostureTexture;

rtDeclareVariable(float3, cameraPosition, , );
rtDeclareVariable(uint, doAccumulate, , );
rtDeclareVariable(float, pdfMc, , );
rtDeclareVariable(float, radius, , );
rtDeclareVariable(uint, misMode, , );
rtDeclareVariable(float, clampingValue, , );

rtDeclareVariable(float, vslRadius, , );
rtDeclareVariable(float, vslInvPiRadius2, , );

RT_PROGRAM void insertPhotons(
	const unsigned int pmIndex,
	const unsigned int numBounce,
	const float3 & position)
{
	RtPhotonRecord & rec = photons[pmIndex + numBounce];
	rec.mPosition = position;
}

RT_PROGRAM void exception()
{
	rtPrintExceptionDetails();
}

// ray type 0
struct PerRayData_radiance
{
	bool done;
	curandState * rngState;

	float pdfW;

	float3 nextPosition;
	float3 nextDirection;
	float3 flux;

	unsigned int photonIndex;
	int flag;
};

rtDeclareVariable(PerRayData_radiance, prdRadiance, rtPayload, );

// ray type 1
struct PerRayData_shadow
{
	bool hit;
};
rtDeclareVariable(PerRayData_shadow, prdShadow, rtPayload, );

///////////////////////////// Light Trace Closest Hit & Any Hit //////////////////////////

__device__ float russianProb(const float3 & throughput)
{
	return min(max(throughput.x, max(throughput.y, throughput.z)), 0.98f);
}

__device__ float pdfW2A(const float3 & n2, const float3 & v12)
{
	float3 nv12 = normalize(v12);
	return max(-dot(n2, nv12), 0.f) / dot(v12, v12);
}

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometryNormal, attribute geometryNormal, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );
rtTextureSampler<float4, 2> lambertReflectanceTexture;
rtTextureSampler<float4, 2> phongReflectanceTexture;
rtTextureSampler<float4, 2> phongExponentTexture;
rtDeclareVariable(float4, lightIntensity, , );

RT_PROGRAM void rtMaterialClosestHit()
{
	float3 worldGeometryNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometryNormal));
	float3 ffNormal = faceforward(worldGeometryNormal, -ray.direction, worldGeometryNormal);

	// update position and normal
	float3 position = prdRadiance.nextPosition;
	float3 nextPosition = ray.origin + tHit * ray.direction;
	float3 nextNormal = ffNormal;

	// reject the result if normal is in the other direction or it's light source
	if (dot(geometryNormal, ray.direction) > 0.f || lightIntensity.x > 0.01f)
	{
		prdRadiance.done = true;
		return;
	}

	// fetch all texture information
	float3 lambertReflectance = make_float3(tex2D(lambertReflectanceTexture, texcoord.x, texcoord.y));
	float3 phongReflectance = make_float3(tex2D(phongReflectanceTexture, texcoord.x, texcoord.y));
	float phongExponent = tex2D(phongExponentTexture, texcoord.x, texcoord.y).x;

	const unsigned int index = prdRadiance.photonIndex;

	float3 direction;
	float pdfW;

	// sample next direction from previous direction
	float maxLambert = MaxColor(lambertReflectance);
	float maxPhong = MaxColor(phongReflectance);
	if (maxLambert + maxPhong <= 0.000001f)
	{
		prdRadiance.done = true;
		return;
	}

	photons[index].mFluxDir = -ray.direction;
	photons[index].mPosition = nextPosition;
	photons[index].mNormal = nextNormal;
	photons[index].mFlux = prdRadiance.flux;
	photons[index].mLambertReflectance = lambertReflectance;
	photons[index].mPhongReflectance = phongReflectance;
	photons[index].mPhongExponent = phongExponent;
	photons[index].mFlags = prdRadiance.flag;

	ASSERT(!isnan(prdRadiance.flux.x) && !isnan(prdRadiance.flux.y) && !isnan(prdRadiance.flux.z), "prdRadiance.flux(1) is nan");
	float pSelectLambert = maxLambert / (maxPhong + maxLambert);
	float chooseMaterial = min(curand_uniform(prdRadiance.rngState), 0.999999f);
	photons[index].mPSelectLambert = pSelectLambert;

	// russian roulette
	float russian = russianProb(prdRadiance.flux);
	prdRadiance.flux /= russian;
	prdRadiance.done = (curand_uniform(prdRadiance.rngState) >= russian);
	if (prdRadiance.done) { return; }

	if (chooseMaterial < pSelectLambert)
	{
		prdRadiance.flux *= LambertSample(&direction, &pdfW, -ray.direction, nextNormal, lambertReflectance, prdRadiance.rngState) / pSelectLambert;
		photons[index].mFlags = prdRadiance.flag | PhotonRecordFlag::LambertOnly;
	}
	else
	{
		prdRadiance.flux *= PhongSample(&direction, &pdfW, -ray.direction, geometryNormal, phongReflectance, phongExponent, prdRadiance.rngState) / (1.0f - pSelectLambert);
		photons[index].mFlags = prdRadiance.flag | PhotonRecordFlag::PhongOnly;
	}

	prdRadiance.nextPosition = nextPosition;
	prdRadiance.nextDirection = direction;
}

RT_PROGRAM void rtMaterialAnyHit()
{
	prdShadow.hit = true;
	rtTerminateRay();
}

///////////////////////////////////// LIGHT TRACING ////////////////////////////////

RT_PROGRAM void tracePhotons()
{
	unsigned int launchId = launchIndex.y * launchDimension.x + launchIndex.x;
	unsigned int pmIndex = launchId * numPhotonsPerLightPath;

	for (unsigned int i = 0;i < numPhotonsPerLightPath;i++)
	{
		photons[i + pmIndex].mFlags = 0;
	}

	curandState localState;
	curand_init(launchIndex.y * launchDimension.x + launchIndex.x, rngSeed, 0, &localState);

	// position and direction of first photon
	float3 position, normal;
	float pdf;
	float3 flux = LightSample(&position, &normal, &pdf, &localState);

	// sample outgoing direction from cosine weighted 
	float3 direction;
	float phongPdf;
	float3 att = PhongSample(&direction, &phongPdf, normal, normal, make_float3(1.0f), areaLightIntensity.w, &localState);

	RtPhotonRecord & photon = photons[pmIndex];
	photon.mPosition = position;
	photon.mNormal = normal;
	photon.mFlux = flux;
	photon.mFlags = PhotonRecordFlag::IsUsableVpl;
	photon.mPSelectLambert = 0.0f;

	photon.mLambertReflectance = make_float3(0.0f);
	photon.mPhongReflectance = make_float3(1.0f);
	photon.mPhongExponent = areaLightIntensity.w;
	photon.mFluxDir = normal;

	PerRayData_radiance prd;
	prd.rngState = &localState;
	prd.flux = flux * att;
	prd.done = false;
	prd.nextPosition = position;
	prd.nextDirection = direction;

	for (unsigned int i = 1;i < numPhotonsPerLightPath;i++)
	{
		Ray ray(prd.nextPosition, prd.nextDirection, 0, 0.0001f);

		prd.photonIndex = pmIndex + i;
		if (i != numPhotonsPerLightPath - 1)
		{
			prd.flag = PhotonRecordFlag::IsUsableVpl | PhotonRecordFlag::IsUsablePhoton;
		}
		else
		{
			prd.flag = PhotonRecordFlag::IsUsablePhoton;
		}
		rtTrace(topObject, ray, prd);
		if (prd.done) { break; }
	}
}

//////////////////////////////////// VPL SPLAT /////////////////////////////////////

__forceinline__ __device__ float BalanceHeuristic(const float pdfA, const float pdfB)
{
	return pdfA / (pdfA + pdfB);
}

__forceinline__ __device__ float MaxHeuristic(const float pdfA, const float pdfB)
{
	if (pdfA > pdfB)
	{
		return 1;
	}
	return 0;
}

__forceinline__ __device__ float PowerHeuristic2(const float pdfA, const float pdfB)
{
	float pdfA2 = pdfA * pdfA;
	float pdfB2 = pdfB * pdfB;
	return BalanceHeuristic(pdfA2, pdfB2);
}

__device__ float3 vplSplat(
	const float3 & wi10, // from shading point to eye
	const float3 & firstPosition, const float3 & firstNormal,
	const float3 & firstLambertReflectance, const float3 & firstPhongReflectance, const float firstPhongExponent,
	const RtPhotonRecord & photonRecord
)
{
	float3 v12 = photonRecord.mPosition - firstPosition;

	float unnormCos1 = max(dot(firstNormal, v12), 0.0f);
	float unnormCos2 = max(-dot(photonRecord.mNormal, v12), 0.0f);
	float unnormCos1Cos2 = unnormCos1 * unnormCos2;

	if (unnormCos1Cos2 <= 0.000f) { return make_float3(0.0f); }

	PerRayData_shadow prd;
	prd.hit = false;
	Ray ray(photonRecord.mPosition, -v12, 1, 0.0001, 1 - 0.0001);
	rtTrace(topObject, ray, prd);
	if (prd.hit) { return make_float3(0.0f); }

	float dist2 = dot(v12, v12);
	float dist = sqrtf(dist2);

	float3 wi12 = v12 / dist;
	float3 incomingDir = photonRecord.mFluxDir;

	float3 brdf2 = LambertEvalF(-wi12, incomingDir, photonRecord.mNormal) * photonRecord.mLambertReflectance
		+ PhongEvalF(-wi12, incomingDir, photonRecord.mNormal, photonRecord.mPhongExponent) * photonRecord.mPhongReflectance;

	float3 brdf1 = LambertEvalF(wi10, wi12, firstNormal) * firstLambertReflectance
		+ PhongEvalF(wi10, wi12, firstNormal, firstPhongExponent) * firstPhongReflectance;

	float g21 = unnormCos1Cos2 / (dist2 * dist2);

	if (misMode == 0)
	{
		return photonRecord.mFlux * brdf1 * brdf2 * g21;
	}
	else if (misMode == 1) // Balance Heuristic
	{
		float pdfDe = LambertPdfA(photonRecord.mNormal, firstNormal, -v12) * photonRecord.mPSelectLambert;
		pdfDe += PhongPdfA(photonRecord.mNormal, firstNormal, -v12, photonRecord.mFluxDir, photonRecord.mPhongReflectance, photonRecord.mPhongExponent) * (1.0f - photonRecord.mPSelectLambert);

		float weight = BalanceHeuristic(pdfMc, pdfDe);
		return weight * photonRecord.mFlux * brdf1 * brdf2 * g21;
	}
	else if (misMode == 2) // Max Heuristic
	{
		float pdfDe = LambertPdfA(photonRecord.mNormal, firstNormal, -v12) * photonRecord.mPSelectLambert;
		pdfDe += PhongPdfA(photonRecord.mNormal, firstNormal, -v12, photonRecord.mFluxDir, photonRecord.mPhongReflectance, photonRecord.mPhongExponent) * (1.0f - photonRecord.mPSelectLambert);

		float weight = MaxHeuristic(pdfMc, pdfDe);
		return weight * photonRecord.mFlux * brdf1 * brdf2 * g21;
	}
	else if (misMode == 3) // Power Heuristic
	{
		float pdfDe = LambertPdfA(photonRecord.mNormal, firstNormal, -v12) * photonRecord.mPSelectLambert;
		pdfDe += PhongPdfA(photonRecord.mNormal, firstNormal, -v12, photonRecord.mFluxDir, photonRecord.mPhongReflectance, photonRecord.mPhongExponent) * (1.0f - photonRecord.mPSelectLambert);

		float weight = PowerHeuristic2(pdfMc, pdfDe);
		return weight * photonRecord.mFlux * brdf1 * brdf2 * g21;
	}
	else if (misMode == 4) // KK weak singularities clamping
	{
		return photonRecord.mFlux * optix::fminf(g21, clampingValue) * brdf1 * brdf2;
	}
	else if (misMode == 5) // Local VPLs clamping
	{
		return photonRecord.mFlux * optix::fminf(g21 * brdf1 * brdf2, make_float3(clampingValue));
	}
}

RT_PROGRAM void splatColor()
{
	float2 screenUv = (make_float2(launchIndex) + make_float2(0.5)) / make_float2(launchDimension);
	float4 positionInfo = tex2D(deferredPositionTexture, screenUv.x, screenUv.y);
	float3 firstPosition = make_float3(positionInfo);
	float stencil = positionInfo.w;
	if (stencil == 0.0f) { return; }

	float3 firstNormal = make_float3(tex2D(deferredNormalTexture, screenUv.x, screenUv.y));
	float3 lambertReflectance = make_float3(tex2D(deferredDiffuseTexture, screenUv.x, screenUv.y));
	float4 phongInfo = tex2D(deferredPhongReflectanceTexture, screenUv.x, screenUv.y);

	float3 phongReflectance = make_float3(phongInfo);
	float phongExponent = phongInfo.w;

	float3 wi01 = normalize(cameraPosition - firstPosition); // from shading point to eye
															 //rtPrintf("%f %f %f", wi01.x, wi01.y, wi01.z);

	float3 result = make_float3(0.0f);

	curandState localState;
	curand_init(launchIndex.y * launchDimension.x + launchIndex.x, rngSeed, 0, &localState);

	unsigned int lightPathOffset = unsigned int(min(curand_uniform(&localState), 0.999999f) * numLightPaths);

	for (int i = 0;i < numVplLightPaths;i++)
	{
		unsigned int lightPathId = (i + lightPathOffset) % numLightPaths;
		unsigned int lightVertexOffset = lightPathId * numPhotonsPerLightPath;
		for (int j = 0;j < numPhotonsPerLightPath;j++)
		{
			if ((photons[lightVertexOffset + j].mFlags & PhotonRecordFlag::IsUsableVpl) != 0)
			{
				result += vplSplat(wi01, firstPosition, firstNormal, lambertReflectance, phongReflectance, phongExponent, photons[lightVertexOffset + j]);
			}
		}
	}

	outputBuffer[launchIndex] = make_float4(result / (float) numVplLightPaths) + doAccumulate * outputBuffer[launchIndex];
}
