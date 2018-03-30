#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optix_device.h>

#include <curand_kernel.h>

#include "rtmath.cuh"
#include "rtlightsource.cuh"
#include "rtmaterial.cuh"

using namespace optix;

// path tracing stuffs

rtBuffer<float4, 2> outputBuffer;

rtTextureSampler<float4, 2> deferredPositionTexture;
rtTextureSampler<float4, 2> deferredNormalTexture;
rtTextureSampler<float4, 2> deferredDiffuseTexture;
rtTextureSampler<float4, 2> deferredPhongReflectanceTexture;

rtDeclareVariable(float3, cameraPosition, , );
rtDeclareVariable(rtObject, topObject, , );
rtDeclareVariable(uint, doAccumulate, , );

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDimension, rtLaunchDim, );

rtDeclareVariable(uint, maxBounces, , );
rtDeclareVariable(uint, rngSeed, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

static __device__ __inline__ uchar4 make_color(const float3& c)
{
	return make_uchar4(static_cast<unsigned char>(c.z * 255.99f),  /* B */
		static_cast<unsigned char>(c.y*255.99f),  /* G */
		static_cast<unsigned char>(c.x*255.99f),  /* R */
		255u);                                                 /* A */
}

__device__ float3 generatePointInSphere(const float2 & sample)
{
	const float z = 1.0f - 2.0f * sample.y;
	const float r = sqrt(sample.y * (1.0f - sample.y));
	const float phi = 2.0f * M_PIf * sample.x; // phi = [0, 2pi)
	const float cosphi = cos(phi);
	const float sinphi = sin(phi);
	return make_float3(2.0f * cosphi * r, 2.0f * sinphi * r, z);
}

__device__ float russianProb(const float3 & throughput)
{
	return max(max(throughput.x, 0.98f), max(throughput.y, throughput.z));
}

RT_PROGRAM void exception()
{
	rtPrintExceptionDetails();
}

// ray type 0
struct PerRayData_radiance
{
	bool done;
	bool hit;
	curandState * rngState;

	float brdfPdfW;
	float3 result;
	float3 position;
	float3 geometryNormal;
	float3 direction;
	float3 attenuation;
};
rtDeclareVariable(PerRayData_radiance, prdRadiance, rtPayload, );

// ray type 1
struct PerRayData_shadow
{
	bool hit;
};
rtDeclareVariable(PerRayData_shadow, prdShadow, rtPayload, );

///////////////////////////// MIS Stuffs /////////////////////////

__device__ float MisWeight(const float pdf1, const float pdf2)
{
	return pdf1 / (pdf1 + pdf2);
}

__device__ float pdfW2A(const float3 & n2, const float3 & v12)
{
	float3 nv12 = normalize(v12);
	return max(-dot(n2, nv12), 0.f) / dot(v12, v12);
}

///////////////////////// RT MATERIAL ///////////////////////////

rtDeclareVariable(float2, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometryNormal, attribute geometryNormal, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );
rtTextureSampler<float4, 2> lambertReflectanceTexture;
rtTextureSampler<float4, 2> phongReflectanceTexture;
rtTextureSampler<float4, 2> phongExponentTexture;
rtDeclareVariable(float4, lightIntensity, , );

//#define FAVOR_LIGHT_SAMPLE
//#define FAVOR_BSDF_SAMPLE

RT_PROGRAM void rtMaterialClosestHit()
{
	ASSERT(!isnan(prdRadiance.attenuation.x) && !isnan(prdRadiance.attenuation.y) && !isnan(prdRadiance.attenuation.z), "prdRadiance.atteanuation(1) is nan");
	//prdRadiance.hit = true;

	float3 worldGeometryNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometryNormal));
	float3 ffNormal = faceforward(worldGeometryNormal, -ray.direction, worldGeometryNormal);

	// update position and normal
	float3 nextPosition = ray.origin + tHit * ray.direction;
	float3 nextNormal = ffNormal;

	// reject the result if normal is in the other direction
	if (dot(geometryNormal, ray.direction) > 0.f)
	{
		prdRadiance.result = make_float3(0.0f);
		prdRadiance.done = true;
		return;
	}

	// check if it hit the light source
	if (lightIntensity.x > 0.01f)
	{
		// compute mis weight
		float brdfPdfA = (prdRadiance.brdfPdfW * pdfW2A(ffNormal, nextPosition - prdRadiance.position));
		float lightPdfA = LightPdfA();
		float weight = MisWeight(brdfPdfA, lightPdfA);
		#ifdef FAVOR_LIGHT_SAMPLE
			weight = 0.0f;
		#elif defined(FAVOR_BSDF_SAMPLE)
			weight = 1.0f;
		#endif
		prdRadiance.result = weight * prdRadiance.attenuation *
			PhongEvalF(geometryNormal, normalize(prdRadiance.position - nextPosition), geometryNormal, lightIntensity.w) * make_float3(lightIntensity);
		prdRadiance.done = true;
		return;
	}

	// this is last bounce. don't do next event estimation
	if (prdRadiance.done) { return; }

	float lightPdf;
	float3 lightPosition, lightNormal;
	float3 lightValue = LightSample(&lightPosition, &lightNormal, &lightPdf, prdRadiance.rngState);

	float3 toLight = lightPosition - nextPosition;
	float3 toLightNorm = normalize(toLight);

	Ray ray(lightPosition, -toLight, 1, 0.00001, 0.99999);
	PerRayData_shadow prd2;
	prd2.hit = false;
	rtTrace(topObject, ray, prd2);

	float3 lambertReflectance = make_float3(tex2D(lambertReflectanceTexture, texcoord.x, texcoord.y));
	float3 phongReflectance = make_float3(tex2D(phongReflectanceTexture, texcoord.x, texcoord.y));
	float phongExponent = tex2D(phongExponentTexture, texcoord.x, texcoord.y).x;

	// check bad color (for reduce bracnching)
	float maxLambert = MaxColor(lambertReflectance);
	float maxPhong = MaxColor(phongReflectance);
	prdRadiance.done = (maxLambert + maxPhong <= 0.000001f);
	if (prdRadiance.done) { return; }

	float pSelectLambert = maxLambert / (maxPhong + maxLambert);
	float chooseMaterial = min(curand_uniform(prdRadiance.rngState), 0.999999f);

	if (chooseMaterial < pSelectLambert)
	{
		ASSERT(1.0f >= pSelectLambert && pSelectLambert > 0.0f, "pSelectLambert is not in (0, 1]");

		if (!prd2.hit)
		{
			// next event estimation
			float brdfPdf = LambertPdfA(ffNormal, lightNormal, toLight);
			float weight = MisWeight(lightPdf, brdfPdf);
			#ifdef FAVOR_LIGHT_SAMPLE
				weight = 1.0f;
			#elif defined(FAVOR_BSDF_SAMPLE)
				weight = 0.0f;
			#endif
			prdRadiance.result = weight * lightValue * LambertEval(toLightNorm, normalize(prdRadiance.position - nextPosition), ffNormal, lambertReflectance) * GeometryTerm(ffNormal, lightNormal, toLight) * prdRadiance.attenuation / pSelectLambert
				* PhongEvalF(lightNormal, -toLightNorm, lightNormal, areaLightIntensity.w); // light source material
		}

		// sample outgoing direction
		prdRadiance.attenuation *= LambertSample(&prdRadiance.direction, &prdRadiance.brdfPdfW, normalize(prdRadiance.position - nextPosition), geometryNormal, lambertReflectance, prdRadiance.rngState) / pSelectLambert;
	}
	else
	{
		ASSERT(1.0f > pSelectLambert && pSelectLambert >= 0.0f, "pSelectLambert is not in [0, 1)");

		if (!prd2.hit)
		{
			// next event estimation
			float brdfPdf = PhongPdfA(ffNormal, lightNormal, toLight, normalize(prdRadiance.position - nextPosition), phongReflectance, phongExponent);
			float weight = MisWeight(lightPdf, brdfPdf);
			#ifdef FAVOR_LIGHT_SAMPLE
				weight = 1.0f;
			#elif defined(FAVOR_BSDF_SAMPLE)
				weight = 0.0f;
			#endif
			prdRadiance.result = weight * lightValue * PhongEval(toLightNorm, normalize(prdRadiance.position - nextPosition), ffNormal, phongReflectance, phongExponent) * GeometryTerm(ffNormal, lightNormal, toLight) * prdRadiance.attenuation / (1.0f - pSelectLambert)
				* PhongEvalF(lightNormal, -toLightNorm, lightNormal, areaLightIntensity.w); // light source material
		}
		prdRadiance.attenuation *= PhongSample(&prdRadiance.direction, &prdRadiance.brdfPdfW, normalize(prdRadiance.position - nextPosition), geometryNormal, phongReflectance, phongExponent, prdRadiance.rngState) / (1.0f - pSelectLambert);
	}

	float russian = russianProb(prdRadiance.attenuation);
	ASSERT(russian > 0.0f, "russian roulette prob is <= 0.0");
	prdRadiance.done = (curand_uniform(prdRadiance.rngState) >= russian);
	if (prdRadiance.done) { return; }

	prdRadiance.position = nextPosition;
	prdRadiance.attenuation /= russian;

	ASSERT(!isnan(prdRadiance.attenuation.x) && !isnan(prdRadiance.attenuation.y) && !isnan(prdRadiance.attenuation.z), "prdRadiance.atteanuation(2) is nan");
}

/////////////////////////// ANY HIT ////////////////////////////

RT_PROGRAM void rtMaterialAnyHit()
{
	prdShadow.hit = true;
	rtTerminateRay();
}

/////////////////////////////////////////////////////////////////

__device__ float3 pathTraceSimple(
	const float3 & cameraPos,
	const float3 & firstPosition,
	const float3 & firstNormal,
	const float3 & firstLambertReflectance,
	const float3 & firstPhongReflectance,
	const float & firstPhongExponent,
	curandState * rngState)
{
	float3 cameraVec = normalize(firstPosition - cameraPos);
	float3 result = make_float3(0.0f);

	PerRayData_radiance prd;
	prd.rngState = rngState;

	const unsigned int numSamples = 1;
	const float invNumSamples = 1.f / (float)numSamples;

	float3 position = firstPosition;
	float3 normal = firstNormal;

	for (int k = 0;k < numSamples;k++)
	{
		prd.position = firstPosition;
		prd.geometryNormal = firstNormal;
		prd.attenuation = make_float3(1.0);

		// first bounce
		{
			// sample light source
			float lightPdf;
			float3 lightPosition, lightNormal;
			float3 lightValue = LightSample(&lightPosition, &lightNormal, &lightPdf, rngState);

			float3 toLight = lightPosition - position;
			float3 toLightNorm = normalize(toLight);

			Ray ray(lightPosition, -toLight, 1, 0.0001f, 1.0f - 0.0001f);
			PerRayData_shadow prd2;
			prd2.hit = false;
			rtTrace(topObject, ray, prd2);

			// select material
			float maxLambert = MaxColor(firstLambertReflectance);
			float maxPhong = MaxColor(firstPhongReflectance);

			float pSelectLambert = maxLambert / (maxPhong + maxLambert);

			if (maxLambert + maxPhong <= 0.000001f) { return make_float3(0.0f); }
			float chooseMaterial = min(curand_uniform(prd.rngState), 0.999999f);
			if (chooseMaterial < pSelectLambert)
			{
				ASSERT(1.0f >= pSelectLambert && pSelectLambert > 0.0f, "pSelectLambert(a) is not in (0, 1]");

				if (!prd2.hit)
				{
					// compute mis weight
					float brdfPdf = LambertPdfA(normal, lightNormal, toLight);
					float weight = MisWeight(lightPdf, brdfPdf);
					#ifdef FAVOR_LIGHT_SAMPLE
						weight = 1.0f;
					#elif defined(FAVOR_BSDF_SAMPLE)
						weight = 0.0f;
					#endif
					result += weight * lightValue * LambertEval(-cameraVec, toLightNorm, normal, firstLambertReflectance) * GeometryTerm(normal, lightNormal, toLight) / pSelectLambert
						* PhongEvalF(lightNormal, -toLightNorm, lightNormal, areaLightIntensity.w); // light source material
				}

				prd.attenuation *= LambertSample(&prd.direction, &prd.brdfPdfW, -cameraVec, normal, firstLambertReflectance, prd.rngState) / pSelectLambert;
			}
			else
			{
				ASSERT(1.0f > pSelectLambert && pSelectLambert >= 0.0f, "pSelectLambert(a) is not in [0, 1)");

				if (!prd2.hit)
				{
					// compute mis weight
					float brdfPdf = PhongPdfA(normal, lightNormal, toLight, -cameraVec, firstPhongReflectance, firstPhongExponent);
					float weight = MisWeight(lightPdf, brdfPdf);
					#ifdef FAVOR_LIGHT_SAMPLE
						weight = 1.0f;
					#elif defined(FAVOR_BSDF_SAMPLE)
						weight = 0.0f;
					#endif
					result += weight * lightValue * PhongEval(-cameraVec, toLightNorm, normal, firstPhongReflectance, firstPhongExponent) * GeometryTerm(normal, lightNormal, toLight) / (1.0f - pSelectLambert)
						* PhongEvalF(lightNormal, -toLightNorm, lightNormal, areaLightIntensity.w); // light source material
				}

				prd.attenuation *= PhongSample(&prd.direction, &prd.brdfPdfW, -cameraVec, normal, firstPhongReflectance, firstPhongExponent, prd.rngState) / (1.0f - pSelectLambert);
			}
			ASSERT(!isnan(prd.attenuation.x) && !isnan(prd.attenuation.y) && !isnan(prd.attenuation.z), "prd.atteanuation(a) is nan");
		}

		for (size_t i = 0;i < maxBounces;i++)
		{
			prd.done = (i == maxBounces - 1);
			prd.result = make_float3(0.f);

			Ray ray(prd.position, prd.direction, 0, 0.00001);
			rtTrace(topObject, ray, prd);

			result += prd.result;

			if (prd.done) { break; }
		}
	}

	return result * invNumSamples;
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

	curandState localState;
	curand_init(launchIndex.y * launchDimension.x + launchIndex.x, rngSeed, 0, &localState);

	float3 result = pathTraceSimple(cameraPosition, firstPosition, firstNormal, lambertReflectance, phongReflectance, phongExponent, &localState);
	ASSERT(!isnan(result.x) && !isnan(result.y) && !isnan(result.z), "result is nan");
	if (doAccumulate == 1)
	{
		outputBuffer[launchIndex] += make_float4(result);
	}
	else
	{
		outputBuffer[launchIndex] = make_float4(result);
	}
}
