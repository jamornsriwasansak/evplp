#pragma once

//#define NUM_PHOTONS 500000
//#define NUM_MAX_PHOTONS_PER_LIGHT_PATH 5

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

enum PhotonRecordFlag
{
	IsUsableVpl = 1 << 0,
	IsUsablePhoton = 1 << 1,
	LambertOnly = 1 << 2,
	PhongOnly = 1 << 3
};

struct RtPhotonRecord
{
	optix::float3 mPosition;				unsigned int mFlags;
	optix::float3 mNormal;					float mPSelectLambert;					
	optix::float3 mFlux;					float padding1;
	optix::float3 mFluxDir;                 float padding2;
	optix::float3 mLambertReflectance;		float padding3;
	optix::float3 mPhongReflectance;		float mPhongExponent;
};

struct RtPhotonInfo
{
	unsigned int numPhotons;
};