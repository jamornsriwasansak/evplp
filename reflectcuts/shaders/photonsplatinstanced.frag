#version 450 core

#define IsUsablePhoton 2
#define IsOnLambertSurface 4
#define IsOnPhongSurface 8

out vec3 color;

in vec3 gPosition;
in vec2 gScreenUv;
flat in int gInstanceId;

uniform int uMisMode;
uniform float uPhotonRadius;
uniform sampler2D uPositionTexture;
uniform sampler2D uNormalTexture;
uniform sampler2D uDiffuseTexture;
uniform sampler2D uPhongInfoTexture;
uniform float uInvNumLightPaths;
uniform float uInvPhotonRadius2;
uniform float uPdfMc;
uniform vec3 uCameraPosition;
uniform float uClampingValue;

struct PhotonRecord
{
	vec3 position;				int flags;
	vec3 normal;				float pSelectLambert;
	vec3 flux;					float padding1;
	vec3 fluxDir;				float padding2;
	vec3 lambertReflectance;	float padding3;
	vec3 phongReflectance;		float phongExponent;
};

layout (std430, binding=0) buffer PhotonRecords
{
	PhotonRecord photons[];
};

#define InvPi 0.318309886183790671537767526745028724068919291480912897495

vec3 LambertEval(const vec3 w10, const vec3 w12, const vec3 normal, const vec3 lambertReflectance)
{
	/*float cos1Unnorm = max(dot(normal1, v12), 0.f);
	float cos2Unnorm = max(-dot(normal2, v12), 0.f);
	if (cos1Unnorm * cos2Unnorm <= 0.0f) { return vec3(0.0f); }
	return InvPi * diffuseColor;*/
	if (dot(w10, normal) <= 0.0f || dot(w12, normal) <= 0.0f) { return vec3(0.0f); }
	return InvPi * lambertReflectance;
}

vec3 PhongEval(const vec3 outVec, const vec3 inVec, const vec3 normal, const vec3 phongReflectance, const float phongExponent)
{
	vec3 reflectVec = reflect(-inVec, normal);
	float dotWrWo = dot(outVec, reflectVec);
	if (dotWrWo <= 0.00001f) { return vec3(0.0f); }
	return phongReflectance * (phongExponent + 2.0f) * pow(dotWrWo, phongExponent) * InvPi * 0.5;
}

//vec3 PhongEval(const vec3 outVec, const vec3 inVec, const vec3 normal, const vec3 phongReflectance, const float phongExponent)
//{
	 
//}

float LambertPdfW(const vec3 normal1, const vec3 v12)
{
	float cos1Unnorm = max(dot(normal1, normalize(v12)), 0.f);
	return cos1Unnorm * InvPi;
}

float LambertPdfA(const vec3 normal1, const vec3 normal2, const vec3 v12)
{
	float cos1Unnorm = max(dot(normal1, v12), 0.f);
	float cos2Unnorm = max(-dot(normal2, v12), 0.f);
	float d2 = dot(v12, v12);
	return cos1Unnorm * cos2Unnorm / (d2 * d2) * InvPi;
}

float PhongPdfW(const vec3 normal1, const vec3 wi12, const vec3 inVec, const vec3 phongReflectance, const float phongExponent)
{
	vec3 reflectVec = reflect(-inVec, normal1);
	float dotWrWo = max(dot(wi12, reflectVec), 0.f);
	if (dotWrWo <= 0.00001f || phongReflectance.x <= 0.00001f) { return 0.0f; }
	return (phongExponent + 1.0f) * 0.5f * InvPi * pow(dotWrWo, phongExponent);
}

float PhongPdfA(const vec3 normal1, const vec3 normal2, const vec3 v12, const vec3 in1, const vec3 phongReflectance, const float phongExponent)
{
	vec3 wi12 = normalize(v12);
	vec3 reflectVec = reflect(-in1, normal1);
	float dotWrWo = max(dot(wi12, reflectVec), 0.f);
	if (dotWrWo <= 0.00001f) { return 0.0f; }
	float pdfW = (phongExponent + 1.0f) * 0.5f * InvPi * pow(dotWrWo, phongExponent);
	float cos2 = max(-dot(normal2, wi12), 0.0f);
	float dist2 = dot(v12, v12);

	return pdfW * cos2 / dist2;
}

#if 1
float MisWeight(float pdfA, float pdfB)
{
	// can be replaced with step function
	if (pdfA + pdfB <= 0.00000001f) { return 0.0f; }
	return pdfA / (pdfA + pdfB);
}
#else
float MisWeight(float pdfA, float pdfB)
{
	if (pdfA >= pdfB)
	{
		return 1;
	}
	return 0;
}
#endif


float BalanceHeuristic(const float pdfA, const float pdfB)
{
	return pdfA / (pdfA + pdfB);
}

float MaxHeuristic(const float pdfA, const float pdfB)
{
	if (pdfA > pdfB)
	{
		return 1;
	}
	return 0;
}

float PowerHeuristic2(const float pdfA, const float pdfB)
{
	float pdfA2 = pdfA * pdfA;
	float pdfB2 = pdfB * pdfB;
	return BalanceHeuristic(pdfA2, pdfB2);
}

float Distance2(vec3 p1, vec3 p2)
{
	vec3 vec = p1 - p2;
	return dot(vec, vec);
}

void main()
{
	int photonFlags = photons[gInstanceId].flags;

	vec3 shadingPosition = texture(uPositionTexture, gScreenUv).xyz;

	// reject photon that are on different position
	float photonRadius2 = uPhotonRadius * uPhotonRadius;
	if (Distance2(photons[gInstanceId].position, shadingPosition) > photonRadius2) { discard; }

	// fetch stuffs
	vec3 shadingNormal = texture(uNormalTexture, gScreenUv).xyz;
	vec3 shadingDiffuseColor = texture(uDiffuseTexture, gScreenUv).xyz;
	vec4 shadingPhongInfo = texture(uPhongInfoTexture, gScreenUv);
	vec3 shadingPhongReflectance = shadingPhongInfo.xyz;
	float shadingPhongExponent = shadingPhongInfo.w;

	int prevInstanceId = gInstanceId - 1;

	#if 0
	vec3 v12 = photons[prevInstanceId].position - shadingPosition;
	vec3 w12 = normalize(v12);
	vec3 n1 = shadingNormal;
	#else
	vec3 v12 = photons[prevInstanceId].position - photons[gInstanceId].position;
	vec3 w12 = normalize(v12);
	vec3 n1 = photons[gInstanceId].normal;
	#endif

	float pdf = 0.f;
	vec3 brdf1 = vec3(0.0f);
	vec3 w10 = normalize(uCameraPosition - shadingPosition);

	vec3 prevFluxDir = photons[prevInstanceId].fluxDir;

	brdf1 = LambertEval(w10, w12, shadingNormal, shadingDiffuseColor) + PhongEval(w10, w12, shadingNormal, shadingPhongReflectance, shadingPhongExponent);
	vec3 brdf2 = LambertEval(-w12, prevFluxDir, photons[prevInstanceId].normal, photons[prevInstanceId].lambertReflectance) + PhongEval(-w12, prevFluxDir, photons[prevInstanceId].normal, photons[prevInstanceId].phongReflectance, photons[prevInstanceId].phongExponent);

	float mixPdfW = LambertPdfW(photons[prevInstanceId].normal, -w12) 
	               * photons[prevInstanceId].pSelectLambert;
	mixPdfW += PhongPdfW(photons[prevInstanceId].normal, -w12, prevFluxDir, photons[prevInstanceId].phongReflectance, photons[prevInstanceId].phongExponent)
	           * (1.0f - photons[prevInstanceId].pSelectLambert);

	float mixPdfA = mixPdfW * max(dot(n1, w12), 0.0f) / dot(v12, v12);

	if (mixPdfW > 0.0f)
	{
		/// TODO:: OPTIMIZATION:: these modes can be further optimize by precomputing weight during the tracing process so that we don't have to compute it in fragment shader
		if (uMisMode == 0)
		{
			//color = brdf1 * brdf2 * dot(photons[prevInstanceId].normal, -w12) * (InvPi * uInvPhotonRadius2) * photons[prevInstanceId].flux * uInvNumLightPaths / mixPdfW;
			color = brdf1 * (InvPi * uInvPhotonRadius2) * photons[gInstanceId].flux * uInvNumLightPaths; 
		}
		else if (uMisMode == 1)
		{
			float weight = BalanceHeuristic(mixPdfA, uPdfMc);
			color = brdf1 * (InvPi * uInvPhotonRadius2) * photons[gInstanceId].flux * uInvNumLightPaths * weight;
		}
		else if (uMisMode == 2)
		{
			float weight = MaxHeuristic(mixPdfA, uPdfMc);
			color = brdf1 * (InvPi * uInvPhotonRadius2) * photons[gInstanceId].flux * uInvNumLightPaths * weight;
		}
		else if (uMisMode == 3)
		{
			float weight = PowerHeuristic2(mixPdfA, uPdfMc);
			color = brdf1 * (InvPi * uInvPhotonRadius2) * photons[gInstanceId].flux * uInvNumLightPaths * weight;
		}
		else if (uMisMode == 4)
		{
			float distance2 = dot(v12, v12);

			float cosCos = max(dot(shadingNormal, w12), 0.0f) * max(-dot(photons[prevInstanceId].normal, w12), 0.0f);
			if (cosCos <= 0.0f) { discard; }

			float geometryTerm = cosCos / distance2;
			color = brdf1 * (InvPi * uInvPhotonRadius2) * photons[gInstanceId].flux  * uInvNumLightPaths * max(geometryTerm - uClampingValue, 0.0f) / geometryTerm;
		}
		else if (uMisMode == 5)
		{
			float distance2 = dot(v12, v12);

			float cosCos = max(dot(shadingNormal, w12), 0.0f) * max(-dot(photons[prevInstanceId].normal, w12), 0.0f);
			if (cosCos <= 0.0f) { discard; }

			float geometryTerm = cosCos / distance2;
			color = (InvPi * uInvPhotonRadius2) * photons[gInstanceId].flux  * uInvNumLightPaths * max((brdf1 * brdf2 * geometryTerm) - uClampingValue, vec3(0.0f)) / (geometryTerm * brdf2);
		}
	}
	else
	{
		color = vec3(0.0f);
	}

}
