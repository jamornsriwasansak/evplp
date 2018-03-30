#version 450 core

#define IsUsablePhoton 2

layout(location = 0) in vec3 vertexPos;

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

uniform float uPhotonRadius;
uniform mat4 uMVP;

out int vInstanceId;
out int vIsValidPhoton;

void main()
{
	vInstanceId = gl_InstanceID;
	vIsValidPhoton = photons[gl_InstanceID].flags & IsUsablePhoton;
	gl_Position = vec4(photons[gl_InstanceID].position + uPhotonRadius * vertexPos, 1.0f);
}
