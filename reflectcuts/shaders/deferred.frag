#version 450 core

layout(location = 0) out vec4 fPosition;
layout(location = 1) out vec3 fNormal;
layout(location = 2) out vec3 fDiffuse;
layout(location = 3) out vec4 fPhongReflectance;

in vec2 gUv;
in vec3 gPosition;
in vec3 gGeomNormal;

uniform sampler2D uLambertReflectance;
uniform sampler2D uPhongReflectance;
uniform sampler2D uPhongExponent;

void main()
{
	fPosition = vec4(gPosition, 1.0f);
	fNormal = gGeomNormal;
	fDiffuse = texture(uLambertReflectance, gUv).xyz;
	fPhongReflectance = vec4(texture(uPhongReflectance, gUv).xyz, texture(uPhongExponent, gUv).x);
}
