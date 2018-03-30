#version 450 core

layout(location = 0) out vec3 fLightColor;

uniform vec3 uLightIntensity;

void main()
{
	fLightColor = uLightIntensity;
}
