#version 450 core

out vec3 color;

uniform samplerBuffer uVplTextureBuffer;
uniform float uVplScale;

uniform sampler2D uPhotonTexture;
uniform float uPhotonScale;

uniform sampler2D uLightTexture;
uniform float uLightScale;

uniform int uScreenWidth;
uniform bool uDoGammaCorrection;

in vec2 vUv;

void main()
{
	vec3 result = vec3(0);
	ivec2 screenCoord = ivec2(gl_FragCoord.xy);
	vec3 vplColor = texelFetch(uVplTextureBuffer, screenCoord.y * uScreenWidth + screenCoord.x).xyz * uVplScale;
	vec3 pmColor = texture(uPhotonTexture, vUv).xyz * uPhotonScale;
	vec3 lightColor = texture(uLightTexture, vUv).xyz * uLightScale;
	vec3 sum = step(lightColor.x, 0.0) * (vplColor + pmColor) + lightColor;
	if (uDoGammaCorrection)
	{
		color = pow(sum, vec3(1 / 2.2));
	}
	else
	{
		color = sum;
	}
}
