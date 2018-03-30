#version 450 core

layout(location = 0) in vec3 vertexPos;

out vec2 vUv;

void main()
{
	vUv = vertexPos.xy * 0.5 + vec2(0.5);
	gl_Position = vec4(vertexPos, 1.0);
}
