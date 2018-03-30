#version 450 core

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec2 uv;

out vec2 vUv;

uniform mat4 uMVP;

void main()
{
    vUv = uv;
	gl_Position = vec4(vertexPos, 1.0);
}
