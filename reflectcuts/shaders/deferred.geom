#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec2 vUv[];

out vec3 gGeomNormal;
out vec2 gUv;
out vec3 gPosition;

uniform mat4 uMVP;
uniform vec2 uJitter;

void main() {
	vec3 edge1 = (gl_in[1].gl_Position - gl_in[0].gl_Position).xyz;
	vec3 edge2 = (gl_in[2].gl_Position - gl_in[0].gl_Position).xyz;
	vec3 geomNormal = normalize(cross(edge1, edge2));
	for(int i = 0; i < 3; i++)
	{
		gl_Position = uMVP * gl_in[i].gl_Position;
		gl_Position.xy += uJitter * gl_Position.w;
		gPosition = gl_in[i].gl_Position.xyz;
		gGeomNormal = geomNormal;
		gUv = vUv[i];
		EmitVertex();
	}
	EndPrimitive();
}
