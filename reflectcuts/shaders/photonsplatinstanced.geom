#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec2 vScreenUv[];
in int vInstanceId[];
flat in int vIsValidPhoton[];

out vec3 gPosition;
out vec2 gScreenUv;
flat out int gInstanceId;

uniform mat4 uMVP;

void main()
{
	for(int i = 0; i < 3; i++)
	{
		if (vIsValidPhoton[0] != 0)
		{
			gPosition = gl_in[i].gl_Position.xyz;
			gl_Position = uMVP * gl_in[i].gl_Position;

			vec2 ndc = gl_Position.xy / gl_Position.w;
			gScreenUv  = ndc.xy * 0.5 + 0.5; //ndc is -1 to 1 in GL. scale for 0 to 1
			gInstanceId = vInstanceId[i];
			EmitVertex();
		}
	}
	EndPrimitive();
}
