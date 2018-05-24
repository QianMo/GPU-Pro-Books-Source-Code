#version 430 core

layout(binding = 0) uniform sampler2D base;
layout(binding = 1) uniform sampler2D detail_1;
layout(binding = 2) uniform sampler2D detail_2;

layout(location = 0) out vec4 color;

in vec2 texcoord;

uniform float w1;
uniform float w2;

void main(void)
{

	vec4 b0 = texture2D(base, texcoord);
	vec4 d1 = texture2D(detail_1, texcoord);
	vec4 d2 = texture2D(detail_2, texcoord);
	
	vec4 res = b0 + (w1 * d1) + (w2 * d2);
	
	color = res;

}
