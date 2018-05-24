#version 430 core

layout(binding = 0) uniform sampler2D image_0;
layout(binding = 1) uniform sampler2D image_1;
layout(binding = 2) uniform sampler2D image_2;

layout(location = 0) out vec4 color;

in vec2 texCoord_0;
in vec2 texCoord_1;
in vec2 texCoord_2;

void main(void)
{

	vec4 c0, c1, c2;

	if (texCoord_0.s < 0.11 || texCoord_0.s > 0.89) c0 = vec4(0.0);
	else c0 = texture2D(image_0, texCoord_0);

	if (texCoord_1.s < 0.11 || texCoord_1.s > 0.89) c1 = vec4(0.0);
	else c1 = 5.0f * texture2D(image_1, texCoord_1) + vec4(0.5,0.5,0.5,1.0);
	
	if (texCoord_2.s < 0.11 || texCoord_2.s > 0.89) c2 = vec4(0.0);
	else c2 = 5.0f * texture2D(image_2, texCoord_2) + vec4(0.5,0.5,0.5,1.0);

	color = c0 + c1 + c2;

}
