#version 430 core

layout(binding = 0) uniform sampler2D image;

layout(location = 0) out vec4 color;

in vec2 texcoord;

void main(void)
{
	
	color = texture2D(image, texcoord);

}
