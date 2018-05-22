#version 120
#extension GL_EXT_gpu_shader4 : require 

uniform usampler2D voxelTexture;

varying out uvec4 copy;

void main()
{
	copy = texture2D(voxelTexture, gl_TexCoord[0].st);
}