///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#version 120
#extension GL_EXT_gpu_shader4 : require 

varying out uvec4 result;

uniform usampler2D voxelTexture;
uniform int level; // read from this mipmap level

uniform float inverseTexSize; // 1.0 / mipmapLevelResolution

void main()
{
   // texture coordinates of 4 neighbor texels in source voxel texture

	vec2 offset1 = vec2(inverseTexSize, 0.0);   // right
	vec2 offset2 = vec2(0.0, inverseTexSize);   // top
	vec2 offset3 = vec2(inverseTexSize, inverseTexSize); // top right
	vec2 coord; // this pixel
	coord.x = (((gl_FragCoord.x-0.5)*2.0)+0.5)*inverseTexSize;
	coord.y = (((gl_FragCoord.y-0.5)*2.0)+0.5)*inverseTexSize;

	// Lookup 4 neighbor texels ( ~ voxel stacks)
	uvec4 val1 = texture2DLod(voxelTexture, coord, level);
	uvec4 val2 = texture2DLod(voxelTexture, coord+offset1, level);
	uvec4 val3 = texture2DLod(voxelTexture, coord+offset2, level);
	uvec4 val4 = texture2DLod(voxelTexture, coord+offset3, level);

	result = val1 | val2 | val3 | val4;
}