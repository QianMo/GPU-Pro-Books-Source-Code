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

uniform usampler1D bitmask;

varying float mappedZ;

varying out uvec4 result;

void main()
{
   // Set bit in voxel grid
   result = texture1D(bitmask, mappedZ);
}