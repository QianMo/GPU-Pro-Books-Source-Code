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

uniform usampler1D bitmaskXOR;

varying out uvec4 bitRay;

void main()
{
   bitRay = 
      texelFetch1D(bitmaskXOR, int(gl_FragCoord.x), 0)
    ^ texelFetch1D(bitmaskXOR, int(gl_FragCoord.y)+1, 0);
}