#version 120
#extension GL_EXT_gpu_shader4 : enable 

uniform usampler1D bitmaskXOR; 
uniform float z1;
uniform float z2;

uniform usampler1D bitmaskOR; 
uniform float z;

uniform bool getXORRay;

uniform usampler2D bitmaskXORRays;


varying out uvec4 bitRay;

void main()
{

   if(getXORRay)
   {
      //bitRay = (texture1D(bitmaskXOR, min(z1, z2))
      //   ^ texture1D(bitmaskXOR, max(z1, z2)+0.0078125/*1.0/128.0*/));
      bitRay = texture2D(bitmaskXORRays, vec2(min(z1, z2), max(z1, z2)));
   }
   else
   {
      bitRay = texture1D(bitmaskOR, z);
   }
}
