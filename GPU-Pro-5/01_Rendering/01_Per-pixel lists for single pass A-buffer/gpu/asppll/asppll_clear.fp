#version 430
#extension GL_NV_shader_buffer_load       : enable
#extension GL_NV_gpu_shader5              : enable
#extension GL_EXT_shader_image_load_store : enable

// --------------------------------------------

#include "implementations.fp"

// --------------------------------------------

in  vec4 u_Pos;
out vec4 o_PixColor;

// --------------------------------------------

void main()
{ 
  uvec2 ij = uvec2(gl_FragCoord.xy);

#ifndef HABuffer

  // Linked-lists
  // initialize heads
  u_Records[ (ij.x + ij.y * u_ScreenSz) ] = uint64_t(0);
  
#ifdef AllocNaive
  // Naive
  // nothing to do
#else 
  // Paged
  // initialize per-page counts
  u_Counts [ (ij.x + ij.y * u_ScreenSz) ] = 0;
#endif

#else // HABuffer

  // Hashed-lists
  // clear all records
  for (uint o = (ij.x + ij.y * u_ScreenSz) ; o < u_NumRecords ; o += (u_ScreenSz*u_ScreenSz) ) {
    u_Records[ o ] = uint64_t(0);
  }
  // clear max age table (max age = 0)
  u_Counts [ (ij.x + ij.y * u_ScreenSz) ] = 0u;

#endif // HABuffer

#ifdef Interruptible  
  if (ij.x == 0 && ij.y == 0) {
    u_Counts [ u_ScreenSz * u_ScreenSz ] = uint32_t(0);
  }
#endif

  o_PixColor = vec4(0,0,1,1);
  
} 

// --------------------------------------------
