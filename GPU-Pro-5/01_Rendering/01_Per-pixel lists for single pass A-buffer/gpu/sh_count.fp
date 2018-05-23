#version 430
#extension GL_NV_shader_buffer_load       : enable
#extension GL_NV_gpu_shader5              : enable
#extension GL_EXT_shader_image_load_store : enable

// --------------------------------------------

in  vec4 u_Pos;
out vec4 o_PixColor;

uniform uint      u_ScreenSz;
uniform uint32_t *u_Counts;

// --------------------------------------------

void main()
{ 
  uvec2 ij = uvec2(gl_FragCoord.xy);
#ifdef Clear
  u_Counts[ (ij.x + ij.y * u_ScreenSz) ] = 0u;
  o_PixColor = vec4(1,0,0,1);
  if (ij.x == 0 && ij.y == 0) {
    u_Counts[ (u_ScreenSz * u_ScreenSz) ] = 0u;
  }
#else
#ifdef Display
  if (u_Counts[ (ij.x + ij.y * u_ScreenSz) ] == 0) discard;
  float c    = float(u_Counts[ (ij.x + ij.y * u_ScreenSz) ]) / float( /*u_Counts[ (u_ScreenSz * u_ScreenSz) ]*/ 64.0);
  o_PixColor = vec4(c,0,1.0-c,1); 
#else
  atomicAdd( u_Counts + (ij.x + ij.y * u_ScreenSz) , 1u );
  atomicMax( u_Counts + (u_ScreenSz*u_ScreenSz) , u_Counts[ (ij.x + ij.y * u_ScreenSz) ] );
  o_PixColor = vec4(0,0,1,1);
#endif
#endif
} 

// --------------------------------------------
