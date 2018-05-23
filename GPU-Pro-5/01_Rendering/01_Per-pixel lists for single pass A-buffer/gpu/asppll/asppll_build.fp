#version 430
#extension GL_NV_shader_buffer_load  : enable
#extension GL_NV_gpu_shader5         : enable
#extension GL_EXT_shader_image_load_store : enable

// --------------------------------------------

#include "implementations.fp"

// --------------------------------------------

in vec4 v_Pos;
in vec3 v_View;
in vec3 v_Normal;
in vec3 v_Vertex;
in vec2 v_Tex;

out vec4 o_PixColor; 

uniform float u_ZNear;
uniform float u_ZFar;

// uniform uint u_FlipOrient;

// -------------------------------------------- 

#string  ComputeData

// -------------------------------------------- 

void main()
{
  // Detect main buffer overflow

#ifdef Interruptible
  uint32_t count = atomicCounter( u_Counter );
#ifndef HABuffer
#ifndef AllocNaive
  count = count * PG_SIZE;
#endif
#endif
  if (count > (u_NumRecords * 10) >> 4) {
    u_Counts[ u_ScreenSz*u_ScreenSz ] = u_NumRecords;
    discard;
  }
#endif

  // Compute fragment data

  vec2  prj = v_Pos.xy / v_Pos.w;
  vec3  pos = ( 
               vec3(prj * 0.5 + 0.5, 
                    1.0 - (v_Pos.z+u_ZNear)/(u_ZFar+u_ZNear) ) 
              );

  uint32_t data  = computeData();
  uint32_t depth = uint32_t(pos.z * MAX_DEPTH);
	uvec2 pix      = uvec2(pos.xy * u_ScreenSz);

	// Selects the chosen method/variant depending on defines

  bool success = true;

#ifdef HABuffer

#ifdef BubbleSort
  success = insert_postopen( depth, pix, data );
#else
  success = insert_preopen( depth, pix, data );
#endif
#ifdef Interruptible
  atomicCounterIncrement( u_Counter );
#endif

#else

#ifdef BubbleSort
  success = insert_postlin( depth, pix, data );
#else

#ifdef AsppllCas32
  success = insert_prelin_cas32( depth, pix, data );
#else
	success = insert_prelin_max64( depth, pix, data );
#endif

#endif
 o_PixColor = vec4(0,1,0,0);

#endif // HABuffer

  o_PixColor = success ? vec4(0,1,0,0) : vec4(1,0,0,0);
}

// --------------------------------------------
