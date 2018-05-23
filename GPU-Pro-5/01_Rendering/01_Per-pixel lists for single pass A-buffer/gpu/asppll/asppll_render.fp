#version 430
#extension GL_NV_shader_buffer_load  : enable
#extension GL_NV_gpu_shader5         : enable

// --------------------------------------------

#include "implementations.fp"

// --------------------------------------------

in  vec4  u_Pos;

out vec4  o_PixColor;
layout (depth_any) out float gl_FragDepth;

// --------------------------------------------

uniform float InnerDensity  = 1.0;
uniform float InnerOpacity  = 1.0;
uniform float InnerExponent = 1.5;
uniform vec3  InnerColor    = vec3(0.5,0.5,0.5);
uniform vec3  BkgColor      = vec3(1,1,1);

uniform float ZNear;
uniform float ZFar;

// --------------------------------------------

#define M 64 // Maximum number of fragments for bubble sort

// --------------------------------------------

// Blending equation for in-order traversal

vec4 blend(vec4 clr,vec4 srf)
{
  return clr + (1.0 - clr.w) * vec4(srf.xyz * srf.w , srf.w);  
}

// --------------------------------------------

void main()
{

  vec2  pos      = ( u_Pos.xy * 0.5 + 0.5 ) * float(u_ScreenSz);
	uvec2 ij       = uvec2(pos.xy);
  uint32_t pix   = (ij.x + ij.y * u_ScreenSz);
  
  gl_FragDepth   = 0.0;
  
  vec4  clr      = vec4(0,0,0,0);

#ifdef BubbleSort
  // if using bubble sort, store values in temporary buffer
  uint32_t valdepth[M]; ///// TODO: check for overflow during gather
  uint32_t valrgba [M];
  int num = 0;
#endif
  
#ifdef HABuffer // ===> hash-list

  uint maxage = u_Counts[ Saddr(ij) ];
  if (maxage == 0) discard; // no fragment, early exit
  for (uint a = 1 ; a <= maxage ; a++ ) {
    uvec2     l   = ( ij + u_Offsets[a] );
    uint64_t  h   = Haddr( l % uvec2(u_HashSz) );
    uint64_t  rec = u_Records[ h ];
    uint32_t  key = uint32_t(rec >> uint64_t(32));
    if ( HA_AGE(key) == a ) {
#ifdef BubbleSort
      if (num < M) {
        valdepth[ num ] = HA_DEPTH(key);
        valrgba [ num ] = uint32_t(rec);
        num = num + 1;
      }
#else      
      clr = blend(clr,RGBA(uint32_t(rec)));
#endif
    }
  }

#else           // ===> linked list

#ifndef AsppllCas32
  // Implementation for all linked-list but CAS32 version of asppll
  uint64_t cur = u_Records[ pix ];
  if (cur == 0) discard; // no fragment, early exit
  // walk linked list
  while (cur > 0) {
#ifdef BubbleSort
    if (num < M) {
      valdepth[ num ] = DEPTH(cur);
      valrgba [ num ] = u_RGBA[PTR(cur)];
      num = num + 1;
    }
#else 
    clr = blend( clr , RGBA(u_RGBA[PTR(cur)]) );
#endif
    cur = u_Records[ PTR(cur) ];
  }
#else
  // Implementation for CAS32 version of asppll
  uint32_t *rec32 = (uint32_t*)u_Records;
  uint32_t  cur   = rec32[ (pix<<1u) + 1u ];
  if (cur == 0) discard; // no fragment, early exit
  // walk linked list
  while (cur > 0) {
#ifdef BubbleSort
    // NOTE: this combination is unused
#else 
    clr = blend( clr , RGBA(u_RGBA[cur>>1u]) );
#endif
    cur = rec32[ cur + 1u ];
  }
#endif

#endif // HABuffer

#ifdef BubbleSort
  /// if using bubble sort, all values are in a temporary array
  // -> sort
  for (int i = (num - 2); i >= 0; --i) {
    for (int j = 0; j <= i; ++j) {
      if (valdepth[j] < valdepth[j+1]) {
        uint32_t tmp  = valdepth[j];
        valdepth[j]     = valdepth[j+1];
        valdepth[j+1]   = tmp;
        tmp             = valrgba[j];
        valrgba[j]      = valrgba[j+1];
        valrgba[j+1]    = tmp;
      }
    }
  }
  // -> combine all fragments
  for (int k=0 ; k < num ; k++) {
    uint depth = valdepth[k];
    uint rgba  = valrgba[k];
    clr        = blend(clr,RGBA(rgba));
  }
  // error visualization
  if (num >= M) {
    clr = vec4(1,0,0,1);
  }
#endif // BubbleSort

  // background
  clr = blend(clr,vec4(BkgColor,1.0));
  // done
  o_PixColor = clr;

} 

// --------------------------------------------
