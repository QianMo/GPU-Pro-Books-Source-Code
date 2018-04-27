//
// ShadowMaps10.1.fx
// Copyright (c) 2008 AMD Corporation. All rights reserved.
// Author: Holger Gruen - GPG ISV relations
//


//
// input/output structres for vertex and pixel shaders
//

// vertex structure for rendering the shadow map and the shadowed scene
struct VSSceneIn
{
    float3 pos          : POSITION;    
    float3 norm         : NORMAL;  
    float2 tex          : TEXTURE0;
};

// input for pixel shader that renders the shadow map
struct PSShadowIn
{
    float4 pos	 : SV_Position;
    float4 color : COLOR0; 
    float2 tex   : TEXTURE0;
};

// input for the pixel shader the renders the shadowed scene
struct PSSceneIn
{
    float4 pos   : SV_Position;
    float4 color : COLOR0; 
    float2 tex   : TEXTURE0;
    float4 smc	 : TEXTURE1;
};

//
// shader constants packed into one constant buffer
//

cbuffer cb1
{
    matrix g_mWorldViewProj;
    matrix g_mWorld;
    float3 g_vLightDir;
    float4 g_vLightColor;
    float4 g_vAmbient;
    matrix g_mWorldViewProjLight;
    float4 g_vShadowMapSize;
};

//
// constants
//
#define SMAP_SIZE g_vShadowMapSize.xy
#define INV_SCALE g_vShadowMapSize.zw 

//
// texture and sampler objects
//

// texture object used for objects and the scene
Texture2D g_txDiffuse;

// sampler for the scene/objects
SamplerState g_samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

// texture object for the 1d depth texture that contains the shadowmap
Texture2D<float> g_txShadowMap;

// pcf comparison sampler for the shadow map - used by the 10.0 code to reduce texture ops 
// for the shadow filter that uses a weight per pcf result
SamplerComparisonState g_samShadowMap
{
    Filter = COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    AddressU = Border;
    AddressV = Border;
    BorderColor = float4(1,1,1,1);
    ComparisonFunc = LESS_EQUAL;
};

// comparison sampler for the shadow map - used by the 10.0 code when weigthing
// each visibility sample with a unique weight. In this case there is no way
// around NxN point samples - 10.1 still gets away with N/2 x N/2 Gathers
SamplerComparisonState g_samCmpPoint
{
    Filter = COMPARISON_MIN_MAG_MIP_POINT;
    AddressU = Border;
    AddressV = Border;
    BorderColor = float4(1,1,1,1);
    ComparisonFunc = LESS_EQUAL;
};

// sampler state used for 10.1 Gather() shadow texture ops
SamplerState g_samPoint
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Border;
    AddressV = Border;
    BorderColor = float4(1,1,1,1);
};

//
// VS for rendering basic textured and lit objects
//

PSSceneIn VSScenemain( VSSceneIn input )
{
    PSSceneIn output = (PSSceneIn)0.0;

    //output our final position in clipspace
    output.pos = mul( float4( input.pos, 1 ), g_mWorldViewProj );
    
    //world space normal
    float3 norm = mul( input.norm, (float3x3)g_mWorld );

    output.color = saturate(dot(normalize(-g_vLightDir),norm))  * (g_vLightColor/15.0f);  
    
    //propogate the texture coordinate
    output.tex = input.tex;
    
    // output position in light space
    output.smc = mul( float4( input.pos, 1 ), g_mWorldViewProjLight );
    
    return output;
}

//
// VS for drawing to the shadow map
//

PSShadowIn VSShadowmain( VSSceneIn input )
{
    PSShadowIn output = (PSShadowIn)0.0;

    //output our final position in clipspace
    output.pos = mul( float4( input.pos, 1 ), g_mWorldViewProjLight );
    
    //world space normal
    float3 norm = mul( input.norm, (float3x3)g_mWorld );

    output.color = saturate(dot(normalize(-g_vLightDir),norm))  * (g_vLightColor/15.0f);  
    
    //propogate the texture coordinate
    output.tex = input.tex;
    
    return output;
}

//
// constants and code for a shadow filter that uses a unique weight
// for each combined 2x2 pcf visbility sample
//

//#define FILTER_SIZE 9
#define GS  ( FILTER_SIZE )
#define GS2 ( FILTER_SIZE / 2 )


// weight matrices that contains a weight for each pcf result of each 2x2
// pixel block of the shadow map

#define SM_FILTER_DISC					 1
#define SM_FILTER_TRIANGLE				 2
#define SM_FILTER_HALFMOON				 3
#define SM_FILTER_GAUSSIAN_LIKE			 4
#define SM_FILTER_UNIFORM				 5

#if FILTER_SIZE == 9

#if FILTER == SM_FILTER_HALFMOON
static const float W[9][9] = 
                 { { 0.2,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0 }, 
			       { 0.0,0.1,1.0,1.0,1.0,1.0,1.0,0.0,0.0 },
			       { 0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0,0.0 },
			       { 0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0 },
			       { 0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,0.5 },
			       { 0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0 },
			       { 0.0,0.0,0.0,0.5,1.0,1.0,1.0,0.0,0.0 },
			       { 0.0,0.1,1.0,1.0,1.0,1.0,0.0,0.0,0.0 },
			       { 0.2,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0 }
			       };
#endif

#if FILTER == SM_FILTER_TRIANGLE
static const float W[9][9] = 
                 { 
			       { 0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0 },
			       { 0.0,0.0,0.0,0.5,1.0,0.5,0.0,0.0,0.0 },
			       { 0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0 },
			       { 0.0,0.0,0.5,1.0,1.0,1.0,0.5,0.0,0.0 },
			       { 0.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0 },
			       { 0.0,0.5,1.0,1.0,1.0,1.0,1.0,0.5,0.0 },
			       { 0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0 },
			       { 0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.5 },
			       { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 }
			       };
#endif

#if FILTER == SM_FILTER_DISC
static const float W[9][9] = 
                 { { 0.0,0.0,0.0,0.5,1.0,0.5,0.0,0.0,0.0 }, 
			       { 0.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0 },
			       { 0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0 },
			       { 0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.5 },
			       { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 },
			       { 0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.5 },
			       { 0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0 },
			       { 0.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0 },
			       { 0.0,0.0,0.0,0.5,1.0,0.5,0.0,0.0,0.0 }
			       };
#endif

#if FILTER == SM_FILTER_UNIFORM
static const float W[9][9] = 
                 { { 1,1,1,1,1,1,1,1,1 }, 
			       { 1,1,1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1,1,1 },
			       };
#endif

#if FILTER == SM_FILTER_GAUSSIAN_LIKE
static const float W[9][9] = 
                 { { 1,2,2,2,2,2,2,2,1 }, 
			       { 2,3,4,4,4,4,4,3,2 },
			       { 2,4,5,6,6,6,5,4,2 },
			       { 2,4,6,7,8,7,6,4,2 },
			       { 2,4,6,8,9,8,6,4,2 },
			       { 2,4,6,7,8,7,6,4,2 },
			       { 2,4,5,6,6,6,5,4,2 },
			       { 2,3,4,4,4,4,4,3,2 },
			       { 1,2,2,2,2,2,2,2,1 }
			       };
#endif

#endif

#if FILTER_SIZE == 7

#if FILTER == SM_FILTER_HALFMOON
static const float W[7][7] = 
                 { { 0.2,1.0,1.0,1.0,0.0,0.0,0.0 }, 
			       { 0.0,0.0,0.5,1.0,1.0,1.0,0.0 },
			       { 0.0,0.0,0.0,1.0,1.0,1.0,0.0 },
			       { 0.0,0.0,0.0,0.5,1.0,1.0,0.5 },
			       { 0.0,0.0,0.0,1.0,1.0,1.0,0.0 },
			       { 0.0,0.0,0.5,1.0,1.0,1.0,0.0 },
			       { 0.2,1.0,1.0,1.0,0.0,0.0,0.0 }
			       };
#endif

#if FILTER == SM_FILTER_TRIANGLE
static const float W[7][7] = 
                 { 
			       { 0.0,0.0,0.0,1.0,0.0,0.0,0.0 },
			       { 0.0,0.0,1.0,1.0,1.0,0.0,0.0 },
			       { 0.0,0.5,1.0,1.0,1.0,0.5,0.0 },
			       { 0.0,1.0,1.0,1.0,1.0,1.0,0.0 },
			       { 0.0,1.0,1.0,1.0,1.0,1.0,0.0 },
			       { 0.5,1.0,1.0,1.0,1.0,1.0,0.5 },
			       { 1.0,1.0,1.0,1.0,1.0,1.0,1.0 }
			       };
#endif

#if FILTER == SM_FILTER_DISC
static const float W[7][7] = 
                 { { 0.0,0.0,0.5,1.0,0.5,0.0,0.0 }, 
			       { 0.0,1.0,1.0,1.0,1.0,1.0,0.0 },
			       { 0.5,1.0,1.0,1.0,1.0,1.0,0.5 },
			       { 1.0,1.0,1.0,1.0,1.0,1.0,1.0 },
			       { 0.5,1.0,1.0,1.0,1.0,1.0,0.5 },
			       { 0.0,1.0,1.0,1.0,1.0,1.0,0.0 },
			       { 0.0,0.0,0.5,1.0,0.5,0.0,0.0 }
			       };
#endif

#if FILTER == SM_FILTER_UNIFORM
static const float W[7][7] = 
                 { { 1,1,1,1,1,1,1 }, 
			       { 1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1 },
			       { 1,1,1,1,1,1,1 },
			       };
#endif

#if FILTER == SM_FILTER_GAUSSIAN_LIKE
static const float W[7][7] = 
                 { { 1,2,2,2,2,2,1 }, 
			       { 2,5,6,6,6,5,2 },
			       { 2,6,7,8,7,6,2 },
			       { 2,6,8,9,8,6,2 },
			       { 2,6,7,8,7,6,2 },
			       { 2,5,6,6,6,5,2 },
			       { 1,2,2,2,2,2,1 }
			       };
#endif

#endif

#if FILTER_SIZE == 5

#if FILTER == SM_FILTER_HALFMOON
static const float W[5][5] = 
                 { { 0.2,1.0,1.0,0.0,0.0 }, 
			       { 0.0,0.0,1.0,1.0,0.0 },
			       { 0.0,0.0,0.5,1.0,0.5 },
			       { 0.0,0.0,1.0,1.0,0.0 },
			       { 0.2,1.0,1.0,0.0,0.0 }
			       };
#endif

#if FILTER == SM_FILTER_TRIANGLE
static const float W[5][5] = 
                 { 
			       { 0.0,0.0,1.0,0.0,0.0 },
			       { 0.0,0.5,1.0,0.5,0.0 },
			       { 0.0,1.0,1.0,1.0,0.0 },
			       { 0.5,1.0,1.0,1.0,0.5 },
			       { 1.0,1.0,1.0,1.0,1.0 }
			       };
#endif

#if FILTER == SM_FILTER_DISC
static const float W[5][5] = 
                 { { 0.0,0.5,1.0,0.5,0.0 }, 
			       { 0.5,1.0,1.0,1.0,0.5 },
			       { 1.0,1.0,1.0,1.0,1.0 },
			       { 0.5,1.0,1.0,1.0,0.5 },
			       { 0.0,0.5,1.0,0.5,0.0 }
			       };
#endif

#if FILTER == SM_FILTER_UNIFORM
static const float W[5][5] = 
                 { { 1,1,1,1,1 }, 
			       { 1,1,1,1,1 },
			       { 1,1,1,1,1 },
			       { 1,1,1,1,1 },
			       { 1,1,1,1,1 },
			       };
#endif

#if FILTER == SM_FILTER_GAUSSIAN_LIKE
static const float W[5][5] = 
                 { { 1,2,2,2,1 }, 
			       { 2,7,8,7,2 },
			       { 2,8,9,8,2 },
			       { 2,7,8,7,2 },
			       { 1,2,2,2,1 }
			       };
#endif

#endif

#if FILTER_SIZE == 3

#if FILTER == SM_FILTER_HALFMOON
static const float W[3][3] = 
                 { { 0.2,1.0,0.0 }, 
			       { 0.0,0.5,1.0 },
			       { 0.2,1.0,0.0 }
			       };
#endif

#if FILTER == SM_FILTER_TRIANGLE
static const float W[3][3] = 
                 { 
			       { 0.0,1.0,0.0 },
			       { 0.5,1.0,0.5 },
			       { 1.0,1.0,1.0 }
			       };
#endif

#if FILTER == SM_FILTER_DISC
static const float W[3][3] = 
                 { { 0.5,1.0,0.5, }, 
			       { 1.0,1.0,1.0, },
			       { 0.5,1.0,0.5, }
			       };
#endif

#if FILTER == SM_FILTER_UNIFORM
static const float W[3][3] = 
                 { { 1,1,1 }, 
			       { 1,1,1 },
			       { 1,1,1 },
			       };
#endif

#if FILTER == SM_FILTER_GAUSSIAN_LIKE
static const float W[3][3] = 
                 { { 1,2,1 }, 
			       { 2,5,2 },
			       { 1,2,1 }
			       };
#endif

#endif


#ifdef DX10_1
// 10.1 shader for one unique weight per pcf sample
// since it uses Gather() only (N/2)x(N/2) texture ops are necessary
// this runs as fast as the uniform or separable filter under 10.0
float shadow_dx10_1( float3 tc )
{
   float4 s = (0.0).xxxx;
   float2 stc = ( SMAP_SIZE * tc.xy ) + float2( 0.5, 0.5 );
   float2 tcs = floor( stc );
   float2 fc;
   int    row;
   int    col;
   float  w = 0.0;
   float4 v1[ GS2 + 1 ];
   float2 v0[ GS2 + 1 ];

   fc.xy = stc - tcs;
   tc.xy = tcs * INV_SCALE;
   
   for( row = 0; row < GS; ++row )
   {
      for( col = 0; col < GS; ++col )
         w += W[row][col];
   }

   // loop over the rows
   for( row = -GS2; row <= GS2; row += 2 )
   {
	   [unroll]for( col = -GS2; col <= GS2; col += 2 )
	   {
	      float fSumOfWeights = W[row+GS2][col+GS2];
	      
	      if( col > -GS2 )
	         fSumOfWeights += W[row+GS2][col+GS2-1];
	      
	      if( col < GS2 )
	         fSumOfWeights += W[row+GS2][col+GS2+1];
	      
	      if( row > -GS2 )
	      {
	         fSumOfWeights += W[row+GS2-1][col+GS2];
	         
	         if( col < GS2 )
	            fSumOfWeights += W[row+GS2-1][col+GS2+1];
	         
	         if( col > -GS2 )
	            fSumOfWeights += W[row+GS2-1][col+GS2-1];
	         
	      }
	      
	      if( fSumOfWeights != 0.0 )
	        v1[(col+GS2)/2] = ( tc.zzzz <= g_txShadowMap.Gather( g_samPoint, tc.xy, int2( col, row ) ) ) ? (1.0).xxxx : (0.0).xxxx; 
	      else
	        v1[(col+GS2)/2] = (0.0f).xxxx;
	        
          if( col == -GS2 )
          {
	         s.x += ( 1 - fc.y ) * ( v1[0].w * ( W[row+GS2][col+GS2] - W[row+GS2][col+GS2] * fc.x ) + 
	                                 v1[0].z * ( fc.x * ( W[row+GS2][col+GS2] - W[row+GS2][col+GS2+1] ) +  W[row+GS2][col+GS2+1] ) );
	         s.y += (     fc.y ) * ( v1[0].x * ( W[row+GS2][col+GS2] - W[row+GS2][col+GS2] * fc.x ) + 
	                                 v1[0].y * ( fc.x * ( W[row+GS2][col+GS2] - W[row+GS2][col+GS2+1] ) +  W[row+GS2][col+GS2+1] ) );
	         if( row > -GS2 )
	         {
		        s.z += ( 1 - fc.y ) * ( v0[0].x * ( W[row+GS2-1][col+GS2] - W[row+GS2-1][col+GS2] * fc.x ) + 
		                                v0[0].y * ( fc.x * ( W[row+GS2-1][col+GS2] - W[row+GS2-1][col+GS2+1] ) +  W[row+GS2-1][col+GS2+1] ) );
		        s.w += (     fc.y ) * ( v1[0].w * ( W[row+GS2-1][col+GS2] - W[row+GS2-1][col+GS2] * fc.x ) + 
		                                v1[0].z * ( fc.x * ( W[row+GS2-1][col+GS2] - W[row+GS2-1][col+GS2+1] ) +  W[row+GS2-1][col+GS2+1] ) );
	         }
          }
          else if( col == GS2 )
          {
	         s.x += ( 1 - fc.y ) * ( v1[GS2].w * ( fc.x * ( W[row+GS2][col+GS2-1] - W[row+GS2][col+GS2] ) + W[row+GS2][col+GS2] ) + 
	                                 v1[GS2].z * fc.x * W[row+GS2][col+GS2] );
	         s.y += (     fc.y ) * ( v1[GS2].x * ( fc.x * ( W[row+GS2][col+GS2-1] - W[row+GS2][col+GS2] ) + W[row+GS2][col+GS2] ) + 
	                                 v1[GS2].y * fc.x * W[row+GS2][col+GS2] );
	         if( row > -GS2 )
	         {
		        s.z += ( 1 - fc.y ) * ( v0[GS2].x * ( fc.x * ( W[row+GS2-1][col+GS2-1] - W[row+GS2-1][col+GS2] ) + W[row+GS2-1][col+GS2] ) + 
		                                v0[GS2].y * fc.x * W[row+GS2-1][col+GS2] );
		        s.w += (     fc.y ) * ( v1[GS2].w * ( fc.x * ( W[row+GS2-1][col+GS2-1] - W[row+GS2-1][col+GS2] ) + W[row+GS2-1][col+GS2] ) + 
		                                v1[GS2].z * fc.x * W[row+GS2-1][col+GS2] );
	         }
          }
          else
          {
	         s.x += ( 1 - fc.y ) * ( v1[(col+GS2)/2].w * ( fc.x * ( W[row+GS2][col+GS2-1] - W[row+GS2][col+GS2+0] ) + W[row+GS2][col+GS2+0] ) +
                                     v1[(col+GS2)/2].z * ( fc.x * ( W[row+GS2][col+GS2-0] - W[row+GS2][col+GS2+1] ) + W[row+GS2][col+GS2+1] ) );
 	         s.y += (     fc.y ) * ( v1[(col+GS2)/2].x * ( fc.x * ( W[row+GS2][col+GS2-1] - W[row+GS2][col+GS2+0] ) + W[row+GS2][col+GS2+0] ) +
		                             v1[(col+GS2)/2].y * ( fc.x * ( W[row+GS2][col+GS2-0] - W[row+GS2][col+GS2+1] ) + W[row+GS2][col+GS2+1] ) );
	         if( row > -GS2 )
	         {
		        s.z += ( 1 - fc.y ) * ( v0[(col+GS2)/2].x * ( fc.x * ( W[row+GS2-1][col+GS2-1] - W[row+GS2-1][col+GS2+0] ) + W[row+GS2-1][col+GS2+0] ) +
		                                v0[(col+GS2)/2].y * ( fc.x * ( W[row+GS2-1][col+GS2-0] - W[row+GS2-1][col+GS2+1] ) + W[row+GS2-1][col+GS2+1] ) );
		        s.w += (     fc.y ) * ( v1[(col+GS2)/2].w * ( fc.x * ( W[row+GS2-1][col+GS2-1] - W[row+GS2-1][col+GS2+0] ) + W[row+GS2-1][col+GS2+0] ) +
		                                v1[(col+GS2)/2].z * ( fc.x * ( W[row+GS2-1][col+GS2-0] - W[row+GS2-1][col+GS2+1] ) + W[row+GS2-1][col+GS2+1] ) );
	         }
          }
		    
		  if( row != GS2 )
			v0[(col+GS2)/2] = v1[(col+GS2)/2].xy;
	   }
   }
  
   return dot(s,(1.0).xxxx)/w;
}

#endif

// 10.0 shader for one unique weight per pcf sample - this shader makes use of
// shifted texture coords and post weights to reduce the texture op counts for dx10.0
// without this trick a naive implementation would need (N-1)x(N-1) pcf samples
// this shaders only does (N/2)x(N-1) pcf samples instead
float shadow_dx10_0( float3 tc )
{
   float  s   = 0.0;
   float2 stc = ( SMAP_SIZE * tc.xy ) + float2( 0.5, 0.5 );
   float2 tcs = floor( stc );
   float2 fc;
   int    row;
   int    col;
   float  w = 0.0;

   fc     = stc - tcs;
   tc.xy  = tc - ( fc * INV_SCALE );
   fc.y  *= INV_SCALE;

   for( row = 0; row < GS; ++row )
   {
      for( col = 0; col < GS; ++col )
         w += W[row][col];
   }

   for( row = 0; row < GS; ++row )
   {
      [unroll]for( col = -GS2; col <= GS2; col += 2 )
	  {
	    if( col == -GS2 )
	    {
			if( W[row][col+GS2+1] != 0 ||  W[row][col+GS2] != 0 )
				s += ( ( 1.0 - fc.x ) * W[row][col+GS2+1] + W[row][col+GS2] ) * g_txShadowMap.SampleCmpLevelZero( g_samShadowMap, tc.xy + float2( g_vShadowMapSize.z * ( ( W[row][col+GS2+1] - fc.x * ( W[row][col+GS2+1] - W[row][col+GS2] ) ) / ( ( 1.0 - fc.x ) * W[row][col+GS2+1] + W[row][col+GS2] ) ), fc.y ), tc.z, int2( col, row - GS2 ) ).x;
		}
	    else if( col == GS2 )
	    {
			if( W[row][col+GS2-1] != 0 ||  W[row][col+GS2] != 0 )
				s += ( fc.x * W[row][col+GS2-1] + W[row][col+GS2] ) * g_txShadowMap.SampleCmpLevelZero( g_samShadowMap, tc.xy + float2( g_vShadowMapSize.z * ( ( fc.x * W[row][col+GS2] ) / ( fc.x  * W[row][col+GS2-1] + W[row][col+GS2] ) ), fc.y ), tc.z, int2( col, row - GS2 ) ).x;
		}
	    else
	    {
			if( ( W[row][col+GS2-1] - W[row][col+GS2+1] ) != 0 || ( W[row][col+GS2] + W[row][col+GS2+1] ) != 0 )
				s += ( fc.x * ( W[row][col+GS2-1] - W[row][col+GS2+1] ) + W[row][col+GS2] + W[row][col+GS2+1] ) * g_txShadowMap.SampleCmpLevelZero( g_samShadowMap, tc.xy + float2( g_vShadowMapSize.z * ( ( W[row][col+GS2+1] - fc.x * ( W[row][col+GS2+1] - W[row][col+GS2] ) ) / ( fc.x * ( W[row][col+GS2-1] - W[row][col+GS2+1] ) + W[row][col+GS2] + W[row][col+GS2+1] ) ), fc.y ), tc.z, int2( col, row - GS2 ) ).x;
		}
	  }
   }	

   return s/w;
}

//
// defines for the different filters
//
#define FILTER_DX10			0
#define FILTER_DX10_1		1

//
// pixel shader for rendering the shadowed scene
//
float4 PSScenemain(PSSceneIn input, uniform int nShadowMappingMethod) : SV_Target
{   
    float4 diffuse = g_txDiffuse.Sample( g_samLinear, input.tex );
   
    //transform from RT space to texture space.
    float2 ShadowTexC = ( (0.5 * input.smc.xy) / input.smc.w ) + float2( 0.5, 0.5 );
    ShadowTexC.y = 1.0f - ShadowTexC.y;
    float3 tc = float3( ShadowTexC, ( input.smc.z / input.smc.w ) - 0.001 );
    
    float fShadow = 0;
    
    if (nShadowMappingMethod==FILTER_DX10)
    {
		fShadow = shadow_dx10_0( tc );
	}
#ifdef DX10_1
	if (nShadowMappingMethod==FILTER_DX10_1)
	{
		fShadow = shadow_dx10_1( tc );
	}
#endif

	// Light equation
    float4 col = ( g_vAmbient + fShadow * input.color ) * diffuse;
    
    return col;
}

//
// render states
//

// dss for rendering the shadowed scene
DepthStencilState DSS_RenderShadowedScene
{
    DepthEnable = true;
    DepthWriteMask = ALL;
    DepthFunc = Less_Equal;
    
    StencilEnable = false;
    StencilReadMask = 0xFFFFFFFF;
    StencilWriteMask = 0x0;
    
    FrontFaceStencilFunc = Always;
    FrontFaceStencilPass = Keep;
    FrontFaceStencilFail = Keep;
    
    BackFaceStencilFunc = Always;
    BackFaceStencilPass = Keep;
    BackFaceStencilFail = Keep;
};

// dss for rendering the shadow map
DepthStencilState DSS_RenderShadowMap
{
    DepthEnable = true;
    DepthWriteMask = ALL;
    DepthFunc = Less_Equal;
    
    StencilEnable = false;
    StencilReadMask = 0xFFFFFFFF;
    StencilWriteMask = 0x0;
    
    FrontFaceStencilFunc = Always;
    FrontFaceStencilPass = Keep;
    FrontFaceStencilFail = Keep;
    
    BackFaceStencilFunc = Always;
    BackFaceStencilPass = Keep;
    BackFaceStencilFail = Keep;
};

// rs to enable culling
RasterizerState EnableCulling
{
    CullMode = BACK;
    MultisampleEnable = true;
};

// blend state to disable the frame buffer - used when rendering the shadow map
BlendState DisableFrameBuffer
{
    BlendEnable[0] = FALSE;
    RenderTargetWriteMask[0] = 0x0;
    RenderTargetWriteMask[1] = 0x0;
};

// blend state to enable the frame buffer - used when rendering the shadowed scene
BlendState EnableFrameBuffer
{
    BlendEnable[0] = FALSE;
    RenderTargetWriteMask[0] = 0x0F;
    RenderTargetWriteMask[1] = 0x0F;
};

//
// RenderShadowMap - renders the shadow map
//
technique10 RenderShadowMap
{
    pass p0
    {
        SetVertexShader( CompileShader( vs_4_0, VSShadowmain() ) );
        SetGeometryShader( NULL );
        SetPixelShader( NULL );
        SetDepthStencilState( DSS_RenderShadowMap, 0 ); //state, stencilref
        SetBlendState( DisableFrameBuffer, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( EnableCulling );
    }  
}

//
// RenderShadowedScene - renders the shadowed scene
//

// 10.0 version for a weight per pcf result
technique10 RenderShadowedScene10_0
{
    pass p0
    {
        SetVertexShader( CompileShader( vs_4_0, VSScenemain() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PSScenemain(FILTER_DX10) ) );
        
        SetDepthStencilState( DSS_RenderShadowedScene, 0 ); //state, stencilref
        SetBlendState( EnableFrameBuffer, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( EnableCulling );
    }  
}

#ifdef DX10_1
// 10.1 version for a weight per pcf result
technique10 RenderShadowedScene10_1
{
    pass p0
    {
        SetVertexShader( CompileShader( vs_4_1, VSScenemain() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_1, PSScenemain(FILTER_DX10_1) ) );
        
        SetDepthStencilState( DSS_RenderShadowedScene, 0 ); //state, stencilref
        SetBlendState( EnableFrameBuffer, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( EnableCulling );
    }  
}
#endif

