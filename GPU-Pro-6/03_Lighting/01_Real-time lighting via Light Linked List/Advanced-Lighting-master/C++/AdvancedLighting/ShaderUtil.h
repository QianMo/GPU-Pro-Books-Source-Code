 #include "ShaderShared.h"

static const float3 vLightDir1 = float3( -1.0f,  1.0f, -1.0f ); 
static const float3 vLightDir2 = float3(  1.0f,  1.0f, -1.0f ); 
static const float3 vLightDir3 = float3(  0.0f, -1.0f,  0.0f );
static const float3 vLightDir4 = float3(  1.0f,  1.0f,  1.0f ); 

//--------------------------------------------------------------------------------------
// Textures 
//--------------------------------------------------------------------------------------
Texture2D              g_txDiffuse          : T_REGISTER( TEX_DIFFUSE);

Texture2D              g_txDepth            : T_REGISTER( TEX_DEPTH  );
Texture2D              g_txNormal           : T_REGISTER( TEX_NRM    );
Texture2D              g_txColor            : T_REGISTER( TEX_COL    );

Texture2D              g_txShadow           : T_REGISTER( TEX_SHADOW );

//--------------------------------------------------------------------------------------
// Samplers 
//--------------------------------------------------------------------------------------
SamplerState           g_samLinear          : S_REGISTER( SAM_LINEAR );
SamplerState           g_samPoint           : S_REGISTER( SAM_POINT  );
SamplerComparisonState g_samShadow          : S_REGISTER( SAM_SHADOW );

struct LightFragmentLink
{
  uint m_DepthInfo; // High bits min depth, low bits max depth
  uint m_IndexNext; // Light index and link to the next fragment 
};
globallycoherent RWStructuredBuffer< LightFragmentLink >  g_LightFragmentLinkedBuffer   : U_REGISTER( UAV_LIGHT_LINKED );
globallycoherent RWByteAddressBuffer                      g_LightStartOffsetBuffer      : U_REGISTER( UAV_LIGHT_OFFSET );
globallycoherent RWByteAddressBuffer                      g_LightBoundsBuffer           : U_REGISTER( UAV_LIGHT_BOUNDS );

StructuredBuffer< LightFragmentLink >                     g_LightFragmentLinkedView     : T_REGISTER( SRV_LIGHT_LINKED );
Buffer<uint>                                              g_LightStartOffsetView        : T_REGISTER( SRV_LIGHT_OFFSET ); 

StructuredBuffer<GPULightEnv>                             g_LightEnvs                   : T_REGISTER( SRV_LIGHT_ENV    );

//--------------------------------------------------------------------------------------
// Input / Output structures
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
    float4 vPosition  : POSITION;
    float3 vNormal    : NORMAL;
    float2 vTexcoord  : TEXCOORD0;
};

struct VS_OUTPUT
{
    float4 vPosition  : SV_POSITION; 
    float3 vNormal    : NORMAL;
    float2 vTexcoord  : TEXCOORD0; 
};

struct VS_OUTPUT2D
{
    float4 vPosition    : SV_POSITION;
    float3 vNormal      : NORMAL;
    float2 vTexcoord    : TEXCOORD0;
};

struct VS_OUTPUT_SIMPLE
{
  float4 vPosition    : SV_POSITION;
  float3 vNormal      : NORMAL;
};

struct VS_OUTPUT_LIGHT
{
  float4 vPosition    : SV_POSITION;
  float  fLightIndex  : TEXCOORD0;
};

//--------------------------------------------------------------------------------------------------
float4  TransformPosition(float3 position, float4x4 transform)
{
  return (position.xxxx * transform[0] + (position.yyyy * transform[1] + (position.zzzz * transform[2] + transform[3])));
}

//--------------------------------------------------------------------------------------------------
float4  TransformPosition(float4 position, float4x4 transform)
{
  return TransformPosition(position.xyz, transform);
}

//--------------------------------------------------------------------------------------------------
uint    ScreenUVsToLLLIndex(float2 screen_uvs)
{
  uint   x_unorm = saturate(screen_uvs.x) * m_iLLLWidth;
  uint   y_unorm = saturate(screen_uvs.y) * m_iLLLHeight;

  return y_unorm * m_iLLLWidth + x_unorm;
}

//--------------------------------------------------------------------------------------
void ComputeCoordinatesTransform(in int        iCascadeIndex, 
                                 in out float4 vShadowTexCoord , 
                                 in out float4 vShadowTexCoordViewSpace ) 
{     
  vShadowTexCoord.x *= m_fShadowPartitionSize;  // precomputed (float)iCascadeIndex / (float)CASCADE_CNT
  vShadowTexCoord.x += (m_fShadowPartitionSize * (float)iCascadeIndex ); 
} 

//--------------------------------------------------------------------------------------
// Use PCF to sample the depth map and return a percent lit value.
//--------------------------------------------------------------------------------------
void CalculatePCFPercentLit (in  float4 vShadowTexCoord, 
                             in  float fRightTexelDepthDelta, 
                             in  float fUpTexelDepthDelta, 
                             in  float fBlurRowSize,
                             out float fPercentLit
                             ) 
{
  fPercentLit = 0.0f;
  // This loop could be unrolled, and texture immediate offsets could be used if the kernel size were fixed.
  // This would be performance improvement.
  for( int x = m_iPCFBlurForLoopStart; x < m_iPCFBlurForLoopEnd; ++x ) 
  {
    for( int y = m_iPCFBlurForLoopStart; y < m_iPCFBlurForLoopEnd; ++y ) 
    {
      float depthcompare = vShadowTexCoord.z;
      // A very simple solution to the depth bias problems of PCF is to use an offset.
      // Unfortunately, too much offset can lead to Peter-panning (shadows near the base of object disappear )
      // Too little offset can lead to shadow acne ( objects that should not be in shadow are partially self shadowed ).
      depthcompare -= m_fShadowBiasFromGUI;

      // Compare the transformed pixel depth to the depth read from the map.
      fPercentLit += g_txShadow.SampleCmpLevelZero( g_samShadow, 
        float2( 
        vShadowTexCoord.x + ( ( (float) x ) * m_fNativeTexelSizeInX ) , 
        vShadowTexCoord.y + ( ( (float) y ) * m_fTexelSize ) 
        ), 
        depthcompare );
    }
  }
  fPercentLit /= (float)fBlurRowSize;
}

//--------------------------------------------------------------------------------------------------
float PhysicalFalloff(float3 v)
{
  // |v| ranges from 0 to 1 over the light's radius 
  float r_sqd = saturate(dot(v,v));
  return 1.0f/r_sqd - 2.0f + r_sqd;
}

//--------------------------------------------------------------------------------------------------
void EvaluatePunctualLight(in    GPULightEnv  light,
                           
                           in    float3       ws_pos,
                           in    float3       ws_norm, 
                                                                       
                           inout float3       diffuse,
                           inout float3       specular)
{ 
  // Normal N, view vector V and light vector L
  float3 N                = ws_norm; 
  float3 L_unrm           = light.m_WorldPos - ws_pos;
  float3 pos_rel          = L_unrm * rcp(light.m_Radius); 
  float3 L                = normalize(L_unrm); 
                          
  float  NL_              = dot(N, L);
  float  NL_front         = saturate( NL_ );                       
                          
  float dist_falloff_lin  = saturate( 1.0f - dot( L, pos_rel ));
  //float light_falloff     = pow(dist_falloff_lin, 2);
  float light_falloff     = PhysicalFalloff( pos_rel );
  
  diffuse                += light.m_LinearColor * light_falloff * NL_front;
  specular                = 0;
}

//--------------------------------------------------------------------------------------------------
void EvaluateDynamicLights(in     float3 ws_pos,
                           in     float3 ws_nrm,
                           in     float2 screen_uvs,
                           in     float  ldepth_exp,
                           inout  float3 dynamic_diffuse,
                           inout  float3 dynamic_specular)
{
  uint   src_index        = ScreenUVsToLLLIndex(screen_uvs);                     
  uint   first_offset     = g_LightStartOffsetView[ src_index ];      

  // Decode the first element index
  uint   element_index    = (first_offset &  0xFFFFFF);         

  // Iterate over the light linked list
  while( element_index != 0xFFFFFF ) 
  {                                                                       
    // Fetch
    LightFragmentLink element  = g_LightFragmentLinkedView[element_index]; 

    // Update the next element index
    element_index              = (element.m_IndexNext &  0xFFFFFF); 

    float light_depth_max      = f16tof32(element.m_DepthInfo >>  0);
    float light_depth_min      = f16tof32(element.m_DepthInfo >> 16);

    // Do depth bounds check 
    if( (ldepth_exp > light_depth_max) || (ldepth_exp < light_depth_min) )
    {
      continue;
    } 

    // Decode the light index
    uint          light_idx   = (element.m_IndexNext >>     24);

    // Access the light environment                
    GPULightEnv   light_env   = g_LightEnvs[ light_idx ];

    EvaluatePunctualLight(light_env, ws_pos, ws_nrm, dynamic_diffuse, dynamic_specular);
  }

  // Done
}

//--------------------------------------------------------------------------------------------------
float3 EvaluateMainLight(in float3 ws_pos,
                         in float3 ws_nrm)
{
  float4 vShadowMapTextureCoordViewSpace = TransformPosition(ws_pos, m_mShadow );

  float4 vShadowMapTextureCoord          = 0.0f;
  float4 vShadowMapTextureCoord_blend    = 0.0f;
  
  float4 vVisualizeCascadeColor = float4(0.0f,0.0f,0.0f,1.0f);
  
  float  fPercentLit = 0.0f;
  float  fPercentLit_blend = 0.0f;
  
  float  fUpTextDepthWeight=0;
  float  fRightTextDepthWeight=0;
  float  fUpTextDepthWeight_blend=0;
  float  fRightTextDepthWeight_blend=0;
  
  int    iBlurRowSize  = m_iPCFBlurForLoopEnd - m_iPCFBlurForLoopStart;
         iBlurRowSize *= iBlurRowSize;
  float  fBlurRowSize  = (float)iBlurRowSize;
      
  int    iCascadeFound        = 0;     
  int    iCurrentCascadeIndex = 0;
    
  for( int iCascadeIndex = 0; iCascadeIndex < CASCADE_COUNT_FLAG && iCascadeFound == 0; ++iCascadeIndex ) 
  {
      vShadowMapTextureCoord = vShadowMapTextureCoordViewSpace * m_vCascadeScale[iCascadeIndex];
      vShadowMapTextureCoord += m_vCascadeOffset[iCascadeIndex];
  
      if ( min( vShadowMapTextureCoord.x, vShadowMapTextureCoord.y ) > m_fMinBorderPadding
        && max( vShadowMapTextureCoord.x, vShadowMapTextureCoord.y ) < m_fMaxBorderPadding )
      { 
          iCurrentCascadeIndex = iCascadeIndex;   
          iCascadeFound        = 1; 
      }
  }
      
  float  fBlendBetweenCascadesAmount     = 1.0f;
  float  fCurrentPixelsBlendBandLocation = 1.0f;
       
  ComputeCoordinatesTransform( iCurrentCascadeIndex,   vShadowMapTextureCoord,  vShadowMapTextureCoordViewSpace );     
  CalculatePCFPercentLit (     vShadowMapTextureCoord, fRightTextDepthWeight,   fUpTextDepthWeight, fBlurRowSize, fPercentLit );
      
  // Some ambient-like lighting.
  float3 fLighting         = saturate( dot( vLightDir1 , ws_nrm ) )*0.05f +
                             saturate( dot( vLightDir2 , ws_nrm ) )*0.05f +
                             saturate( dot( vLightDir3 , ws_nrm ) )*0.05f +
                             saturate( dot( vLightDir4 , ws_nrm ) )*0.05f ;
                         
  float3 vShadowLighting    = fLighting * 0.5f;
  fLighting                += saturate( dot(m_vLightDir, ws_nrm ) ) * float3(0.411764741f, 0.411764741f, 0.411764741f);
                           
  fLighting                 = lerp( vShadowLighting, fLighting, fPercentLit );
  
  // Done
  return fLighting;
}