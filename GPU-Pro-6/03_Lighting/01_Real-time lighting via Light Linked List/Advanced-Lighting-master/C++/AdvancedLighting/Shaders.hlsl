//--------------------------------------------------------------------------------------
// File: RenderScene.hlsl
//
// This is the file containing all shaders
// This sample is based off Microsoft DirectX SDK sample CascadedShadowMap11
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#define __HLSL_SHADER__
 
#include "ShaderUtil.h"

//--------------------------------------------------------------------------------------
// Vertex Shaders
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
VS_OUTPUT_SIMPLE VSMainSimple( VS_INPUT Input )
{
  VS_OUTPUT_SIMPLE Output;
   
  Output.vPosition = TransformPosition( Input.vPosition, g_SimpleWorldViewProj );
  Output.vNormal   = mul( Input.vNormal,  (float3x3)g_SimpleWorld);

  return Output;
}

//--------------------------------------------------------------------------------------------------
VS_OUTPUT_LIGHT VSMainLight(float3 vPosition  : POSITION, uint iid : SV_InstanceID)
{
  VS_OUTPUT_LIGHT output;
  LightInstance   instance   = m_LightInstances[iid];

  // Compute the local position
  float3          local_pos  = vPosition * instance.m_Radius;

  // Compute clip_pos.w
  float           clip_pos_w = (local_pos.x * instance.m_WorldViewProj[0].w) +
                               (local_pos.y * instance.m_WorldViewProj[1].w) + 
                               (local_pos.z * instance.m_WorldViewProj[2].w) +
                                              instance.m_WorldViewProj[3].w;

  // Compute how far we need to push out the vertex to avoid missing the pixel center of a LLL tile 
  // If LLL tile size is 8 pixels then we need to expand the light shell by roughly 5 pixels
  float  push_length  = max( clip_pos_w, 0.0f ) * m_fLightPushScale;
                      
  float3 push_vec     = normalize( vPosition )  * push_length;

  float3 new_position = local_pos + push_vec;

  output.vPosition    = TransformPosition(new_position, instance.m_WorldViewProj);
  
  output.fLightIndex  = instance.m_LightIndex;

  // Done
  return output;
}

//--------------------------------------------------------------------------------------------------
VS_OUTPUT VSMainScene( VS_INPUT Input )
{
    VS_OUTPUT Output;

    Output.vPosition  = mul( Input.vPosition, g_SimpleWorldViewProj );
    Output.vNormal    = mul( Input.vNormal,  (float3x3)g_SimpleWorld);
    Output.vTexcoord  = Input.vTexcoord; 
       
    return Output;    
}

//--------------------------------------------------------------------------------------------------
VS_OUTPUT2D VSMain2D(in VS_INPUT input)
{
  VS_OUTPUT2D output = (VS_OUTPUT2D)0;
  output.vPosition   = float4(input.vPosition.xy * m_vViewportToClip.xy + m_vViewportToClip.zw, input.vPosition.z, 1);
  output.vNormal     = input.vNormal;
  output.vTexcoord   = input.vTexcoord;

  return output;
}
   
//--------------------------------------------------------------------------------------------------
void PSGBuffer(in  VS_OUTPUT input, 
               out float4    normal_w : SV_TARGET0, 
               out float4    color    : SV_TARGET1) 
{
  float4 vDiffuse = g_txDiffuse.Sample( g_samLinear, input.vTexcoord );

  normal_w.xyz    = normalize(input.vNormal);
  normal_w.w      = 1;

  color.rgb       = vDiffuse.rgb;
  color.w         = 1;
}

//--------------------------------------------------------------------------------------------------
float4 PSTexture(in VS_OUTPUT2D input) : SV_TARGET0
{
  return g_txDiffuse.Sample( g_samPoint, input.vTexcoord );
}

//--------------------------------------------------------------------------------------------------
float4 PSLit3D(in  VS_OUTPUT input, in float4 vpos_f : SV_POSITION) : SV_TARGET0
{
  float  ldepth_exp = vpos_f.w;
  float2 screen_uvs = vpos_f.xy * m_vViewportToScreenUVs.xy + m_vViewportToScreenUVs.zw;

  float3 ws_nrm     = normalize(input.vNormal);
  float3 ws_pos;
  float3 vs_pos;

  vs_pos.z                  = 1;
  vs_pos.xy                 = m_vScreenToView.xy + screen_uvs * m_vScreenToView.zw;

  ws_pos                    = TransformPosition(vs_pos * ldepth_exp, m_mViewToWorld).xyz;

  float3 dynamic_diffuse    = 0;      
  float3 dynamic_specular   = 0;
     
  // Evaluate the dynamic lights
  EvaluateDynamicLights(ws_pos, ws_nrm, screen_uvs, ldepth_exp, dynamic_diffuse, dynamic_specular);

  // Evaluate the main light
  float3 main_light         = EvaluateMainLight(ws_pos, ws_nrm);

  // Done
  return float4(main_light + dynamic_diffuse, 0.50f);
}


//--------------------------------------------------------------------------------------------------
float4 PSComposite(in VS_OUTPUT2D input, in float4 vpos_f : SV_POSITION) : SV_TARGET0
{
  int2   vpos_i             = vpos_f.xy;
           
  float2 screen_uvs         = input.vTexcoord;
                             
  float4 normal_xyzw        = g_txNormal[vpos_i];

  // Don't process the skybox or special BRDF elements
  if(normal_xyzw.w == 0)
  {
    discard;
  }

  float4 color              = g_txColor[vpos_i];

  float  hdepth             = g_txDepth[vpos_i].x;
  float  depth_rcp          = rcp(m_vLinearDepthConsts.x - hdepth); 
  float  ldepth_nrm         = m_vLinearDepthConsts.y * depth_rcp;
  float  ldepth_exp         = m_vLinearDepthConsts.z * depth_rcp;
                            
  float3 ws_pos             = ldepth_nrm * input.vNormal + m_vCameraPos.xyz;

  float3 ws_nrm             = normal_xyzw.xyz;
  
  float3 dynamic_diffuse    = 0;      
  float3 dynamic_specular   = 0;
     
  // Evaluate the dynamic lights
  EvaluateDynamicLights(ws_pos, ws_nrm, screen_uvs, ldepth_exp, dynamic_diffuse, dynamic_specular);
 
  // Evaluate the main light
  float3 main_light = EvaluateMainLight(ws_pos, ws_nrm);

  // Done
  return float4((main_light + dynamic_diffuse) * color.rgb, 1);
}

//--------------------------------------------------------------------------------------------------
float PSClearLLLEighth(in VS_OUTPUT2D input) : SV_TARGET0
{
  float2 screen_uvs     = input.vTexcoord;
                        
  uint   dst_index      = ScreenUVsToLLLIndex( screen_uvs );
  uint   dst_offset     = dst_index * 4;
  uint   buffer_size    = m_iLLLWidth * m_iLLLHeight * 4;

  g_LightStartOffsetBuffer.Store( dst_offset, 0xFFFFFFFF);

  // Clear all the bounds buffer layers
  [unroll]
  for(uint idx = 0; idx < MAX_LLL_BLAYERS; ++idx)
  {
    g_LightBoundsBuffer.Store( dst_offset + idx * buffer_size, 0xFFFF77D0);
  }

  float4 d4_max;

  {
    float4 d4_00  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2(-3, -3) );
    float4 d4_01  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2(-1, -3) );
    float4 d4_10  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2(-3, -1) );
    float4 d4_11  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2(-1, -1) );
           d4_max = max(d4_00,  max( d4_01, max( d4_10, d4_11)));
  }

  {
    float4 d4_00  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2(-3,  3) );
    float4 d4_01  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2(-1,  3) );
    float4 d4_10  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2(-3,  1) );
    float4 d4_11  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2(-1,  1) );
           d4_max = max(d4_max, max(d4_00, max( d4_01, max( d4_10, d4_11))));
  }

  {
    float4 d4_00  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2( 3, -3) );
    float4 d4_01  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2( 1, -3) );
    float4 d4_10  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2( 3, -1) );
    float4 d4_11  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2( 1, -1) );
           d4_max = max(d4_max, max(d4_00, max( d4_01, max( d4_10, d4_11))));
  }

  {
    float4 d4_00  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2( 3,  3) );
    float4 d4_01  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2( 1,  3) );
    float4 d4_10  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2( 3,  1) );
    float4 d4_11  = g_txDepth.GatherRed(g_samPoint, screen_uvs, int2( 1,  1) );
           d4_max = max(d4_max, max(d4_00, max( d4_01, max( d4_10, d4_11))));
  }

  // Calculate the final max depth
  float  depth_max    = max(d4_max.x, max( d4_max.y, max( d4_max.z, d4_max.w) ));

  // Convert hyper to linear depth unnormalized
  return m_vLinearDepthConsts.z / (m_vLinearDepthConsts.x - depth_max);
}

//--------------------------------------------------------------------------------------------------
float4 PSDebugLight(in VS_OUTPUT2D input) : SV_TARGET0
{
  uint   src_index      = ScreenUVsToLLLIndex(input.vTexcoord);                     
  uint   first_offset   = g_LightStartOffsetView[ src_index ];     
                     
  float4 debug_color    = float4( 1.0, 1.0, 1.0, 0.0 ); 
  float  num_lights     = 0;      

  // Decode the first element index
  uint   element_index  = (first_offset &  0xFFFFFF);         
  
  // Iterate over the light linked list
  while( element_index != 0xFFFFFF ) 
  {                                                                       
    // Fetch
    LightFragmentLink element  = g_LightFragmentLinkedView[element_index]; 
                                                         
    // Update the next element index
    element_index              = (element.m_IndexNext &  0xFFFFFF);  

    // Increment the light count
    ++num_lights;
  }

  float  coefficient = pow(1.0f - saturate(num_lights/MAX_LINKED_LIGHTS_PER_PIXEL), 4);

  return lerp(float4(1, 0, 0, 1), float4(0, 1, 0, 1), coefficient) * saturate(num_lights);
}

//--------------------------------------------------------------------------------------------------
void AllocateLightFragmentLink(uint dst_offset, uint light_index, float min_d, float max_d)
{
  // Allocate 
  uint    new_lll_idx  = g_LightFragmentLinkedBuffer.IncrementCounter();

  // Don't overflow
  if(new_lll_idx >= m_iLLLMaxAlloc)
  {
    return;
  }

  uint    prev_lll_idx;

  // Get the index of the last linked element stored and refresh it in the process
  g_LightStartOffsetBuffer.InterlockedExchange( dst_offset, new_lll_idx, prev_lll_idx );

  // Encode the light depth values
  uint   light_depth_max  = f32tof16( max_d );// Back  face depth
  uint   light_depth_min  = f32tof16( min_d );// Front face depth

  // Final output
  LightFragmentLink element;

  // Pack the light depth
  element.m_DepthInfo     = (light_depth_min << 16) | light_depth_max;

  // Index/Link
  element.m_IndexNext     = (light_index     << 24) | (prev_lll_idx & 0xFFFFFF);
 
  // Store the element
  g_LightFragmentLinkedBuffer[ new_lll_idx ] = element;
}

//--------------------------------------------------------------------------------------------------
void PSInsertLightNoCulling(in float4 vpos_f      : SV_POSITION,
                            in float  fLightIndex : TEXCOORD0,
                            in bool   front_face  : SV_IsFrontFace )
{
  // Float to unsigned int
  uint2   vpos_i           = vpos_f.xy;
  
  // Incoming light shell linear depth
  float   light_depth      = vpos_f.w; 

  // Light global index
  uint    light_index      = fLightIndex;
  
  // Detect front faces
  if((front_face == true) && ( g_txDepth[vpos_i].x < light_depth))
  {   
    return;
  }  
  
  // Generate an offset based on the light id to allow for instancing
  uint    bbuffer_size     = m_iLLLWidth * m_iLLLHeight * 4;
  uint    bbuffer_idx      = light_index % MAX_LLL_BLAYERS;
  uint    bbuffer_offset   = bbuffer_idx * bbuffer_size;

  // Calculate the ByteAddressBuffer destination offset
  uint    dst_offset       = (vpos_i.y  * m_iLLLWidth + vpos_i.x) << 2; 
  
  // Encode the light index in the upper 16 bits and the linear depth in the lower 16
  uint    new_bounds_info  = (light_index << 16) | f32tof16( light_depth);

  // Load the content that was written by the front faces
  uint    bounds_info;

  g_LightBoundsBuffer.InterlockedExchange( dst_offset + bbuffer_offset, new_bounds_info, bounds_info );
  
  // Decode the stored light index   
  uint    stored_index    = (bounds_info >> 16);
 
  // Decode the stored light depth  
  float   stored_depth    = f16tof32( bounds_info >>  0 );
  
  // Check if both front and back faces were processed
  if(stored_index != light_index)
  {   
    return;
  }
    
  float  front_depth      = min(stored_depth, light_depth);
  float  back_depth       = max(stored_depth, light_depth);

  // Allocate 
  AllocateLightFragmentLink(dst_offset, light_index, front_depth, back_depth);
}

//--------------------------------------------------------------------------------------------------
void PSInsertLightBackFace(in float4 vpos_f      : SV_POSITION,
                           in float  fLightIndex : TEXCOORD0,
                           in bool   front_face  : SV_IsFrontFace )
{
  // Float to unsigned int
  uint2   vpos_i           = vpos_f.xy;
  
  // Incoming light shell linear depth
  float   light_depth      = vpos_f.w; 

  // Light global index
  uint    light_index      = fLightIndex;
   
  // Calculate the ByteAddressBuffer destination offset:
  // Since we don't do instancing in this code path stick with the first layer
  uint    dst_offset       = (vpos_i.y  * m_iLLLWidth + vpos_i.x) << 2; 
       
  // Detect front faces
  if(front_face == true)
  {   
    // Sign will be negative if the light shell is occluded 
    float   depth_test    = sign( g_txDepth[vpos_i].x - light_depth);

    // Encode the light index in the upper 16 bits and the linear depth in the lower 16
    uint    bounds_info   = (light_index << 16) | f32tof16( light_depth * depth_test );

    // Store the front face info
    g_LightBoundsBuffer.Store( dst_offset, bounds_info );
     
    // Only allocate a LightFragmentLink on back faces
    return;
  }

  // Load the content that was written by the front faces
  uint    bounds_info     = g_LightBoundsBuffer.Load( dst_offset );

  // Decode the stored light index   
  uint    stored_index    = (bounds_info >> 16);
 
  // Decode the stored light depth  
  float   front_depth     = f16tof32( bounds_info >>  0 );
  
  // Check if both front and back faces were processed
  if(stored_index == light_index)
  {   
    // Check the case where front faces rendered but were occluded by the scene geometry
    if(front_depth < 0)
    {
      return;      
    }
  }
  // Mismatch, the front face was culled by the near clip
  else
  {
    front_depth = 0;
  }
    
  // Allocate 
  AllocateLightFragmentLink(dst_offset, light_index, front_depth, light_depth);
}