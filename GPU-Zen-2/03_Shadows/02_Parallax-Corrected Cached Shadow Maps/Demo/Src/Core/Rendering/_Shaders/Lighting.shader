#if USE_PCF9 && FAST_RENDER
  #error "incompatible options"
#endif

#include "Lighting.inc"

cbuffer Constants : register(b0)
{
#if SHADOWS_CUBEMAP
  float2 g_ShadowMapAtlasSize;
  float g_EffectiveFaceSize;
  float g_SelfShadowOffset;
#endif
  float4x4 g_InvViewProj;
  uint g_LightBufferWidthInQuads;
  uint g_VisibilityBufferWidthInQuads;
};

RWStructuredBuffer<float4> g_LightBuffer : register(u0);

StructuredBuffer<uint> g_VisibilityBuffer : register(t0);
Texture2D<float4> g_NormalBuffer : register(t1);
Texture2D<float>  g_DepthBuffer  : register(t2);

#if SHADOWS_CUBEMAP
  #include "CubeMapShadows.inc"
#endif

groupshared uint s_ScratchPad[LIGHTING_GROUP_NTHREADS];

void ComputeLighting(bool perQuadShadows, uint2 threadID, uint nLights, float3 N, float3 pointPos, float3 G,
                     inout float3 diffuse, inout float specular)
{
  for(uint i=0; i<nLights; ++i)
  {
    uint lightIndex = s_ScratchPad[i];
#if SHADOWS_CUBEMAP && FAST_RENDER
    lightIndex &= LIGHT_INDEX_MASK;
#endif

    PointLightShaderData light = g_PointLights[lightIndex];
    float3 d = light.Position - pointPos;
    float distSq = dot(d, d);
    float attenuation = saturate(1 - distSq*light.Position.w);
    float3 L = normalize(d);

    float Kd = attenuation*saturate(dot(N, L));
#if SHADOWS_CUBEMAP
    Kd *= perQuadShadows ? DecompressPerQuadShadowFactor(s_ScratchPad[i], threadID) : GetShadowFactor(lightIndex, d, G);
#endif
#if HEMISPHERICAL_LIGHT
    Kd *= saturate(dot(g_HemisphericalLightData[lightIndex].xyz, -L));
#endif
    diffuse += light.Color*Kd;
  }
}

[numthreads(LIGHTING_QUAD_SIZE, LIGHTING_QUAD_SIZE, 1)]
void mainCS(uint tindex : SV_GroupIndex, uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
  uint2 screenCoord = LIGHTING_QUAD_SIZE*gid.xy + tid.xy;
  float3 N = 2*g_NormalBuffer[screenCoord].xyz - 1;
  float4 pppt = mul(g_InvViewProj, float4(float2(screenCoord) + 0.5, g_DepthBuffer[screenCoord], 1));
  float3 pointPos = pppt.xyz/pppt.w;

  s_ScratchPad[tindex] = g_VisibilityBuffer[GetLightBufferAddr(gid, tindex, g_VisibilityBufferWidthInQuads)];
  GroupMemoryBarrierWithGroupSync();

  int nLights = s_ScratchPad[MAX_VISIBLE_LIGHTS_PER_QUAD];
  float3 diffuse = 0;
  float specular = 0;

#if SHADOWS_CUBEMAP && FAST_RENDER
  bool perQuadShadows = nLights>=0;
  if(perQuadShadows)
  {
    ComputeLighting(true, tid.xy, nLights, N, pointPos, 0, diffuse, specular);
  }
  else
  {
    float3 G = g_SelfShadowOffset*(2*g_GeomNormalBuffer[screenCoord].xyz - 1);
    ComputeLighting(false, 0, -nLights, N, pointPos, G, diffuse, specular);
//    diffuse = float3(1,0,0);
  }
#elif SHADOWS_CUBEMAP
  float3 G = g_SelfShadowOffset*(2*g_GeomNormalBuffer[screenCoord].xyz - 1);
  ComputeLighting(false, 0, nLights, N, pointPos, G, diffuse, specular);
#else
  ComputeLighting(false, tid.xy, nLights, N, pointPos, 0, diffuse, specular);
#endif

  uint writeAddr = GetLightBufferAddr(gid, tindex, g_LightBufferWidthInQuads);
#if BLENDING
  float4 data = g_LightBuffer[writeAddr];
  diffuse += data.xyz;
  specular += data.w;
#endif
  g_LightBuffer[writeAddr] = float4(diffuse, specular);
}
