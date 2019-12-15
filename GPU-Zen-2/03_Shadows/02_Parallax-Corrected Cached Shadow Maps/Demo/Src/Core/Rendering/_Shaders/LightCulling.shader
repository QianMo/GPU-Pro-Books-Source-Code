#include "Lighting.inc"

cbuffer Constants : register(b0)
{
#if SHADOWS_CUBEMAP
  float4x4 g_InvViewProj;
  float2 g_ShadowMapAtlasSize;
  float g_EffectiveFaceSize;
  float g_SelfShadowOffset;
#endif
#if HEMISPHERICAL_LIGHT
  float4x4 g_HPlaneTransform;
  float4 g_OriginVS;
  float2 g_StepVS;
  float g_QuadBRadius;
  float __Unused__;
#endif
  uint g_VisibilityBufferWidthInQuads;
};

bool IsIntersecting(float2 spanA, float2 spanB)
{
  return !(spanA.y<spanB.x || spanA.x>spanB.y);
}

bool IsIntersecting(float4 aabbMinMax /* float4(aabbMin.x, aabbMin.y, aabbMax.x, aabbMax.y) */,
                    float3 circleCenterRadius /* float3(circleCenter.x, circleCenter.y, circleRadius) */)
{
  // Is circle's center inside the bbox?
  if(!any((aabbMinMax.xy>circleCenterRadius.xy) + (circleCenterRadius.xy>aabbMinMax.zw)))
    return true;

  // Is any of bbox's corners inside circle?
  float4 r0 = float4(aabbMinMax.xy - circleCenterRadius.xy, circleCenterRadius.xy - aabbMinMax.zw);
  float4 r1 = r0*r0;
  float4 r5 = r1.xxzz + r1.ywyw;
  if(any(r5 < circleCenterRadius.z*circleCenterRadius.z))
    return true;

  // Is any of bbox's edges intersecting with the circle?
  float2 r2 = aabbMinMax.zw - aabbMinMax.xy;
  float2 a = r2*r2;
  float4 b = r2.xyxy*r0;
  float4 c = r1 + r1.yxwz - circleCenterRadius.z*circleCenterRadius.z;
  float4 d = b*b - a.xyxy*c;
  float4 sqrtd = sqrt(d);
  float2 r3 = rcp(a);
  float4 t0 = ( sqrtd - b)*r3.xyxy;
  float4 t1 = (-sqrtd - b)*r3.xyxy;
  return any(((t0>=0)*(t0<=1) + (t1>=0)*(t1<=1))*(d>=0));
}

#if HEMISPHERICAL_LIGHT

bool IsInHemisphere(float2 screenPos, uint lightIndex, float2 depthBounds)
{
  float4 planeVS = mul(g_HPlaneTransform, g_HemisphericalLightData[lightIndex]);
  [unroll] for(int i=0; i<2; ++i)
  {
    float4 posVS = g_OriginVS;
    posVS.xy += screenPos*g_StepVS;
    posVS.xyz *= depthBounds[i];
    if(dot(planeVS, posVS) > -(g_QuadBRadius*depthBounds[i]))
      return true;
  }
  return false;
}

#endif

RWStructuredBuffer<uint> g_VisibilityBuffer : register(u0);

Texture2D<float2> g_DepthBounds : register(t0);
Texture2D<uint2> g_IndexTexture : register(t1);
StructuredBuffer<float4> g_CullData : register(t2);

#if SHADOWS_CUBEMAP
  Texture2D<float> g_DepthBuffer : register(t6);
  #include "CubeMapShadows.inc"
#endif

groupshared float4 s_ScratchPad[LIGHTING_GROUP_NTHREADS];

[numthreads(LIGHTING_QUAD_SIZE, LIGHTING_QUAD_SIZE, 1)]
void mainCS(uint tindex : SV_GroupIndex, uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
  uint2 lightBufferGroupID = LIGHTING_QUAD_SIZE*gid.xy + tid.xy;

  float4 quadBBox;
  quadBBox.xy = LIGHTING_QUAD_SIZE*float2(lightBufferGroupID);
  quadBBox.zw = quadBBox.xy + LIGHTING_QUAD_SIZE;

  uint NLights = g_IndexTexture[gid.xy].x;
  uint firstIndex = g_IndexTexture[gid.xy].y;
  float2 quadBounds = g_DepthBounds[lightBufferGroupID];

  uint writeIndex = GetLightBufferAddr(lightBufferGroupID, 0, g_VisibilityBufferWidthInQuads);
  uint visibleLights = 0;

#if !TGSM_WORKAROUND
  for(uint i=0; i<NLights; i+=LIGHTING_GROUP_NTHREADS)
  {
    s_ScratchPad[tindex] = g_CullData[firstIndex + i + tindex];
    GroupMemoryBarrierWithGroupSync();
    uint toProcess = min(LIGHTING_GROUP_NTHREADS, NLights - i);
    for(uint j=0; j<toProcess; ++j)
    {
      float4 cullData = s_ScratchPad[j];
#else
  for(uint i=0; i<NLights; ++i)
  {
    float4 cullData = g_CullData[firstIndex + i];
#endif
      float3 BSphere = float3(asint(cullData.xy) >> 16, asint(cullData.x) & 0xffff);
      uint lightIndex = asint(cullData.y) & 0xffff;
      if(IsIntersecting(quadBBox, BSphere) && IsIntersecting(abs(quadBounds), cullData.zw) &&
#if HEMISPHERICAL_LIGHT
         IsInHemisphere(0.5*(quadBBox.xy + quadBBox.zw), lightIndex, abs(quadBounds)) &&
#endif
         visibleLights<MAX_VISIBLE_LIGHTS_PER_QUAD)
      {
        uint dataToWrite = lightIndex;
#if SHADOWS_CUBEMAP
        if(quadBounds.x>0)
        {
          uint shadowFactors = CompressPerQuadShadowFactor(lightIndex, lightBufferGroupID, g_DepthBuffer, g_InvViewProj, g_SelfShadowOffset);
          if(shadowFactors>0)
          {
            dataToWrite |= shadowFactors;
            g_VisibilityBuffer[writeIndex + visibleLights] = dataToWrite;
            ++visibleLights;
          }
        }
        else
        {
          g_VisibilityBuffer[writeIndex + visibleLights] = dataToWrite;
          ++visibleLights;
        }
#else
        g_VisibilityBuffer[writeIndex + visibleLights] = dataToWrite;
        ++visibleLights;
#endif
      }
#if !TGSM_WORKAROUND
    }
    GroupMemoryBarrierWithGroupSync();
#endif
  }

#if SHADOWS_CUBEMAP
  if(quadBounds.x<0)
    visibleLights = -visibleLights;
#endif
  g_VisibilityBuffer[writeIndex + MAX_VISIBLE_LIGHTS_PER_QUAD] = visibleLights;
}
