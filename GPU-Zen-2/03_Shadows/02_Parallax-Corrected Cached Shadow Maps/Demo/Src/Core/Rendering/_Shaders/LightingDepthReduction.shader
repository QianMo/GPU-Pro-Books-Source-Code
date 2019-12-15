#include "Lighting.inc"

float4 mainVS(float4 pos : POSITION) : SV_Position
{
  return float4(pos.xy, 0, 1);
}

cbuffer Constants : register(b0)
{
  float4 g_ProjMatC2;
};

Texture2D<float> g_DepthMap : register(t0);
Texture2D<float3> g_GeomNormalBuffer : register(t1);

float4 mainPS(float4 pos : SV_Position) : SV_Target
{
  float3 n00 = 2.0*g_GeomNormalBuffer.Load(int3((pos.xy - 0.5)*LIGHTING_QUAD_SIZE, 0)) - 1.0;
  float minD = 1, maxD = 0;
  bool bCrease = false;
  for(int i=0; i<LIGHTING_QUAD_SIZE; ++i)
  {
    for(int j=0; j<LIGHTING_QUAD_SIZE; ++j)
    {
      float d = g_DepthMap.Load(int3((pos.xy - 0.5)*LIGHTING_QUAD_SIZE, 0), int2(i, j));
      minD = min(minD, d);
      if(d<1) maxD = max(maxD, d);
      float c_AngleThreshold = 0.8;
      float3 n = 2.0*g_GeomNormalBuffer.Load(int3((pos.xy - 0.5)*LIGHTING_QUAD_SIZE, 0), int2(i, j)) - 1.0;
      if(dot(n, n00)<c_AngleThreshold)
        bCrease = true;
    }
  }
  float2 eyeDepth = g_ProjMatC2.ww/(float2(minD, maxD) - g_ProjMatC2.zz);
  float c_DepthThreshold = 0.3;
  if((eyeDepth.y - eyeDepth.x)>c_DepthThreshold)
    bCrease = true;
  return float4(bCrease ? -eyeDepth : eyeDepth, 0, 0);
}
