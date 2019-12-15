#include "Lighting.inc"

float4 mainVS(float4 pos : POSITION) : SV_Position
{
  return float4(pos.xy, 0, 1);
}

cbuffer Constants : register(b0)
{
  float g_FrameBufferWidthQuads;
};

StructuredBuffer<float4> g_LightBuffer : register(t0);
Texture2D<float4> g_DiffuseBuffer : register(t1);

float4 mainPS(float4 pos : SV_Position) : SV_Target
{
  float4 diffuse = pow(g_DiffuseBuffer[pos.xy], 2.2);

  float4 lighting = 0;
#if USE_LIGHTBUFFER
  lighting += g_LightBuffer[GetLightBufferAddr(pos, g_FrameBufferWidthQuads)];
#endif

#if DEBUG_LIGHTING
  return lighting;
#elif DEBUG_DIFFUSE
  return diffuse;
#endif

  return pow(diffuse*lighting, 1.0/2.2);
}
