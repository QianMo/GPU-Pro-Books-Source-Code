#include "RSMFloodFill.inc"

float4 mainVS(float4 pos : POSITION) : SV_Position
{
  return float4(pos.xy, 0, 1);
}

cbuffer Constants : register(b0)
{
  float4x4 g_InvViewProj;
  float4 g_WSOrigin;
  float4 g_CameraPos;
  float4 g_WSStepX;
  float4 g_WSStepY;
  float4 g_LightPos;
};

Texture2D<float>  g_DepthBuffer      : register(t0);
Texture2D<float3> g_GeomNormalBuffer : register(t1);

float4 GetWorldPos(float2 pos, int2 offset)
{
  float z = g_DepthBuffer.Load(int3(pos, 0), offset);
  float4 pppt = mul(g_InvViewProj, float4(pos + float2(offset), z, 1));
  return pppt/pppt.w;
}

struct PSOutput
{
  uint Map        : SV_Target0;
  float4 Position : SV_Target1;
};

PSOutput mainPS(float4 pos : SV_Position)
{
  PSOutput Out;
  float3 G = 2*g_GeomNormalBuffer[pos.xy] - 1;
  float3 worldPos = GetWorldPos(pos, 0);
  float4 plane = float4(G, -dot(G, worldPos));
  Out.Map = 0;
  [unroll] for(uint i=0; i<8; ++i)
  {
    float3 G = 2*g_GeomNormalBuffer.Load(int3(pos.xy, 0), GetSampleOffset(i)) - 1;
    float4 samplePos = GetWorldPos(pos, GetSampleOffset(i));
    float c_DistanceThreshold = 0.05;
    float c_AngleThreshold = 0.95;
    Out.Map |= (((abs(dot(samplePos, plane))<c_DistanceThreshold) && (dot(G, plane)>c_AngleThreshold)) << i);
  }
  
  float3 npp = g_WSOrigin.xyz + pos.x*g_WSStepX.xyz + pos.y*g_WSStepY.xyz;
  float f = dot(G, worldPos - g_CameraPos.xyz);
  float3 p[3];
  [unroll] for(uint i=0; i<3; ++i)
  {
    float2 c_Corner[] = { float2(0, 0), float2(1, 0), float2(0, 1) };
    float3 dir = normalize(npp + c_Corner[i].x*g_WSStepX.xyz + c_Corner[i].y*g_WSStepY.xyz);
    p[i] = (f/dot(G, dir))*dir;
  }
  Out.Position.xyz = worldPos;
  float texelArea = length(cross(p[1] - p[0], p[2] - p[0]));

  float3 d = g_LightPos - worldPos;
  float distSq = dot(d, d);
  float attenuation = saturate(1 - distSq*g_LightPos.a);
  float3 L = normalize(d);
  float formFactor = texelArea*attenuation*saturate(dot(G, L));
  Out.Position.w = formFactor;
  return Out;
}
