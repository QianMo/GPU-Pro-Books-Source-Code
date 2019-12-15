#include "GlobalConst.inc"

#if D3DX_VERSION==0xa2b
#pragma ruledisable 0x0802405f // Otherway it breaks the tesselation with the SDK June 2010
#endif

cbuffer InstanceData : register(b0)
{
  float4x4 g_InstanceTransform[16];
};

struct VSInput
{
  float3 Position : POSITION;
#if ALPHATESTED
  float4 TexCoord : TEXCOORD0;
#endif
  uint InstanceID : SV_InstanceID;
};

struct VSOutput
{
  float4 Position : SV_Position;
#if ALPHATESTED
  float2 TexCoord : TEXCOORD0;
#endif
};

#define HSControlPointOutput VSOutput
#define DSOutput VSOutput

VSOutput mainVS(VSInput In)
{
  VSOutput Out;
  float4x4 instanceTransform = g_InstanceTransform[In.InstanceID];
  float3 worldPos = mul(instanceTransform, float4(In.Position, 1));
#if BILLBOARD
  worldPos += In.TexCoord.w*g_CameraUp.xyz - In.TexCoord.z*g_CameraRight.xyz;
#endif
  Out.Position = mul(g_ViewProjection, float4(worldPos, 1));
#if ALPHATESTED
  Out.TexCoord = In.TexCoord.xy;
#endif
  return Out;
}

cbuffer HullShaderData : register(b0)
{
  float2 g_HalfScreenSize;
  float g_InvDesiredEdgeSize;
};

struct HSConstantOutput
{
  float EdgeTess[3] : SV_TessFactor;
  float InsideTess  : SV_InsideTessFactor;
};

float2 GetScreenPos(float4 pos)
{
  return (pos.xy/pos.w + 1.0)*g_HalfScreenSize;
}

float GetEdgeTessFactor(float2 a, float2 b)
{
  return max(1.0, length(a - b)*g_InvDesiredEdgeSize);
}

HSConstantOutput ConstHS(InputPatch<VSOutput, 3> In, uint PatchID : SV_PrimitiveID)
{
  HSConstantOutput Out;

  float2 p0 = GetScreenPos(In[0].Position);
  float2 p1 = GetScreenPos(In[1].Position);
  float2 p2 = GetScreenPos(In[2].Position);

  float4 TessFactor;
  TessFactor.x = GetEdgeTessFactor(p1, p2);
  TessFactor.y = GetEdgeTessFactor(p2, p0);
  TessFactor.z = GetEdgeTessFactor(p0, p1);
  TessFactor.w = max(TessFactor.x, max(TessFactor.y, TessFactor.z));

  Out.EdgeTess[0] = TessFactor.x;
  Out.EdgeTess[1] = TessFactor.y;
  Out.EdgeTess[2] = TessFactor.z;
  Out.InsideTess = TessFactor.w;
  return Out;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("ConstHS")]
[maxtessfactor(15.0)]
HSControlPointOutput mainHS(InputPatch<VSOutput, 3> In, uint PointID : SV_OutputControlPointID)
{
  HSControlPointOutput Out;
  Out.Position = In[PointID].Position;
#if ALPHATESTED
  Out.TexCoord = In[PointID].TexCoord;
#endif
  return Out;
}

[domain("tri")]
DSOutput mainDS(HSConstantOutput HSConst, float3 Coord : SV_DomainLocation, const OutputPatch<HSControlPointOutput, 3> Patch)
{
  DSOutput Out;
  Out.Position = Coord.x*Patch[0].Position + Coord.y*Patch[1].Position + Coord.z*Patch[2].Position;
#if ALPHATESTED
  Out.TexCoord = Coord.x*Patch[0].TexCoord + Coord.y*Patch[1].TexCoord + Coord.z*Patch[2].TexCoord;
#endif
  return Out;
}

Texture2D<float4> g_DiffuseMap : register(t0);

sampler g_DiffuseSampler : register(s0);

float4 mainPS(DSOutput In) : SV_Target
{
#if ALPHATESTED
  clip(g_DiffuseMap.Sample(g_DiffuseSampler, In.TexCoord).a - 0.5);
#endif

/////////////////////////////////////////
return 1;
////////////////////////////////////////
}
