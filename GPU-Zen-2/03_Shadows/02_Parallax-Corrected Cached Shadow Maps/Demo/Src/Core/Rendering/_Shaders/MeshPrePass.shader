#include "GlobalConst.inc"

cbuffer InstanceData : register(b0)
{
  float4x4 g_InstanceTransform[16];
};

struct VSInput
{
  float3 Position : POSITION;
#if VERTEXCOLOR
  float4 Color    : COLOR;
#endif
#if DIFFUSEMAP || NORMALMAP
  float4 TexCoord : TEXCOORD0;
#endif
  float3 Normal   : NORMAL;
#if NORMALMAP
  float4 Tangent  : TANGENT;
#endif
  uint InstanceID : SV_InstanceID;
};

struct VSOutput
{
  float4 Position : SV_Position;
#if VERTEXCOLOR
  float4 Color    : COLOR0;
#endif
#if DIFFUSEMAP || NORMALMAP
  float2 TexCoord : TEXCOORD0;
#endif
  float3 Normal   : TEXCOORD1;
#if NORMALMAP
  float4 Tangent  : TEXCOORD2;
#endif
  float3 WorldPos : TEXCOORD3;
};

VSOutput mainVS(VSInput In)
{
  VSOutput Out;

  float4x4 instanceTransform = g_InstanceTransform[In.InstanceID];
  float3 worldPos = mul(instanceTransform, float4(In.Position, 1));
#if BILLBOARD
  worldPos += In.TexCoord.w*g_CameraUp.xyz - In.TexCoord.z*g_CameraRight.xyz;
#endif
  Out.Position = mul(g_ProjMat, mul(g_ViewMat,float4(worldPos, 1)));
  Out.WorldPos = worldPos;
#if VERTEXCOLOR
  Out.Color = In.Color;
#endif
#if DIFFUSEMAP || NORMALMAP
  Out.TexCoord = In.TexCoord.xy;
#endif
  float3 localNormal = 2*In.Normal - 1;
  Out.Normal = normalize(mul(instanceTransform, float4(localNormal, 0)).xyz);
#if NORMALMAP
  float4 localTangent = 2*In.Tangent - 1;
  Out.Tangent.xyz = normalize(mul(instanceTransform, float4(localTangent.xyz, 0)).xyz);
  Out.Tangent.w = localTangent.w;
#endif
  return Out;
}

struct GSOutput
{
  float4 Position : SV_Position;
#if VERTEXCOLOR
  float4 Color    : COLOR0;
#endif
#if DIFFUSEMAP || NORMALMAP
  float2 TexCoord : TEXCOORD0;
#endif
  float3 Normal   : TEXCOORD1;
#if NORMALMAP
  float4 Tangent  : TEXCOORD2;
#endif
  nointerpolation float3 GeomNormal : TEXCOORD3;
};

[maxvertexcount(3)]
void mainGS(triangle VSOutput In[3], inout TriangleStream<GSOutput> Out)
{
  GSOutput vertex;
  vertex.GeomNormal = normalize(cross(In[1].WorldPos - In[0].WorldPos, In[2].WorldPos - In[0].WorldPos));
  [unroll] for(uint i=0; i<3; ++i)
  {
    vertex.Position = In[i].Position;
#if VERTEXCOLOR
    vertex.Color = In[i].Color;
#endif
#if DIFFUSEMAP || NORMALMAP
    vertex.TexCoord = In[i].TexCoord;
#endif
    vertex.Normal = In[i].Normal;
#if NORMALMAP
    vertex.Tangent = In[i].Tangent;
#endif
    Out.Append(vertex);
  }
  Out.RestartStrip();
}

Texture2D<float4> g_DiffuseMap : register(t0);
Texture2D<float2> g_NormalMap  : register(t1);
Texture2D<float>  g_GlossMap   : register(t2);
Texture2D<float3> g_DetailMap  : register(t3);

sampler g_DiffuseSampler : register(s0);
sampler g_NormalSampler  : register(s1);
sampler g_GlossSampler   : register(s2);
sampler g_DetailSampler  : register(s3);

struct PSOutput
{
  float4 Color      : SV_Target0;
  float4 GeomNormal : SV_Target1;
  float4 Normal     : SV_Target2;
};

PSOutput mainPS(GSOutput In)
{
  float4 color = 1;
#if DIFFUSEMAP
  float4 decal = g_DiffuseMap.Sample(g_DiffuseSampler, In.TexCoord);
  #if ALPHATESTED
    clip(decal.a - 0.5);
  #endif
  color *= decal;
#endif
#if VERTEXCOLOR
  color *= In.Color;
#endif

  float3 normal = In.Normal;
#if NORMALMAP
  float3 Bn = In.Normal;
  float3 Bt = In.Tangent.xyz;
  float3 Bb = In.Tangent.w*cross(Bn, Bt);
  normal.xy = 2*g_NormalMap.Sample(g_NormalSampler, In.TexCoord).yx - 1;
  normal.z = sqrt(saturate(1 - dot(normal.xy, normal.xy)));
  normal = normal.x*Bt + normal.y*Bb + normal.z*Bn;
#endif

  float gloss = 0;
#if SPECULAR
  gloss = 1;
  #if GLOSSMAP
    gloss = g_GlossMap.Sample(g_GlossSampler, In.TexCoord);
  #endif
#endif

  PSOutput Out;
  Out.Color = color;
  Out.Normal = float4(normal*0.5 + 0.5, gloss);
  Out.GeomNormal = float4(In.GeomNormal*0.5 + 0.5, 0);
  return Out;
}
