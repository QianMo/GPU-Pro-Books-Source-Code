#include "GlobalConst.inc"

#if ASM_LAYER
  #include "ASMLayerShaderData.inc"
#endif

#if CUBEMAP
  #include "CubeMapGlobalConst.inc"
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
#if CUBEMAPS_ARRAY
  float3 NormDistanceVec : TEXCOORD1;
#endif
#if ASM_LAYER
  float3 TileCoord : TEXCOORD2;
#endif
};

VSOutput mainVS(VSInput In)
{
  VSOutput Out;
  float4x4 instanceTransform = g_InstanceTransform[In.InstanceID];
  float3 worldPos = mul(instanceTransform, float4(In.Position, 1)).xyz;
#if BILLBOARD
  worldPos += In.TexCoord.w*g_CameraUp.xyz - In.TexCoord.z*g_CameraRight.xyz;
#endif
#if CUBEMAP
  Out.Position = float4(worldPos, 1);
  #if CUBEMAPS_ARRAY
    Out.NormDistanceVec = g_InvViewRange*(worldPos - g_CameraPos.xyz);
  #endif
#else
  Out.Position = mul(g_ViewProjection, float4(worldPos, 1));
#endif
#if ALPHATESTED
  Out.TexCoord = In.TexCoord.xy;
#endif
#if ASM_LAYER
  float3 indexCoord = mul( g_ASMIndirectionTexMat, float4( worldPos, 1.0 ) ).xyz;
  Out.TileCoord = GetASMTileCoord( indexCoord, g_ASMTileData, g_ASMOneOverDepthAtlasSize );
#endif
  return Out;
}

#if CUBEMAP

struct GSOutput
{
  float4 Position : SV_Position;
#if ALPHATESTED
  float2 TexCoord : TEXCOORD0;
#endif
#if CUBEMAPS_ARRAY
  float3 NormDistanceVec : TEXCOORD1;
  uint SliceIndex : SV_RenderTargetArrayIndex;
#else
  uint FaceIndex  : SV_ViewportArrayIndex;
#endif
#if ASM_LAYER
  float3 TileCoord : TEXCOORD2;
#endif
};

[maxvertexcount(18)]
void mainGS(triangle VSOutput In[3], inout TriangleStream<GSOutput> Out)
{
  float3 a = In[1].Position.xyz - In[0].Position.xyz;
  float3 b = In[2].Position.xyz - In[0].Position.xyz;
  float3 c = g_CameraPos.xyz - In[0].Position.xyz;
  if(dot(cross(a, b), c)>0)
  {
    [unroll] for(uint i=0; i<6; ++i)
    {
      GSOutput vertex;
#if CUBEMAPS_ARRAY
      vertex.SliceIndex = g_FirstSliceIndex + i;
#else
      vertex.FaceIndex = i;
#endif
      [unroll] for(uint j=0; j<3; ++j)
      {
        vertex.Position = mul(g_CubeMapViewProjection[i], In[j].Position);
#if ALPHATESTED
        vertex.TexCoord = In[j].TexCoord;
#endif
#if CUBEMAPS_ARRAY
        vertex.NormDistanceVec = In[j].NormDistanceVec;
#endif
#if ASM_LAYER
        vertex.TileCoord = In[j].TileCoord;
#endif
        Out.Append(vertex);
      }
      Out.RestartStrip();
    }
  }
}

#endif //#if CUBEMAP

Texture2D<float4> g_DiffuseMap : register(t0);
sampler g_DiffuseSampler : register(s0);

#if ASM_LAYER
  Texture2D<float> g_ASMDepthExtentMapAtlasTexture : register(t10);
  sampler g_ASMDepthExtentMapAtlasSampler : register(s10);
#endif

#if CUBEMAP
  #define PSInput GSOutput
#else
  #define PSInput VSOutput
#endif

#if CUBEMAPS_ARRAY
  struct PSOutput { float DistanceSq : SV_Depth; };
#else
  #define PSOutput void
#endif

PSOutput mainPS(PSInput In)
{
#if ASM_LAYER
  float demDepth = g_ASMDepthExtentMapAtlasTexture.SampleLevel( g_ASMDepthExtentMapAtlasSampler, In.TileCoord.xy, 0 );
  if( In.TileCoord.z < ( demDepth + 0.005 ) )
    discard;
#endif
#if ALPHATESTED
  clip(g_DiffuseMap.Sample(g_DiffuseSampler, In.TexCoord).a - 0.5);
#endif
#if CUBEMAPS_ARRAY
  PSOutput Out;
  Out.DistanceSq = dot(In.NormDistanceVec, In.NormDistanceVec);
  return Out;
#endif
}
