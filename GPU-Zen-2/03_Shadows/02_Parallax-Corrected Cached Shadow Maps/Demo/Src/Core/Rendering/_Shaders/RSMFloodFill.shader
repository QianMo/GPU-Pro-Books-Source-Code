#include "RSMFloodFill.inc"

struct VSOutput
{
  float4 Position : SV_Position;
  float2 TexCoord : TEXCOORD0;
};

VSOutput mainVS(float4 pos : POSITION)
{
  VSOutput Out;
  Out.Position = float4(pos.xy, 0, 1);
  Out.TexCoord = pos.zw;
  return Out;
}

Texture2D<float> g_SampleID : register(t0);
Texture2D<uint>  g_FloodMap : register(t1);

sampler g_ClampToEdgeSampler : register(s0);

struct PSOutput
{
#if FINAL_PASS
  uint ID : SV_Target;
#else
  float ID : SV_Target;
#endif
};

PSOutput mainPS(VSOutput In)
{
  uint map = g_FloodMap[In.Position.xy];
  float sampleID = g_SampleID[In.Position.xy];
  [unroll] for(uint i=0; i<8; ++i)
  {
    float nbrID = g_SampleID.SampleLevel(g_ClampToEdgeSampler, In.TexCoord, 0, GetSampleOffset(i));
    if(map & (1<<i)) sampleID = min(sampleID, nbrID);
  }
  PSOutput Out;
  Out.ID = sampleID;
  return Out;
}
