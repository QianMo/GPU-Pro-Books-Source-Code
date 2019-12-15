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

Texture2D<float> g_DepthTexture : register(t0);
sampler g_Sampler : register(s0);

float mainPS(VSOutput In) : SV_Depth
{
  return g_DepthTexture.SampleLevel(g_Sampler, In.TexCoord, 0);
}
