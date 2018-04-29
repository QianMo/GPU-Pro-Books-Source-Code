struct
{
  float4 Data[2];
} c : register(c0);

struct Output
{
  float4 Position : POSITION0;
  float2 TexCoord : TEXCOORD0;
};

float2 ScaleOffset(float2 a, float4 p)
{
  return a*p.xy + p.zw;
}

Output main(float4 In : POSITION0)
{
  Output Out;
  Out.Position = float4(ScaleOffset(In.xy, c.Data[0]), 0, 1);
  Out.TexCoord = ScaleOffset(In.zw, c.Data[1]);
  return Out;
}
