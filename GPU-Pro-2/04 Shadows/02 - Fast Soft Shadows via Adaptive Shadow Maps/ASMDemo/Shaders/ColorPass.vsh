struct
{
  float4x4 WorldMat;
  float4x4 ViewProj;
} c : register(c0);

struct Input
{
  float4 Position : POSITION0;
  float3 Normal   : NORMAL0;
#if ALPHATEST
  float2 TexCoord : TEXCOORD0;
#endif
};

struct Output
{
  float4 Position : POSITION0;
  float4 WorldPos : TEXCOORD0;
  float3 Normal   : TEXCOORD1;
#if ALPHATEST
  float2 TexCoord : TEXCOORD2;
#endif
};

Output main(Input In)
{
  Output Out;
  Out.WorldPos = mul(c.WorldMat, In.Position);
  Out.Position = mul(c.ViewProj, Out.WorldPos);
  Out.Normal   = mul(c.WorldMat, float4(In.Normal, 0));
#if ALPHATEST
  Out.TexCoord = In.TexCoord;
#endif
  return Out;
}
