struct
{
  float4x4 WorldMat;
  float4x4 ViewProj;
  float4 LightTexRect;
} c : register(c0);

struct Input
{
  float4 Position : POSITION0;
#if ALPHATEST
  float2 TexCoord : TEXCOORD0;
#endif
};

struct Output
{
  float4 Position  : POSITION0;
  float4 SMapCoord : TEXCOORD0;
#if ALPHATEST
  float2 TexCoord  : TEXCOORD1;
#endif
};

float2 ScaleOffset(float2 a, float4 p)
{
  return a*p.xy + p.zw;
}

Output main(Input In)
{
  Output Out;
  Out.Position = mul(c.ViewProj, mul(c.WorldMat, In.Position));

  Out.SMapCoord.x = 0.5*(1.0 + Out.Position.x);
  Out.SMapCoord.y = 0.5*(1.0 - Out.Position.y);
  Out.SMapCoord.xy = ScaleOffset(Out.SMapCoord.xy, c.LightTexRect);
  Out.SMapCoord.zw = Out.Position.zw;

#if ALPHATEST
  Out.TexCoord = In.TexCoord;
#endif
  return Out;
}
