#define PCF_TAPS 9

half PCF(sampler2D ShadowmapSMP, float3 SMCoord, half KernelSize, float4 ShadowmapSizeParams)
{
  float2 c_TapOffset[PCF_TAPS] =
  {
    float2( 0.0000000, 0.0000000),
    float2( 1.2000000, 0.0000000),
    float2(-1.2000000, 0.0000000),
    float2( 0.0000000, 1.2000000),
    float2( 0.0000000,-1.2000000),
    float2( 0.8400000, 0.8400000),
    float2(-0.8400000, 0.8400000),
    float2(-0.8400000,-0.8400000),
    float2( 0.8400000,-0.8400000),
  };
  half c_TapWeight[PCF_TAPS] =
  {
    0.120892,
    0.110858,
    0.110858,
    0.110858,
    0.110858,
    0.111050,
    0.111050,
    0.111050,
    0.111050,
  };

  float4 Coord = float4(SMCoord, 0);
  half ShadowFactor = 0;
  [unroll] for(int z=0; z<PCF_TAPS; ++z)
  {
    Coord.xy = SMCoord.xy + ShadowmapSizeParams.xy*c_TapOffset[z]*KernelSize;
    half Sample = tex2Dlod(ShadowmapSMP, Coord).r;
    ShadowFactor += c_TapWeight[z]*Sample;
  }
  return ShadowFactor;
}
