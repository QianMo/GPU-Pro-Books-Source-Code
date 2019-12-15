float PCF9(Texture2D depthTexture, SamplerComparisonState PCFSampler, float2 texCoord, float depth, float kernelSize)
{
  float2 c_TapOffset[9] =
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
  float c_TapWeight[9] =
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

  float texWidth, texHeight;
  depthTexture.GetDimensions(texWidth, texHeight);
  float2 offsetFactor = kernelSize/float2(texWidth, texHeight);
  float shadowFactor = 0;
  [unroll] for(int i=0; i<9; ++i)
  {
    float2 sampleCoord = texCoord + offsetFactor*c_TapOffset[i];
    shadowFactor += c_TapWeight[i]*depthTexture.SampleCmpLevelZero(PCFSampler, sampleCoord, depth);
  }
  return shadowFactor;
}
