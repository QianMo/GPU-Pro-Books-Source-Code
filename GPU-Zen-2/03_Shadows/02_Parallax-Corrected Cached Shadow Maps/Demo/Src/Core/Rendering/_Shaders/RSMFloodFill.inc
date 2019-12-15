int2 GetSampleOffset(uint i)
{
  int2 c_Offset[] =
  {
    float2(-1, 1), float2( 0, 1), float2( 1, 1),
    float2(-1, 0),                float2( 1, 0),
    float2(-1,-1), float2( 0,-1), float2( 1,-1),
  };
  return c_Offset[i];
}
