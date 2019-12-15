cbuffer CubeMapGlobalConst : register(b2)
{
  float4x4 g_CubeMapViewProjection[6];
};

cbuffer CubeMapsArrayConfig : register(b1)
{
  unsigned int g_FirstSliceIndex;
  float g_InvViewRange;
};
