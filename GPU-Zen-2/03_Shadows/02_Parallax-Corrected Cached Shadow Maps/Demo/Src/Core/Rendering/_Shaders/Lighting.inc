#define LIGHTING_QUAD_SIZE           8
#define LIGHTING_GROUP_NTHREADS      (LIGHTING_QUAD_SIZE*LIGHTING_QUAD_SIZE)
#define MAX_VISIBLE_LIGHTS_PER_QUAD  (LIGHTING_GROUP_NTHREADS - 1)

struct PointLightShaderData
{
  float4 Position;
  float4 Color;
};

#define MAX_CB_SIZE         65536
#define MAX_LIGHTS_PER_PASS 2048 // must be less than MAX_CB_SIZE/sizeof(PointLightShaderData)
#define LIGHT_INDEX_BITS    11
#define LIGHT_INDEX_MASK    ((1<<LIGHT_INDEX_BITS) - 1)

cbuffer PointLightsShaderData : register(b1)
{
  PointLightShaderData g_PointLights[MAX_LIGHTS_PER_PASS];
};

cbuffer HemisphericalLightData : register(b3)
{
  float4 g_HemisphericalLightData[MAX_LIGHTS_PER_PASS];
};

#ifndef __cplusplus

uint GetLightBufferAddr(uint2 groupID, uint threadIndex, uint bufferWidthInQuads)
{
  return (groupID.y*bufferWidthInQuads + groupID.x)*LIGHTING_GROUP_NTHREADS + threadIndex;
}

float GetLightBufferAddr(float2 screenCoord, float bufferWidthInQuads) // float is enough for resolutions up to 2048x2048
{
  float2 groupID = floor((1.0/LIGHTING_QUAD_SIZE)*screenCoord);
  float2 threadID = floor(screenCoord - groupID*LIGHTING_QUAD_SIZE);
  float threadIndex = threadID.y*LIGHTING_QUAD_SIZE + threadID.x;
  return (groupID.y*bufferWidthInQuads + groupID.x)*LIGHTING_GROUP_NTHREADS + threadIndex;
}

#endif //#ifndef __cplusplus
