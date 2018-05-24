layout(binding = COLOR_TEX_BP) uniform sampler2D depthMap;

#if defined(TILED_SHADOW)
SB_LAYOUT(CUSTOM0_SB_BP) buffer LightBuffer 
{
  TiledShadowLight lights[];
} lightBuffer;
#elif defined(CUBE_SHADOW)
SB_LAYOUT(CUSTOM0_SB_BP) buffer LightBuffer 
{
  CubeShadowLight lights[];
} lightBuffer;
#else
SB_LAYOUT(CUSTOM0_SB_BP) buffer LightBuffer 
{
  Light lights[];
} lightBuffer;
#endif

SB_LAYOUT(3) buffer LightIndexBuffer 
{
  uint counter;
  uint lightIndices[];
} lightIndexBuffer;

SB_LAYOUT(4) buffer TileInfoBuffer 
{
  TileInfo tileInfos[];
} tileInfoBuffer;

UB_LAYOUT(CUSTOM0_UB_BP) uniform CustomUB
{
  vec2 invShadowMapSize;
  int numLights;
} customUB;

#define MAX_NUM_LIGHTS 1024
#define LOCAL_SIZE_X 16
#define LOCAL_SIZE_Y 16
#define GROUP_SIZE (LOCAL_SIZE_X*LOCAL_SIZE_Y)

shared uint iDepthMin;
shared uint iDepthMax;
shared vec3 tileMins;
shared vec3 tileMaxes;
shared uint groupCounter;
shared uint groupLightIndices[MAX_NUM_LIGHTS];
shared uint startLightIndex;

layout(local_size_x=LOCAL_SIZE_X, local_size_y=LOCAL_SIZE_Y) in;
void main() 
{         
  float depth = texelFetch(depthMap, ivec2(gl_GlobalInvocationID.xy), 0).r;
  if(gl_LocalInvocationIndex == 0)
  {
    iDepthMin = 0xffffffff;
    iDepthMax = 0;
    groupCounter = 0;
  }
  barrier();
  memoryBarrierShared(); 

  uint iDepth = floatBitsToUint(depth);
  atomicMin(iDepthMin, iDepth);
  atomicMax(iDepthMax, iDepth);
  barrier();
  memoryBarrierShared();

  if(gl_LocalInvocationIndex == 0)
  {
    vec2 tileSize = 1.0/vec2(gl_NumWorkGroups.xy);
    tileMins.xy = tileSize*vec2(gl_WorkGroupID.xy);
    tileMins.z = uintBitsToFloat(iDepthMin);
    tileMaxes.xy = tileMins.xy+tileSize;
    tileMaxes.z = uintBitsToFloat(iDepthMax);
  }
  barrier();
  memoryBarrierShared();

  for(uint i=0; i<customUB.numLights; i+=GROUP_SIZE)
  {
    uint lightIndex = gl_LocalInvocationIndex+i;
    if(lightIndex < customUB.numLights)
    {
      vec3 lightMins = lightBuffer.lights[lightIndex].mins.xyz;
      vec3 lightMaxes = lightBuffer.lights[lightIndex].maxes.xyz;
      bool isInside = true;
      if((tileMins.x > lightMaxes.x) || (lightMins.x > tileMaxes.x) ||
         (tileMins.y > lightMaxes.y) || (lightMins.y > tileMaxes.y) ||
         (tileMins.z > lightMaxes.z) || (lightMins.z > tileMaxes.z)) 
      {
        isInside = false;
      }
      if(isInside)
      {
        uint index = atomicAdd(groupCounter, 1);
        groupLightIndices[index] = lightIndex;
      }
    }
  }
  barrier();
  memoryBarrierShared(); 

  if(gl_LocalInvocationIndex == 0)
  {
    startLightIndex = atomicAdd(lightIndexBuffer.counter, groupCounter);
    TileInfo tileInfo;
    tileInfo.startLightIndex = startLightIndex;
    tileInfo.endLightIndex = startLightIndex+groupCounter;
    uint tileInfoIndex = (gl_WorkGroupID.y*gl_NumWorkGroups.x)+gl_WorkGroupID.x;
    tileInfoBuffer.tileInfos[tileInfoIndex] = tileInfo;
  }
  barrier();
  memoryBarrierShared();

  for(uint i=gl_LocalInvocationIndex; i<groupCounter; i+=GROUP_SIZE)
    lightIndexBuffer.lightIndices[startLightIndex+i] = groupLightIndices[i];
}

