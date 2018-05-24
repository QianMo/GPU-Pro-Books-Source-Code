SB_LAYOUT(CUSTOM0_SB_BP) buffer MeshInfoBuffer 
{
  MeshInfo infos[];
} meshInfoBuffer;

SB_LAYOUT(CUSTOM1_SB_BP) buffer LightBuffer 
{
  TiledShadowLight lights[];
} lightBuffer;

SB_LAYOUT(3) buffer DrawIndirectCmdBuffer 
{
  uint counter;
  DrawIndirectCmd cmds[];
} drawIndirectCmdBuffer;

SB_LAYOUT(4) buffer ShadowLightIndexBuffer 
{
  uint counter;
  uint lightIndices[];
} shadowLightIndexBuffer;

UB_LAYOUT(CUSTOM0_UB_BP) uniform CustomUB
{
  vec2 invShadowMapSize;
  uint numLights;
  uint tileSize;
  uint numTilesX;
} customUB;

#define MAX_NUM_LIGHTS 1024
#define LOCAL_SIZE_X 256

shared uint groupCounter;
shared uint groupLightIndices[MAX_NUM_LIGHTS];
shared uint startLightIndex;

layout(local_size_x=LOCAL_SIZE_X) in;
void main()
{         
  if(gl_LocalInvocationIndex == 0)
    groupCounter = 0; 
  barrier();
  memoryBarrierShared();

  uint meshIndex = gl_WorkGroupID.x;
  for(uint i=0; i<customUB.numLights; i+=LOCAL_SIZE_X)
  {
    uint lightIndex = gl_LocalInvocationIndex+i;
    if(lightIndex < customUB.numLights)
    {
      vec3 lightPosition = lightBuffer.lights[lightIndex].position;
      float lightRadius = lightBuffer.lights[lightIndex].radius;
      vec3 mins = meshInfoBuffer.infos[meshIndex].mins;
      vec3 maxes = meshInfoBuffer.infos[meshIndex].maxes;
      vec3 distances = max(mins-lightPosition, 0.0) + max(lightPosition-maxes, 0.0);  
      bool boxInSphere = (dot(distances, distances) <= (lightRadius*lightRadius));
      if(boxInSphere)
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
    if(groupCounter > 0)
    {
      uint cmdIndex = atomicAdd(drawIndirectCmdBuffer.counter, 1);
      startLightIndex = atomicAdd(shadowLightIndexBuffer.counter, groupCounter);
      drawIndirectCmdBuffer.cmds[cmdIndex].count = meshInfoBuffer.infos[meshIndex].numIndices;
      drawIndirectCmdBuffer.cmds[cmdIndex].instanceCount = groupCounter;
      drawIndirectCmdBuffer.cmds[cmdIndex].firstIndex = meshInfoBuffer.infos[meshIndex].firstIndex;
      drawIndirectCmdBuffer.cmds[cmdIndex].baseVertex = 0;
      drawIndirectCmdBuffer.cmds[cmdIndex].baseInstance = startLightIndex;
    }
  }
  barrier();
  memoryBarrierShared();

  for(uint i=gl_LocalInvocationIndex; i<groupCounter; i+=LOCAL_SIZE_X)
    shadowLightIndexBuffer.lightIndices[startLightIndex+i] = groupLightIndices[i];
}

