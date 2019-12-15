#if USE_PCF9 && USE_CBM_ARRAY
  #error "incompatible options"
#endif

#ifdef USE_PCF9
  #include "PCF9.inc"
#endif

#define CUBE_SHADOW_FACTOR_BITS 5
#define CUBE_SHADOW_FACTOR_MASK ((1<<CUBE_SHADOW_FACTOR_BITS) - 1)

cbuffer CubeShadowMapData : register(b2)
{
  float4 g_CubeShadowMapData[MAX_LIGHTS_PER_PASS];
};

Texture2D<float3> g_GeomNormalBuffer : register(t3);

#if USE_CBM_ARRAY

TextureCubeArray g_ShadowMap : register(t4);
SamplerComparisonState g_PCFSampler : register(s0);

float GetShadowFactor(uint lightIndex, float3 d, float3 G)
{
  float3 samplingPos = G*length(d) - d;
  float4 cubeSMapData = g_CubeShadowMapData[lightIndex];
  float depth = cubeSMapData.x*dot(samplingPos, samplingPos);
  return g_ShadowMap.SampleCmpLevelZero(g_PCFSampler, float4(samplingPos, cubeSMapData.y), depth);
}

#else //#if USE_CBM_ARRAY

TextureCube<float3> g_CubeIndexTexture : register(t4);
Texture2D g_ShadowMap : register(t5);
sampler g_CubeIndexSampler : register(s0);
SamplerComparisonState g_PCFSampler : register(s1);

float GetShadowFactor(uint lightIndex, float3 d, float3 G)
{
  float3 samplingPos = G*length(d) - d;
  float3 cubeFaceData = g_CubeIndexTexture.SampleLevel(g_CubeIndexSampler, samplingPos, 0);
  float4 cubeSMapData = g_CubeShadowMapData[lightIndex];
  float3 cubeViewPos = samplingPos.zyx;
  if(abs(cubeFaceData.x)==1) cubeViewPos = samplingPos.xzy;
  if(abs(cubeFaceData.x)==2) cubeViewPos = samplingPos.xyz;
  cubeViewPos = asfloat(asuint(cubeViewPos) ^ (asuint(cubeFaceData.xyz) & 0x80000000));
  float rcpZ = rcp(cubeViewPos.z);
  float2 atlasCoord = (cubeViewPos.xy*g_EffectiveFaceSize*rcpZ + abs(cubeFaceData.yz))*g_ShadowMapAtlasSize + cubeSMapData.zw;
  float depth = cubeSMapData.x*rcpZ + cubeSMapData.y;
#if USE_PCF9
  return PCF9(g_ShadowMap, g_PCFSampler, atlasCoord, depth, 1);
#else
  return g_ShadowMap.SampleCmpLevelZero(g_PCFSampler, atlasCoord, depth);
#endif
}

#endif //#if USE_CBM_ARRAY

uint CompressPerQuadShadowFactor(uint lightIndex, uint2 lightBufferGroupID, Texture2D<float> depthBuffer, float4x4 invViewProj, float selfShadowOffset)
{
  float f[4];
  [unroll] for(uint i=0; i<4; ++i)
  {
    uint2 c_SamplingPoint[4] =
    {
      uint2(LIGHTING_QUAD_SIZE - 1, 0), uint2(0, LIGHTING_QUAD_SIZE - 1),
      uint2(LIGHTING_QUAD_SIZE - 1, LIGHTING_QUAD_SIZE - 1), uint2(0, 0),
    };
    uint2 screenCoord = lightBufferGroupID*LIGHTING_QUAD_SIZE + c_SamplingPoint[i];
    float3 G = (2*g_GeomNormalBuffer[screenCoord].xyz - 1)*selfShadowOffset;
    float4 pppt = mul(invViewProj, float4(float2(screenCoord) + 0.5, depthBuffer[screenCoord], 1));
    float3 pointPos = pppt.xyz/pppt.w;
    f[i] = GetShadowFactor(lightIndex, g_PointLights[lightIndex].Position - pointPos, G);
  }
  int4 c_ShiftFactor = int4(1<<(LIGHT_INDEX_BITS + 0*CUBE_SHADOW_FACTOR_BITS),
                            1<<(LIGHT_INDEX_BITS + 1*CUBE_SHADOW_FACTOR_BITS), 
                            1<<(LIGHT_INDEX_BITS + 2*CUBE_SHADOW_FACTOR_BITS), 
                            1<<(LIGHT_INDEX_BITS + 3*CUBE_SHADOW_FACTOR_BITS));
  return dot(int4(float4(f[0], f[1], f[2], f[3])*CUBE_SHADOW_FACTOR_MASK), c_ShiftFactor);
}

float DecompressPerQuadShadowFactor(uint data, uint2 threadID)
{
  int4 c_ShadowFactorMask = int4(CUBE_SHADOW_FACTOR_MASK<<(LIGHT_INDEX_BITS + 0*CUBE_SHADOW_FACTOR_BITS),
                                 CUBE_SHADOW_FACTOR_MASK<<(LIGHT_INDEX_BITS + 1*CUBE_SHADOW_FACTOR_BITS), 
                                 CUBE_SHADOW_FACTOR_MASK<<(LIGHT_INDEX_BITS + 2*CUBE_SHADOW_FACTOR_BITS), 
                                 CUBE_SHADOW_FACTOR_MASK<<(LIGHT_INDEX_BITS + 3*CUBE_SHADOW_FACTOR_BITS));
  float4 c_NormFactor = 1.0/float4(c_ShadowFactorMask);
  float4 shadowFactor = c_NormFactor*(float4)(c_ShadowFactorMask & data);
  float4 f;
  f.xy = (1.0/(LIGHTING_QUAD_SIZE - 1))*float2(threadID);
  f.zw = 1.0 - f.xy;
  return dot(shadowFactor, f.xzxz*f.wyyw);
}
