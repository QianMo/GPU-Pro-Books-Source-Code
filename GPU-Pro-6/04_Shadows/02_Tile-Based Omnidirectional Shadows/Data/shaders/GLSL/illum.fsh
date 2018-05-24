layout(binding = COLOR_TEX_BP) uniform sampler2D albedoGlossMap;
layout(binding = NORMAL_TEX_BP) uniform sampler2D normalMap;
layout(binding = SPECULAR_TEX_BP) uniform sampler2D depthMap;

#if defined(TILED_SHADOW)
layout(binding = CUSTOM0_TEX_BP) uniform sampler2DShadow shadowMap;

SB_LAYOUT(CUSTOM0_SB_BP) buffer LightBuffer 
{
  TiledShadowLight lights[];
} lightBuffer;
#elif defined(CUBE_SHADOW)
layout(binding = CUSTOM0_TEX_BP) uniform samplerCubeArrayShadow shadowMap;

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

SB_LAYOUT(CUSTOM1_SB_BP) buffer LightIndexBuffer 
{
  uint counter;
  uint lightIndices[];
} lightIndexBuffer;

SB_LAYOUT(CUSTOM2_SB_BP) buffer TileInfoBuffer 
{
  TileInfo tileInfos[];
} tileInfoBuffer;

GLOBAL_CAMERA_UB(cameraUB);

UB_LAYOUT(CUSTOM0_UB_BP) uniform CustomUB
{
  vec2 invShadowMapSize;
  uint numLights;
  uint tileSize;
  uint numTilesX;
} customUB;

in VS_Output
{
  vec2 texCoords;
} inputFS;

out vec4 fragColor;

#if defined(TILED_SHADOW) || defined(CUBE_SHADOW)
#define NUM_SAMPLES 16

const vec2 filterKernel[NUM_SAMPLES] =
{
  vec2(-0.94201624, -0.39906216),
  vec2(0.94558609, -0.76890725),
  vec2(-0.094184101, -0.92938870),
  vec2(0.34495938, 0.29387760),
  vec2(-0.91588581, 0.45771432),
  vec2(-0.81544232, -0.87912464),
  vec2(-0.38277543, 0.27676845),
  vec2(0.97484398, 0.75648379),
  vec2(0.44323325, -0.97511554),
  vec2(0.53742981, -0.47373420),
  vec2(-0.26496911, -0.41893023),
  vec2(0.79197514, 0.19090188),
  vec2(-0.24188840, 0.99706507),
  vec2(-0.81409955, 0.91437590),
  vec2(0.19984126, 0.78641367),
  vec2(0.14383161, -0.14100790)
};
#endif

#if defined(TILED_SHADOW)
#define FILTER_RADIUS 1.5

const vec3 faceVectors[4] =
{
  vec3(0.0, -0.57735026, 0.81649661),
  vec3(0.0, -0.57735026, -0.81649661),
  vec3(-0.81649661, 0.57735026, 0.0),
  vec3(0.81649661, 0.57735026, 0.0)
};

uint GetFaceIndex(in vec3 dir)
{
  mat4x3 faceMatrix;
  faceMatrix[0] = faceVectors[0];
  faceMatrix[1] = faceVectors[1];
  faceMatrix[2] = faceVectors[2];
  faceMatrix[3] = faceVectors[3]; 
  vec4 dotProducts = dir*faceMatrix;
  float maximum = max (max(dotProducts.x, dotProducts.y), max(dotProducts.z, dotProducts.w));
  uint index;
  if(maximum == dotProducts.x)
    index = 0;
  else if(maximum == dotProducts.y)
    index = 1;
  else if(maximum == dotProducts.z)
    index = 2;
  else 
    index = 3;
  return index;
}
#endif

#if defined(CUBE_SHADOW)
#define FILTER_RADIUS 4.0

mat3 GetRotation(in vec3 dir)
{
  vec3 up;
  if((abs(dir.x) < 0.0001) && (abs(dir.z) < 0.0001))
  {
    up = (dir.y > 0) ? vec3(0.0, 0.0, -1.0) : vec3(0.0, 0.0, 1.0);
  }
  else
  {
    up = vec3(0.0, 1.0, 0.0);
  }
  vec3 right = normalize(cross(up, dir));
  up = normalize(cross(dir, right));
  return mat3(right, up, -dir);
}
#endif

void main() 
{
  ivec2 screenTC = ivec2(gl_FragCoord.xy);

  // reconstruct world-space position from depth
  float depth = texelFetch(depthMap, screenTC, 0).r; 
  vec4 projPosition = vec4(inputFS.texCoords, depth, 1.0);
  projPosition.xyz = (projPosition.xyz*2.0)-1.0;
  vec4 position = cameraUB.invViewProjMatrix*projPosition;
  position.xyz /= position.w;
  position.w = 1.0;

  uvec2 tilePos = uvec2(gl_FragCoord.xy)/customUB.tileSize;
  uint tileIndex = (tilePos.y*customUB.numTilesX)+tilePos.x;
  const TileInfo tileInfo = tileInfoBuffer.tileInfos[tileIndex];
  const vec3 viewVecN = normalize(cameraUB.position-position.xyz);

  const vec4 albedoGloss = texelFetch(albedoGlossMap, screenTC, 0);
  const vec3 bump = texelFetch(normalMap, screenTC, 0).rgb*2.0-1.0;
  
  vec3 albedo = albedoGloss.rgb;
#ifdef AMD_GPU
  albedo = SrgbToLinear(albedo);
#endif

  vec3 outputColor = albedo*0.1;
  for(uint i=tileInfo.startLightIndex; i<tileInfo.endLightIndex; i++)
  {
    uint lightIndex = lightIndexBuffer.lightIndices[i];

    vec3 lightVec = lightBuffer.lights[lightIndex].position-position.xyz;
    float lightVecLen = length(lightVec);
    float lightRadius = lightBuffer.lights[lightIndex].radius;
    float att = clamp(1.0f-(lightVecLen/lightRadius), 0.0, 1.0);
    vec3 lightVecN = lightVec/lightVecLen;

    vec3 halfVecN = normalize(lightVecN+viewVecN); 
    float nDotL = clamp(dot(lightVecN, bump), 0.0, 1.0);

    // diffuse term
    vec3 diffuseTerm = albedo;

    // simple Blinn-Phong specular term
    const float shininess = 100.0;
    float specular = pow(clamp(dot(halfVecN, bump), 0.0, 1.0), shininess);
    vec3 specularTerm = albedoGloss.aaa*specular;

  #if defined(TILED_SHADOW)
    // shadow term
    uint index = GetFaceIndex(-lightVecN);
    vec4 result = lightBuffer.lights[lightIndex].shadowViewProjTexMatrices[index]*position;
    result.xyz /= result.w;
    result.xyz = (result.xyz*0.5)+0.5;
    float shadowTerm = 0.0;
    const vec2 filterRadius = customUB.invShadowMapSize.xy*FILTER_RADIUS;  
    for(uint i=0; i<NUM_SAMPLES; i++)
    {
      vec3 texCoords;
      texCoords.xy = result.xy+(filterKernel[i]*filterRadius);
      texCoords.z = result.z;
      shadowTerm += texture(shadowMap, texCoords);
    }
    shadowTerm /= NUM_SAMPLES;
  #elif defined(CUBE_SHADOW)
    // shadow term
    const mat3 rotMatrix = GetRotation(-lightVecN);
    const float currentDist = lightVecLen/lightRadius;
    float shadowTerm = 0.0; 
    const vec2 filterRadius = customUB.invShadowMapSize.xy*FILTER_RADIUS;  
    for(uint i=0; i<NUM_SAMPLES; i++)
    {
      vec4 texCoords;
      texCoords.xy = filterKernel[i]*filterRadius;  
      texCoords.z = -sqrt(1.0-dot(texCoords.xy, texCoords.xy));
      texCoords.xyz = rotMatrix*texCoords.xyz;
      texCoords.w = float(lightIndex);
      shadowTerm += texture(shadowMap, texCoords, currentDist);
    }
    shadowTerm /= NUM_SAMPLES;
  #endif
 
    vec3 directIllum = (diffuseTerm+specularTerm)*lightBuffer.lights[lightIndex].color.rgb;

  #if defined(TILED_SHADOW) || defined(CUBE_SHADOW)
    directIllum *= (nDotL*att*shadowTerm);
  #else
    directIllum *= (nDotL*att);
  #endif

    outputColor += directIllum; 
  }

  fragColor = vec4(outputColor, 0.0);
}

