SB_LAYOUT(CUSTOM0_SB_BP) buffer LightBuffer 
{
  CubeShadowLight lights[];
} lightBuffer;

UB_LAYOUT(CUSTOM0_UB_BP) uniform Custom0UB
{
  int lightIndex;
} custom0UB;

in GS_Output
{
  vec3 lightVec;
  vec3 viewDirection;
} inputFS;

#define BIAS_SCALE_FACTOR 0.024

void main()
{
  float lightVecLen = length(inputFS.lightVec);
  vec3 lightVecN = inputFS.lightVec/lightVecLen;
  float bias = (1.0-dot(inputFS.viewDirection, -lightVecN))*BIAS_SCALE_FACTOR;
  gl_FragDepth = (lightVecLen/lightBuffer.lights[custom0UB.lightIndex].radius)+bias;
}

