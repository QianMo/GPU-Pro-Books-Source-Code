SB_LAYOUT(CUSTOM0_SB_BP) buffer LightBuffer 
{
  CubeShadowLight lights[];
} lightBuffer;

UB_LAYOUT(CUSTOM0_UB_BP) uniform Custom0UB
{
  int lightIndex;
} custom0UB;

UB_LAYOUT(CUSTOM1_UB_BP) uniform Custom1UB
{
  int faceIndex;
} custom1UB;

out GS_Output
{
  vec3 lightVec;
  vec3 viewDirection;
} outputGS;

const vec3 viewDirections[6] = 
{
  vec3(1.0, 0.0, 0.0),
  vec3(-1.0, 0.0, 0.0),
  vec3(0.0, 1.0, 0.0),
  vec3(0.0, -1.0, 0.0),
  vec3(0.0, 0.0, 1.0),
  vec3(0.0, 0.0, -1.0)
};

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
void main()
{
  gl_Layer = (custom0UB.lightIndex*6)+custom1UB.faceIndex;
  mat4 shadowViewProjMatrix = lightBuffer.lights[custom0UB.lightIndex].shadowViewProjMatrices[custom1UB.faceIndex];
  vec3 lightPosition = lightBuffer.lights[custom0UB.lightIndex].position;
  vec3 viewDirection = viewDirections[custom1UB.faceIndex];
  for(uint i=0; i<3; i++)
  {
    gl_Position = shadowViewProjMatrix*gl_in[i].gl_Position;
    outputGS.lightVec = gl_in[i].gl_Position.xyz-lightPosition;
    outputGS.viewDirection = viewDirection;
    EmitVertex(); 
  }
  EndPrimitive(); 
}

